import unittest
import torch
import numpy as np
from inpaint_cropandstitch import CPUProcessorLogic


class TestCPUProcessorLogic(unittest.TestCase):
    def setUp(self):
        self.processor = CPUProcessorLogic()

    def test_rescale_i_algorithms(self):
        img = torch.rand(1, 32, 32, 3, dtype=torch.float32)
        algorithms = ["bicubic", "bilinear", "nearest", "nearest-exact", "lanczos", "box", "area", "hamming"]
        for algo in algorithms:
            with self.subTest(algorithm=algo):
                res = self.processor.rescale_i(img, 64, 48, algo)
                self.assertEqual(res.shape, (1, 48, 64, 3))
                self.assertEqual(res.dtype, torch.float32)

    def test_rescale_i_channels_and_batches(self):
        for c in [1, 3, 4]:
            with self.subTest(channels=c):
                img = torch.rand(2, 20, 20, c)
                res = self.processor.rescale_i(img, 40, 30, "bilinear")
                self.assertEqual(res.shape, (2, 30, 40, c))

    def test_rescale_i_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                img = torch.rand(1, 16, 16, 3, dtype=dtype)
                res = self.processor.rescale_i(img, 32, 32, "bicubic")
                self.assertEqual(res.shape, (1, 32, 32, 3))
                self.assertIn(res.dtype, [torch.float32, dtype])

    def test_rescale_i_boundary_sizes(self):
        img = torch.rand(1, 10, 10, 3)
        res_1x1 = self.processor.rescale_i(img, 1, 1, "bilinear")
        self.assertEqual(res_1x1.shape, (1, 1, 1, 3))

        res_zero = self.processor.rescale_i(img, 0, -5, "bilinear")
        self.assertEqual(res_zero.shape, (1, 1, 1, 3))

    def test_rescale_m_algorithms_and_shapes(self):
        mask_3d = torch.rand(1, 24, 24, dtype=torch.float32)
        mask_2d = torch.rand(24, 24, dtype=torch.float32)
        algorithms = ["bicubic", "bilinear", "nearest", "nearest-exact", "lanczos", "box", "area", "hamming"]
        for algo in algorithms:
            with self.subTest(algorithm=algo):
                res_3d = self.processor.rescale_m(mask_3d, 48, 36, algo)
                self.assertEqual(res_3d.shape, (1, 36, 48))
                res_2d = self.processor.rescale_m(mask_2d, 48, 36, algo)
                self.assertEqual(res_2d.shape, (1, 36, 48))

    def test_rescale_m_dtypes(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                mask = torch.rand(1, 16, 16, dtype=dtype)
                res = self.processor.rescale_m(mask, 32, 32, "bilinear")
                self.assertEqual(res.shape, (1, 32, 32))

    def test_fillholes_functional_hollow_ring(self):
        # Frame enclosing empty hole in center
        mask = torch.zeros(1, 40, 40, dtype=torch.float32)
        mask[:, 5:35, 5:10] = 1.0
        mask[:, 5:35, 30:35] = 1.0
        mask[:, 5:10, 5:35] = 1.0
        mask[:, 30:35, 5:35] = 1.0

        # Center is originally zero
        self.assertEqual(mask[0, 20, 20].item(), 0.0)
        filled = self.processor.fillholes_iterative_hipass_fill_m(mask)
        # Inside center must now be filled to 1.0
        self.assertAlmostEqual(filled[0, 20, 20].item(), 1.0)
        # Outside corner remains 0.0
        self.assertEqual(filled[0, 0, 0].item(), 0.0)

    def test_fillholes_multilevel_gradient(self):
        # Soft threshold ring (0.8) with 0.0 hole
        mask = torch.zeros(1, 30, 30, dtype=torch.float32)
        mask[:, 5:25, 5:25] = 0.8
        mask[:, 10:20, 10:20] = 0.0
        filled = self.processor.fillholes_iterative_hipass_fill_m(mask)
        self.assertAlmostEqual(filled[0, 15, 15].item(), 0.8, places=2)
        self.assertEqual(filled[0, 0, 0].item(), 0.0)

    def test_fillholes_2d_mask(self):
        mask_2d = torch.zeros(40, 40, dtype=torch.float32)
        mask_2d[5:35, 5:10] = 1.0
        mask_2d[5:35, 30:35] = 1.0
        mask_2d[5:10, 5:35] = 1.0
        mask_2d[30:35, 5:35] = 1.0
        filled_2d = self.processor.fillholes_iterative_hipass_fill_m(mask_2d)
        self.assertEqual(filled_2d.ndim, 2)
        self.assertAlmostEqual(filled_2d[20, 20].item(), 1.0)
        self.assertEqual(filled_2d[0, 0].item(), 0.0)

    def test_fillholes_solid_and_empty(self):
        # Solid mask should remain solid
        solid = torch.ones(1, 20, 20, dtype=torch.float32)
        self.assertTrue(torch.equal(self.processor.fillholes_iterative_hipass_fill_m(solid), solid))

        # Empty mask should remain empty
        empty = torch.zeros(1, 20, 20, dtype=torch.float32)
        self.assertTrue(torch.equal(self.processor.fillholes_iterative_hipass_fill_m(empty), empty))

    def test_hipassfilter_m(self):
        mask = torch.tensor([[[0.1, 0.4], [0.6, 0.9]]], dtype=torch.float32)
        filtered = self.processor.hipassfilter_m(mask, 0.5)
        self.assertEqual(filtered[0, 0, 0].item(), 0.0)
        self.assertEqual(filtered[0, 0, 1].item(), 0.0)
        self.assertAlmostEqual(filtered[0, 1, 0].item(), 0.6)
        self.assertAlmostEqual(filtered[0, 1, 1].item(), 0.9)

        filtered_neg = self.processor.hipassfilter_m(mask, -1.0)
        self.assertTrue(torch.equal(filtered_neg, mask))

        filtered_high = self.processor.hipassfilter_m(mask, 1.5)
        self.assertEqual(torch.count_nonzero(filtered_high).item(), 0)

    def test_expand_m_exact_radii(self):
        mask = torch.zeros(1, 31, 31, dtype=torch.float32)
        mask[0, 15, 15] = 1.0
        exp = self.processor.expand_m(mask, 4)
        self.assertEqual(exp[0, 15, 15].item(), 1.0)
        self.assertEqual(exp[0, 15, 16].item(), 1.0)
        self.assertEqual(exp[0, 15, 14].item(), 1.0)
        self.assertEqual(exp[0, 0, 0].item(), 0.0)

    def test_expand_m_boundary_cases(self):
        mask = torch.zeros(1, 20, 20, dtype=torch.float32)
        mask[0, 10, 10] = 1.0
        self.assertTrue(torch.equal(self.processor.expand_m(mask, 0), mask))
        self.assertTrue(torch.equal(self.processor.expand_m(mask, -5), mask))

        # 2D mask
        mask_2d = torch.zeros(20, 20, dtype=torch.float32)
        mask_2d[10, 10] = 1.0
        exp_2d = self.processor.expand_m(mask_2d, 4)
        self.assertEqual(exp_2d.ndim, 2)

        # Huge expansion
        exp_large = self.processor.expand_m(mask, 100)
        self.assertEqual(exp_large.shape, mask.shape)

    def test_invert_m(self):
        mask = torch.tensor([[[0.0, 0.25], [0.75, 1.0]]], dtype=torch.float32)
        inv = self.processor.invert_m(mask)
        self.assertAlmostEqual(inv[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(inv[0, 0, 1].item(), 0.75)
        self.assertAlmostEqual(inv[0, 1, 0].item(), 0.25)
        self.assertAlmostEqual(inv[0, 1, 1].item(), 0.0)

        # Bool mask
        mask_bool = torch.tensor([[[False, True], [True, False]]], dtype=torch.bool)
        inv_bool = self.processor.invert_m(mask_bool)
        self.assertAlmostEqual(inv_bool[0, 0, 0].item(), 1.0)
        self.assertAlmostEqual(inv_bool[0, 0, 1].item(), 0.0)

    def test_blur_m_bool_and_int_dtypes(self):
        mask_bool = torch.zeros(1, 21, 21, dtype=torch.bool)
        mask_bool[0, 10, 10] = True
        blurred_bool = self.processor.blur_m(mask_bool, 3)
        self.assertGreater(blurred_bool[0, 10, 10].item(), 0.0)

        mask_u8 = torch.zeros(1, 21, 21, dtype=torch.uint8)
        mask_u8[0, 10, 10] = 255
        blurred_u8 = self.processor.blur_m(mask_u8, 3)
        self.assertGreater(blurred_u8[0, 10, 10].item(), 0.0)

    def test_blur_m_gaussian_decay(self):
        mask = torch.zeros(1, 31, 31, dtype=torch.float32)
        mask[0, 15, 15] = 1.0
        blurred = self.processor.blur_m(mask, 4)

        peak = blurred[0, 15, 15].item()
        d1 = blurred[0, 15, 16].item()
        d2 = blurred[0, 15, 17].item()
        self.assertGreater(peak, d1)
        self.assertGreater(d1, d2)
        self.assertAlmostEqual(blurred[0, 0, 0].item(), 0.0, places=3)

    def test_pad_to_multiple(self):
        for val, mult, expected in [
            (0, 8, 0), (7, 8, 8), (8, 8, 8), (9, 8, 16),
            (60, 8, 64), (64, 8, 64), (65, 8, 72),
            (64, 16, 64), (65, 16, 80), (100, 32, 128),
            (50, 0, 50), (50, -8, 50)
        ]:
            with self.subTest(val=val, mult=mult):
                self.assertEqual(self.processor.pad_to_multiple(val, mult), expected)

    def test_debug_context_location_in_image_inversion(self):
        img = torch.full((1, 30, 30, 3), 0.25, dtype=torch.float32)
        deb = self.processor.debug_context_location_in_image(img, 10, 10, 10, 10)
        # Inside box: 1.0 - 0.25 = 0.75
        self.assertAlmostEqual(deb[0, 15, 15, 0].item(), 0.75)
        # Outside box: remains 0.25
        self.assertAlmostEqual(deb[0, 0, 0, 0].item(), 0.25)

        # Clamping out-of-bounds coordinates
        deb_out = self.processor.debug_context_location_in_image(img, -10, -10, 20, 20)
        self.assertEqual(deb_out.shape, img.shape)
        deb_far = self.processor.debug_context_location_in_image(img, 100, 100, 20, 20)
        self.assertEqual(deb_far.shape, img.shape)

    def test_extend_imm_edge_preservation(self):
        img = torch.zeros(1, 10, 10, 3, dtype=torch.float32)
        img[:, 0, :, 0] = 0.42
        img[:, -1, :, 1] = 0.77
        mask = torch.zeros(1, 10, 10, dtype=torch.float32)
        e_img, e_mask, _ = self.processor.extend_imm(img, mask, None, 2.0, 2.0, 1.0, 1.0)
        self.assertAlmostEqual(e_img[0, 0, 5, 0].item(), 0.42)
        self.assertAlmostEqual(e_img[0, -1, 5, 1].item(), 0.77)
        self.assertAlmostEqual(e_mask[0, 0, 5].item(), 1.0)
        self.assertAlmostEqual(e_mask[0, -1, 5].item(), 1.0)

    def test_preresize_imm_modes(self):
        img = torch.rand(1, 100, 100, 3, dtype=torch.float32)
        mask = torch.zeros(1, 100, 100, dtype=torch.float32)
        opt_mask = torch.zeros(1, 100, 100, dtype=torch.float32)

        # Ensure min resolution
        p_img, p_mask, p_opt = self.processor.preresize_imm(img, mask, opt_mask, "bilinear", "bicubic", "ensure minimum resolution", 200, 150, 400, 400)
        self.assertGreaterEqual(p_img.shape[2], 200)
        self.assertGreaterEqual(p_img.shape[1], 150)

        # Ensure max resolution
        p_img2, p_mask2, p_opt2 = self.processor.preresize_imm(img, mask, opt_mask, "bilinear", "bicubic", "ensure maximum resolution", 10, 10, 50, 80)
        self.assertLessEqual(p_img2.shape[2], 50)
        self.assertLessEqual(p_img2.shape[1], 80)


if __name__ == '__main__':
    unittest.main()
