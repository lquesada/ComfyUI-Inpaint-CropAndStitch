import unittest
import torch
from inpaint_cropandstitch import GPUProcessorLogic


class TestGPUProcessorLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.devices = ["cpu"]
        if torch.cuda.is_available():
            cls.devices.append("cuda")

    def setUp(self):
        self.processor = GPUProcessorLogic()

    def test_rescale_i_algorithms(self):
        for dev in self.devices:
            img = torch.rand(1, 32, 32, 3, dtype=torch.float32, device=dev)
            algorithms = ["bicubic", "bilinear", "nearest", "nearest-exact", "lanczos", "box", "area", "hamming"]
            for algo in algorithms:
                with self.subTest(device=dev, algorithm=algo):
                    res = self.processor.rescale_i(img, 64, 48, algo)
                    self.assertEqual(res.shape, (1, 48, 64, 3))
                    self.assertEqual(res.device.type, dev)

    def test_rescale_i_channels_and_dtypes(self):
        for dev in self.devices:
            for c in [1, 3, 4]:
                with self.subTest(device=dev, channels=c):
                    img = torch.rand(2, 20, 20, c, device=dev)
                    res = self.processor.rescale_i(img, 30, 30, "bilinear")
                    self.assertEqual(res.shape, (2, 30, 30, c))

            for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                with self.subTest(device=dev, dtype=dtype):
                    img = torch.rand(1, 16, 16, 3, dtype=dtype, device=dev)
                    res = self.processor.rescale_i(img, 32, 32, "bicubic")
                    self.assertEqual(res.shape, (1, 32, 32, 3))

    def test_rescale_m_algorithms_and_shapes(self):
        for dev in self.devices:
            mask_3d = torch.rand(1, 24, 24, dtype=torch.float32, device=dev)
            mask_2d = torch.rand(24, 24, dtype=torch.float32, device=dev)
            for algo in ["bicubic", "bilinear", "nearest-exact", "box"]:
                with self.subTest(device=dev, algorithm=algo):
                    res_3d = self.processor.rescale_m(mask_3d, 40, 40, algo)
                    self.assertEqual(res_3d.shape, (1, 40, 40))
                    self.assertEqual(res_3d.device.type, dev)
                    res_2d = self.processor.rescale_m(mask_2d, 40, 40, algo)
                    self.assertEqual(res_2d.shape, (1, 40, 40))

    def test_fillholes_functional_hollow_ring(self):
        for dev in self.devices:
            mask = torch.zeros(1, 40, 40, dtype=torch.float32, device=dev)
            mask[:, 5:35, 5:10] = 1.0
            mask[:, 5:35, 30:35] = 1.0
            mask[:, 5:10, 5:35] = 1.0
            mask[:, 30:35, 5:35] = 1.0

            self.assertEqual(mask[0, 20, 20].item(), 0.0)
            filled = self.processor.fillholes_iterative_hipass_fill_m(mask)
            self.assertEqual(filled.device.type, dev)
            self.assertAlmostEqual(filled[0, 20, 20].item(), 1.0)
            self.assertEqual(filled[0, 0, 0].item(), 0.0)

    def test_fillholes_2d_mask(self):
        for dev in self.devices:
            mask_2d = torch.zeros(40, 40, dtype=torch.float32, device=dev)
            mask_2d[5:35, 5:10] = 1.0
            mask_2d[5:35, 30:35] = 1.0
            mask_2d[5:10, 5:35] = 1.0
            mask_2d[30:35, 5:35] = 1.0
            filled_2d = self.processor.fillholes_iterative_hipass_fill_m(mask_2d)
            self.assertEqual(filled_2d.ndim, 2)
            self.assertAlmostEqual(filled_2d[20, 20].item(), 1.0)
            self.assertEqual(filled_2d[0, 0].item(), 0.0)

    def test_fillholes_bfloat16(self):
        for dev in self.devices:
            mask = torch.zeros(1, 20, 20, dtype=torch.bfloat16, device=dev)
            mask[:, 5:15, 5:15] = 1.0
            mask[:, 8:12, 8:12] = 0.0
            filled = self.processor.fillholes_iterative_hipass_fill_m(mask)
            self.assertEqual(filled.shape, mask.shape)

    def test_hipassfilter_m(self):
        for dev in self.devices:
            mask = torch.tensor([[[0.1, 0.4], [0.6, 0.9]]], dtype=torch.float32, device=dev)
            filtered = self.processor.hipassfilter_m(mask, 0.5)
            self.assertEqual(filtered[0, 0, 0].item(), 0.0)
            self.assertAlmostEqual(filtered[0, 1, 1].item(), 0.9)

    def test_expand_m_exact_radii(self):
        for dev in self.devices:
            mask = torch.zeros(1, 31, 31, dtype=torch.float32, device=dev)
            mask[0, 15, 15] = 1.0
            exp = self.processor.expand_m(mask, 4)
            self.assertEqual(exp[0, 15, 15].item(), 1.0)
            self.assertEqual(exp[0, 15, 16].item(), 1.0)
            self.assertEqual(exp[0, 15, 14].item(), 1.0)
            self.assertEqual(exp[0, 0, 0].item(), 0.0)

    def test_expand_m_boundary_cases(self):
        for dev in self.devices:
            mask = torch.zeros(1, 20, 20, dtype=torch.float32, device=dev)
            mask[0, 10, 10] = 1.0
            self.assertTrue(torch.equal(self.processor.expand_m(mask, 0), mask))
            self.assertTrue(torch.equal(self.processor.expand_m(mask, -3), mask))

            mask_2d = torch.zeros(20, 20, dtype=torch.float32, device=dev)
            mask_2d[10, 10] = 1.0
            exp_2d = self.processor.expand_m(mask_2d, 4)
            self.assertEqual(exp_2d.ndim, 2)

            exp_huge = self.processor.expand_m(mask, 80)
            self.assertEqual(exp_huge.shape, mask.shape)

    def test_expand_m_bool_and_int_dtypes(self):
        for dev in self.devices:
            mask_bool = torch.zeros(1, 21, 21, dtype=torch.bool, device=dev)
            mask_bool[0, 10, 10] = True
            exp_bool = self.processor.expand_m(mask_bool, 3)
            self.assertEqual(exp_bool[0, 10, 10].item(), True)
            self.assertEqual(exp_bool[0, 10, 11].item(), True)
            self.assertEqual(exp_bool[0, 0, 0].item(), False)

            mask_u8 = torch.zeros(1, 21, 21, dtype=torch.uint8, device=dev)
            mask_u8[0, 10, 10] = 255
            exp_u8 = self.processor.expand_m(mask_u8, 3)
            self.assertEqual(exp_u8[0, 10, 10].item(), 255)

    def test_invert_m(self):
        for dev in self.devices:
            mask = torch.tensor([[[0.0, 0.25], [0.75, 1.0]]], dtype=torch.float32, device=dev)
            inv = self.processor.invert_m(mask)
            self.assertAlmostEqual(inv[0, 0, 0].item(), 1.0)
            self.assertAlmostEqual(inv[0, 1, 1].item(), 0.0)

            # Bool mask
            mask_bool = torch.tensor([[[False, True], [True, False]]], dtype=torch.bool, device=dev)
            inv_bool = self.processor.invert_m(mask_bool)
            self.assertAlmostEqual(inv_bool[0, 0, 0].item(), 1.0)
            self.assertAlmostEqual(inv_bool[0, 0, 1].item(), 0.0)

    def test_blur_m_gaussian_decay(self):
        for dev in self.devices:
            mask = torch.zeros(1, 31, 31, dtype=torch.float32, device=dev)
            mask[0, 15, 15] = 1.0
            blurred = self.processor.blur_m(mask, 4)

            peak = blurred[0, 15, 15].item()
            d1 = blurred[0, 15, 16].item()
            d2 = blurred[0, 15, 17].item()
            self.assertGreater(peak, d1)
            self.assertGreater(d1, d2)
            self.assertAlmostEqual(blurred[0, 0, 0].item(), 0.0, places=3)

    def test_blur_m_bool_and_int_dtypes(self):
        for dev in self.devices:
            mask_bool = torch.zeros(1, 21, 21, dtype=torch.bool, device=dev)
            mask_bool[0, 10, 10] = True
            blurred_bool = self.processor.blur_m(mask_bool, 3)
            self.assertGreater(blurred_bool[0, 10, 10].item(), 0.0)
            self.assertAlmostEqual(blurred_bool[0, 0, 0].item(), 0.0, places=3)

            mask_u8 = torch.zeros(1, 21, 21, dtype=torch.uint8, device=dev)
            mask_u8[0, 10, 10] = 1
            blurred_u8 = self.processor.blur_m(mask_u8, 3)
            self.assertGreater(blurred_u8[0, 10, 10].item(), 0.0)

    def test_pad_to_multiple(self):
        self.assertEqual(self.processor.pad_to_multiple(60, 8), 64)
        self.assertEqual(self.processor.pad_to_multiple(50, 0), 50)
        self.assertEqual(self.processor.pad_to_multiple(50, -4), 50)

    def test_debug_context_location_in_image(self):
        for dev in self.devices:
            img = torch.full((1, 30, 30, 3), 0.25, dtype=torch.float32, device=dev)
            deb = self.processor.debug_context_location_in_image(img, 10, 10, 10, 10)
            self.assertAlmostEqual(deb[0, 15, 15, 0].item(), 0.75)
            self.assertAlmostEqual(deb[0, 0, 0, 0].item(), 0.25)

    def test_extend_imm_edge_preservation(self):
        for dev in self.devices:
            img = torch.zeros(1, 10, 10, 3, dtype=torch.float32, device=dev)
            img[:, 0, :, 0] = 0.55
            img[:, -1, :, 2] = 0.88
            mask = torch.zeros(1, 10, 10, dtype=torch.float32, device=dev)
            e_img, e_mask, _ = self.processor.extend_imm(img, mask, None, 2.0, 2.0, 1.0, 1.0)
            self.assertAlmostEqual(e_img[0, 0, 5, 0].item(), 0.55)
            self.assertAlmostEqual(e_img[0, -1, 5, 2].item(), 0.88)
            self.assertAlmostEqual(e_mask[0, 0, 5].item(), 1.0)

    def test_preresize_imm_modes(self):
        for dev in self.devices:
            img = torch.rand(1, 50, 60, 3, dtype=torch.float32, device=dev)
            mask = torch.rand(1, 50, 60, dtype=torch.float32, device=dev)

            # ensure minimum
            r_img, r_mask, _ = self.processor.preresize_imm(
                img, mask, mask, "bilinear", "bicubic", "ensure minimum resolution",
                100, 100, 200, 200
            )
            self.assertGreaterEqual(r_img.shape[2], 100)
            self.assertGreaterEqual(r_img.shape[1], 100)

            # ensure maximum
            r_img2, r_mask2, _ = self.processor.preresize_imm(
                img, mask, mask, "bilinear", "bicubic", "ensure maximum resolution",
                20, 20, 30, 30
            )
            self.assertLessEqual(r_img2.shape[2], 30)
            self.assertLessEqual(r_img2.shape[1], 30)


if __name__ == '__main__':
    unittest.main()
