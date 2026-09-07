import unittest
import torch
from inpaint_cropandstitch import CPUProcessorLogic, GPUProcessorLogic


class TestCropAndStitch(unittest.TestCase):
    def setUp(self):
        self.processors = [
            ("cpu", CPUProcessorLogic()),
            ("gpu", GPUProcessorLogic()),
        ]

    def test_crop_magic_im_normal(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                img = torch.rand(1, 100, 100, 3, dtype=torch.float32)
                mask = torch.zeros(1, 100, 100, dtype=torch.float32)
                canvas, cto_x, cto_y, cto_w, cto_h, crop_im, crop_m, ctc_x, ctc_y, ctc_w, ctc_h = proc.crop_magic_im(
                    img, mask, x=20, y=20, w=40, h=40, target_w=64, target_h=64, padding=8,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic", resize_output=True
                )
                self.assertEqual(crop_im.shape, (1, 64, 64, 3))
                self.assertEqual(crop_m.shape, (1, 64, 64))
                self.assertEqual(crop_im.dtype, torch.float32)

    def test_crop_magic_im_non_positive_dimensions(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                img = torch.rand(1, 50, 50, 3, dtype=torch.float32)
                mask = torch.zeros(1, 50, 50, dtype=torch.float32)
                # w=0, h=0
                res = proc.crop_magic_im(
                    img, mask, x=0, y=0, w=0, h=0, target_w=64, target_h=64, padding=8,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic", resize_output=True
                )
                self.assertEqual(res[5].shape, (1, 64, 64, 3))

                # target_w <= 0
                res_bad_target = proc.crop_magic_im(
                    img, mask, x=0, y=0, w=20, h=20, target_w=-10, target_h=0, padding=8,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic", resize_output=True
                )
                self.assertEqual(res_bad_target[5].shape, img.shape)

    def test_crop_magic_im_channels(self):
        for name, proc in self.processors:
            for c in [1, 3, 4]:
                with self.subTest(processor=name, channels=c):
                    img = torch.rand(1, 60, 60, c, dtype=torch.float32)
                    mask = torch.zeros(1, 60, 60, dtype=torch.float32)
                    _, _, _, _, _, crop_im, crop_m, _, _, _, _ = proc.crop_magic_im(
                        img, mask, x=10, y=10, w=30, h=30, target_w=32, target_h=32, padding=0,
                        downscale_algorithm="bilinear", upscale_algorithm="bicubic", resize_output=True
                    )
                    self.assertEqual(crop_im.shape[-1], c)

    def test_stitch_magic_im_reproduce_rgba_canvas_rgb_inpaint(self):
        """
        Direct test for the user-reported bug:
        RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 3
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
        """
        for name, proc in self.processors:
            with self.subTest(processor=name):
                # Canvas is RGBA (4 channels)
                canvas_image = torch.ones(1, 100, 100, 4, dtype=torch.float32)
                # Set specific alpha in canvas to verify preservation
                canvas_image[:, :, :, 3] = 0.75

                # Inpainted crop is RGB (3 channels) as returned by standard VAE decode
                inpainted_image = torch.zeros(1, 40, 40, 3, dtype=torch.float32)
                # Mask
                mask = torch.ones(1, 40, 40, dtype=torch.float32)

                ctc_x, ctc_y, ctc_w, ctc_h = 20, 20, 40, 40
                cto_x, cto_y, cto_w, cto_h = 0, 0, 100, 100

                output = proc.stitch_magic_im(
                    canvas_image, inpainted_image, mask,
                    ctc_x, ctc_y, ctc_w, ctc_h,
                    cto_x, cto_y, cto_w, cto_h,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic"
                )

                # Output should have 4 channels and match original canvas size
                self.assertEqual(output.shape, (1, 100, 100, 4))
                # Stitched region RGB should be 0.0 (from inpaint)
                self.assertAlmostEqual(output[0, 30, 30, 0].item(), 0.0)
                # Stitched region Alpha should preserve original canvas alpha (0.75)
                self.assertAlmostEqual(output[0, 30, 30, 3].item(), 0.75)

    def test_stitch_magic_im_rgb_canvas_rgba_inpaint(self):
        # Canvas is RGB (3 channels), Inpaint is RGBA (4 channels)
        for name, proc in self.processors:
            with self.subTest(processor=name):
                canvas_image = torch.ones(1, 100, 100, 3, dtype=torch.float32)
                inpainted_image = torch.zeros(1, 40, 40, 4, dtype=torch.float32)
                mask = torch.ones(1, 40, 40, dtype=torch.float32)

                output = proc.stitch_magic_im(
                    canvas_image, inpainted_image, mask,
                    20, 20, 40, 40, 0, 0, 100, 100,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic"
                )
                self.assertEqual(output.shape, (1, 100, 100, 3))
                self.assertAlmostEqual(output[0, 30, 30, 0].item(), 0.0)

    def test_stitch_magic_im_channel_matching_variants(self):
        # Grayscale canvas (1) + RGB inpaint (3)
        for name, proc in self.processors:
            with self.subTest(processor=name, mode="gray_rgb"):
                canvas_image = torch.ones(1, 80, 80, 1, dtype=torch.float32)
                inpainted_image = torch.zeros(1, 30, 30, 3, dtype=torch.float32)
                mask = torch.ones(1, 30, 30, dtype=torch.float32)
                output = proc.stitch_magic_im(
                    canvas_image, inpainted_image, mask,
                    10, 10, 30, 30, 0, 0, 80, 80,
                    "bilinear", "bicubic"
                )
                self.assertEqual(output.shape, (1, 80, 80, 1))

            # RGB canvas (3) + Grayscale inpaint (1)
            with self.subTest(processor=name, mode="rgb_gray"):
                canvas_image = torch.ones(1, 80, 80, 3, dtype=torch.float32)
                inpainted_image = torch.zeros(1, 30, 30, 1, dtype=torch.float32)
                mask = torch.ones(1, 30, 30, dtype=torch.float32)
                output = proc.stitch_magic_im(
                    canvas_image, inpainted_image, mask,
                    10, 10, 30, 30, 0, 0, 80, 80,
                    "bilinear", "bicubic"
                )
                self.assertEqual(output.shape, (1, 80, 80, 3))

    def test_stitch_magic_im_boundary_clipping_and_out_of_bounds(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                canvas_image = torch.ones(1, 100, 100, 3, dtype=torch.float32)
                inpainted_image = torch.zeros(1, 50, 50, 3, dtype=torch.float32)
                mask = torch.ones(1, 50, 50, dtype=torch.float32)

                # Coordinate extending past image right and bottom: x=80, w=50 (exceeds 100)
                output = proc.stitch_magic_im(
                    canvas_image, inpainted_image, mask,
                    ctc_x=80, ctc_y=80, ctc_w=50, ctc_h=50,
                    cto_x=0, cto_y=0, cto_w=100, cto_h=100,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic"
                )
                self.assertEqual(output.shape, (1, 100, 100, 3))

                # Negative coordinates: x=-10, y=-10
                output_neg = proc.stitch_magic_im(
                    canvas_image, inpainted_image, mask,
                    ctc_x=-10, ctc_y=-10, ctc_w=50, ctc_h=50,
                    cto_x=0, cto_y=0, cto_w=100, cto_h=100,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic"
                )
                self.assertEqual(output_neg.shape, (1, 100, 100, 3))

                # Completely outside canvas: x=500, y=500
                output_disjoint = proc.stitch_magic_im(
                    canvas_image, inpainted_image, mask,
                    ctc_x=500, ctc_y=500, ctc_w=50, ctc_h=50,
                    cto_x=0, cto_y=0, cto_w=100, cto_h=100,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic"
                )
                self.assertEqual(output_disjoint.shape, (1, 100, 100, 3))

    def test_stitch_magic_im_dtypes_and_shapes(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                # canvas float32, inpaint float16
                canvas = torch.ones(1, 60, 60, 3, dtype=torch.float32)
                inpaint = torch.zeros(1, 30, 30, 3, dtype=torch.float16)
                mask = torch.ones(1, 30, 30, dtype=torch.float32)

                out = proc.stitch_magic_im(
                    canvas, inpaint, mask,
                    15, 15, 30, 30, 0, 0, 60, 60,
                    "bilinear", "bicubic"
                )
                self.assertEqual(out.dtype, torch.float32)

                # 3D inpainted image [H, W, C] without batch dimension
                inpaint_3d = torch.zeros(30, 30, 3, dtype=torch.float32)
                mask_2d = torch.ones(30, 30, dtype=torch.float32)
                out_3d = proc.stitch_magic_im(
                    canvas, inpaint_3d, mask_2d,
                    15, 15, 30, 30, 0, 0, 60, 60,
                    "bilinear", "bicubic"
                )
    def test_crop_magic_im_pixel_accuracy(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                img = torch.zeros(1, 50, 50, 3, dtype=torch.float32)
                # Specific marker pixel
                img[0, 25, 25, :] = torch.tensor([0.123, 0.456, 0.789])
                mask = torch.zeros(1, 50, 50, dtype=torch.float32)

                # Crop 10x10 around (20, 20) without resize
                canvas, cto_x, cto_y, cto_w, cto_h, crop_im, crop_m, ctc_x, ctc_y, ctc_w, ctc_h = proc.crop_magic_im(
                    img, mask, x=20, y=20, w=10, h=10, target_w=10, target_h=10, padding=0,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic", resize_output=False
                )
                # Marker should be at local offset (5, 5)
                self.assertAlmostEqual(crop_im[0, 5, 5, 0].item(), 0.123, places=3)
                self.assertAlmostEqual(crop_im[0, 5, 5, 1].item(), 0.456, places=3)
                self.assertAlmostEqual(crop_im[0, 5, 5, 2].item(), 0.789, places=3)

    def test_stitch_magic_im_exact_blending(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                # Canvas has value 0.2
                canvas = torch.full((1, 40, 40, 3), 0.2, dtype=torch.float32)
                # Inpaint has value 0.8
                inpaint = torch.full((1, 20, 20, 3), 0.8, dtype=torch.float32)
                # Mask has value 0.5 (exact 50/50 blend)
                mask = torch.full((1, 20, 20), 0.5, dtype=torch.float32)

                out = proc.stitch_magic_im(
                    canvas, inpaint, mask,
                    ctc_x=10, ctc_y=10, ctc_w=20, ctc_h=20,
                    cto_x=0, cto_y=0, cto_w=40, cto_h=40,
                    downscale_algorithm="bilinear", upscale_algorithm="bicubic"
                )
                # 0.5 * 0.8 + 0.5 * 0.2 = 0.500 (allowing for 8-bit PIL quantization)
                self.assertAlmostEqual(out[0, 15, 15, 0].item(), 0.5, places=2)
                # Outside stitched box, canvas remains 0.2
                self.assertAlmostEqual(out[0, 0, 0, 0].item(), 0.2, places=2)


if __name__ == '__main__':
    unittest.main()
