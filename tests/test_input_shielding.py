import unittest
import torch
from inpaint_cropandstitch import (
    InpaintCropImproved,
    InpaintStitchImproved,
    _sanitize_image_tensor,
    _sanitize_mask_tensor,
)


class TestInputShielding(unittest.TestCase):
    def setUp(self):
        self.crop_node = InpaintCropImproved()
        self.stitch_node = InpaintStitchImproved()

    # =========================================================================
    # 1. Tests for _sanitize_image_tensor
    # =========================================================================
    def test_sanitize_image_non_tensor_raises(self):
        with self.assertRaises(TypeError):
            _sanitize_image_tensor("not_a_tensor")
        with self.assertRaises(TypeError):
            _sanitize_image_tensor([1.0, 2.0, 3.0])

    def test_sanitize_image_2d(self):
        # [H, W] -> [1, H, W, 3] (allow_grayscale=False default)
        t = torch.rand(40, 50, dtype=torch.float32)
        sanitized = _sanitize_image_tensor(t)
        self.assertEqual(sanitized.shape, (1, 40, 50, 3))
        # With allow_grayscale=True -> [1, 40, 50, 1]
        sanitized_gray = _sanitize_image_tensor(t, allow_grayscale=True)
        self.assertEqual(sanitized_gray.shape, (1, 40, 50, 1))

    def test_sanitize_image_3d_unbatched_hwc(self):
        # [H, W, 3] -> [1, H, W, 3]
        t = torch.rand(30, 40, 3, dtype=torch.float32)
        sanitized = _sanitize_image_tensor(t)
        self.assertEqual(sanitized.shape, (1, 30, 40, 3))

    def test_sanitize_image_3d_channels_first_chw(self):
        # [3, H, W] -> [1, H, W, 3]
        t = torch.rand(3, 30, 40, dtype=torch.float32)
        sanitized = _sanitize_image_tensor(t)
        self.assertEqual(sanitized.shape, (1, 30, 40, 3))

    def test_sanitize_image_4d_channels_first_bchw(self):
        # [2, 3, 30, 40] -> [2, 30, 40, 3]
        t = torch.rand(2, 3, 30, 40, dtype=torch.float32)
        sanitized = _sanitize_image_tensor(t)
        self.assertEqual(sanitized.shape, (2, 30, 40, 3))

    def test_sanitize_image_5d_singleton(self):
        # [1, 1, 30, 40, 3] -> [1, 30, 40, 3]
        t = torch.rand(1, 1, 30, 40, 3, dtype=torch.float32)
        sanitized = _sanitize_image_tensor(t)
        self.assertEqual(sanitized.shape, (1, 30, 40, 3))

    def test_sanitize_image_2channel_to_rgba(self):
        # [1, 30, 40, 2] (grayscale + alpha) -> [1, 30, 40, 4]
        t = torch.rand(1, 30, 40, 2, dtype=torch.float32)
        sanitized = _sanitize_image_tensor(t)
        self.assertEqual(sanitized.shape, (1, 30, 40, 4))

    def test_sanitize_image_6channel_clamped(self):
        # >4 channels -> clamped to first 4 channels
        t = torch.rand(1, 30, 40, 6, dtype=torch.float32)
        sanitized = _sanitize_image_tensor(t)
        self.assertEqual(sanitized.shape, (1, 30, 40, 4))

    def test_sanitize_image_uint8_conversion(self):
        # uint8 0..255 -> float32 0..1
        t = (torch.rand(1, 20, 20, 3) * 255).to(torch.uint8)
        sanitized = _sanitize_image_tensor(t)
        self.assertEqual(sanitized.dtype, torch.float32)
        self.assertLessEqual(sanitized.max().item(), 1.0)
        self.assertGreaterEqual(sanitized.min().item(), 0.0)

    def test_sanitize_image_nans_and_infs(self):
        t = torch.tensor([[[[float("nan"), float("inf"), float("-inf"), 0.5]]]])
        sanitized = _sanitize_image_tensor(t)
        self.assertFalse(torch.isnan(sanitized).any())
        self.assertFalse(torch.isinf(sanitized).any())

    # =========================================================================
    # 2. Tests for _sanitize_mask_tensor
    # =========================================================================
    def test_sanitize_mask_none_and_empty(self):
        self.assertIsNone(_sanitize_mask_tensor(None))
        self.assertIsNone(_sanitize_mask_tensor(torch.tensor([])))

    def test_sanitize_mask_non_tensor_raises(self):
        with self.assertRaises(TypeError):
            _sanitize_mask_tensor("not_a_mask")

    def test_sanitize_mask_2d(self):
        # [H, W] -> [1, H, W]
        t = torch.rand(30, 40)
        sanitized = _sanitize_mask_tensor(t)
        self.assertEqual(sanitized.shape, (1, 30, 40))

    def test_sanitize_mask_3d_standard(self):
        # [B, H, W] stays [B, H, W]
        t = torch.rand(2, 30, 40)
        sanitized = _sanitize_mask_tensor(t)
        self.assertEqual(sanitized.shape, (2, 30, 40))

    def test_sanitize_mask_3d_trailing_singleton(self):
        # [30, 40, 1] unbatched with channel -> [1, 30, 40]
        t = torch.rand(30, 40, 1)
        sanitized = _sanitize_mask_tensor(t)
        self.assertEqual(sanitized.shape, (1, 30, 40))

    def test_sanitize_mask_4d_rgb_image_input(self):
        # RGB IMAGE [B, H, W, 3] wired to mask input -> converted to [B, H, W] luminance
        t = torch.rand(2, 30, 40, 3)
        sanitized = _sanitize_mask_tensor(t)
        self.assertEqual(sanitized.shape, (2, 30, 40))

    def test_sanitize_mask_4d_rgba_image_input(self):
        # RGBA IMAGE [B, H, W, 4] wired to mask input -> converted to [B, H, W] luminance
        t = torch.rand(1, 30, 40, 4)
        sanitized = _sanitize_mask_tensor(t)
        self.assertEqual(sanitized.shape, (1, 30, 40))

    def test_sanitize_mask_4d_channels_first_image_input(self):
        # Channels-first [B, 3, H, W] -> converted to [B, H, W]
        t = torch.rand(2, 3, 30, 40)
        sanitized = _sanitize_mask_tensor(t)
        self.assertEqual(sanitized.shape, (2, 30, 40))

    def test_sanitize_mask_uint8_and_clamping(self):
        t = (torch.rand(1, 20, 20) * 255).to(torch.uint8)
        sanitized = _sanitize_mask_tensor(t)
        self.assertEqual(sanitized.dtype, torch.float32)
        self.assertLessEqual(sanitized.max().item(), 1.0)
        self.assertGreaterEqual(sanitized.min().item(), 0.0)

    def test_sanitize_mask_nans_and_infs(self):
        t = torch.tensor([[float("nan"), float("inf"), float("-inf"), 0.5]])
        sanitized = _sanitize_mask_tensor(t)
        self.assertFalse(torch.isnan(sanitized).any())
        self.assertFalse(torch.isinf(sanitized).any())
        self.assertLessEqual(sanitized.max().item(), 1.0)
        self.assertGreaterEqual(sanitized.min().item(), 0.0)

    # =========================================================================
    # 3. Integration Tests for InpaintCropImproved Shielding
    # =========================================================================
    def _default_crop_args(self, image, mask=None, **kwargs):
        args = {
            "image": image,
            "mask": mask,
            "optional_context_mask": None,
            "downscale_algorithm": "bilinear",
            "upscale_algorithm": "bicubic",
            "preresize": False,
            "preresize_mode": "no preresize",
            "preresize_min_width": 256,
            "preresize_min_height": 256,
            "preresize_max_width": 1024,
            "preresize_max_height": 1024,
            "extend_for_outpainting": False,
            "extend_up_factor": 1.0,
            "extend_down_factor": 1.0,
            "extend_left_factor": 1.0,
            "extend_right_factor": 1.0,
            "mask_hipass_filter": 0.0,
            "mask_fill_holes": False,
            "mask_expand_pixels": 0,
            "mask_invert": False,
            "mask_blend_pixels": 0,
            "context_from_mask_extend_factor": 1.2,
            "output_resize_to_target_size": False,
            "output_target_width": 256,
            "output_target_height": 256,
            "output_padding": 8,
            "device_mode": "cpu (compatible)",
        }
        args.update(kwargs)
        return args

    def test_crop_with_rgb_image_as_mask(self):
        # User mistakenly wired IMAGE [1, 64, 64, 3] to MASK input
        image = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        image_as_mask = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
        image_as_mask[0, 20:40, 20:40, :] = 1.0  # White square

        args = self._default_crop_args(image, mask=image_as_mask)
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]
        self.assertEqual(crop_im.ndim, 4)
        self.assertEqual(crop_im.shape[-1], 3)
        self.assertEqual(crop_m.ndim, 3)

    def test_crop_with_rgba_image_as_mask(self):
        # User mistakenly wired RGBA IMAGE [1, 64, 64, 4] to MASK input
        image = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        rgba_as_mask = torch.zeros(1, 64, 64, 4, dtype=torch.float32)
        rgba_as_mask[0, 20:40, 20:40, :] = 1.0

        args = self._default_crop_args(image, mask=rgba_as_mask)
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]
        self.assertEqual(crop_im.ndim, 4)
        self.assertEqual(crop_m.ndim, 3)

    def test_crop_with_unbatched_inputs(self):
        # 3D unbatched image [64, 64, 3] and 2D unbatched mask [64, 64]
        image = torch.rand(64, 64, 3, dtype=torch.float32)
        mask = torch.zeros(64, 64, dtype=torch.float32)
        mask[20:40, 20:40] = 1.0

        args = self._default_crop_args(image, mask=mask)
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]
        self.assertEqual(crop_im.shape[0], 1)
        self.assertEqual(crop_m.shape[0], 1)

    def test_crop_mismatched_batch_harmonization(self):
        # Image batch = 1, Mask batch = 3
        image = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        mask = torch.zeros(3, 64, 64, dtype=torch.float32)
        mask[:, 20:40, 20:40] = 1.0

        args = self._default_crop_args(image, mask=mask)
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]
        self.assertEqual(crop_im.shape[0], 3)
        self.assertEqual(crop_m.shape[0], 3)
        self.assertEqual(len(stitcher["canvas_image"]), 3)

    def test_crop_with_extreme_and_negative_widget_values(self):
        # Negative factors, high blend pixels, negative padding should not crash
        image = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        mask = torch.zeros(1, 64, 64, dtype=torch.float32)
        mask[0, 20:40, 20:40] = 1.0

        args = self._default_crop_args(
            image,
            mask=mask,
            extend_up_factor=-0.5,
            extend_down_factor=-1.0,
            mask_blend_pixels=-5,
            mask_expand_pixels=-2,
            context_from_mask_extend_factor=0.5,  # should be bounded to >= 1.0
            output_padding=-4,
            mask_hipass_filter=2.5,  # should be clamped to 1.0
        )
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]
        self.assertIsNotNone(crop_im)
        self.assertIsNotNone(stitcher)

    # =========================================================================
    # 4. Integration Tests for InpaintStitchImproved Shielding (Issue #179 & Edge Cases)
    # =========================================================================
    def test_stitch_qwen_rgba_inpaint_into_rgb_canvas_issue_179(self):
        # Issue #179 scenario:
        # Canvas was cropped from standard RGB [1, 100, 100, 3]
        # Inpaint node (e.g. Qwen 2.1 / WanVAE) outputs RGBA [1, H, W, 4]
        orig_image = torch.rand(1, 100, 100, 3, dtype=torch.float32)
        mask = torch.zeros(1, 100, 100, dtype=torch.float32)
        mask[0, 30:60, 30:60] = 1.0

        args = self._default_crop_args(orig_image, mask=mask)
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]

        # Simulate Qwen 2.1 returning RGBA with 4 channels
        h, w = crop_im.shape[1], crop_im.shape[2]
        rgba_inpaint = torch.rand(1, h, w, 4, dtype=torch.float32)

        # Execute stitch
        (stitched,) = self.stitch_node.inpaint_stitch(stitcher, rgba_inpaint)
        self.assertEqual(stitched.shape, (1, 100, 100, 3))
        self.assertFalse(torch.isnan(stitched).any())

    def test_stitch_rgb_inpaint_into_rgba_canvas(self):
        # Canvas was RGBA [1, 100, 100, 4]
        # Inpainted image is RGB [1, H, W, 3]
        orig_image = torch.rand(1, 100, 100, 4, dtype=torch.float32)
        mask = torch.zeros(1, 100, 100, dtype=torch.float32)
        mask[0, 30:60, 30:60] = 1.0

        args = self._default_crop_args(orig_image, mask=mask)
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]

        # Standard 3-channel RGB inpaint
        h, w = crop_im.shape[1], crop_im.shape[2]
        rgb_inpaint = torch.rand(1, h, w, 3, dtype=torch.float32)

        (stitched,) = self.stitch_node.inpaint_stitch(stitcher, rgb_inpaint)
        # Stitched preserves the canvas RGBA channels
        self.assertEqual(stitched.shape, (1, 100, 100, 4))
        self.assertFalse(torch.isnan(stitched).any())

    def test_stitch_channels_first_inpainted_image(self):
        # Upstream node outputs [B, C, H, W] instead of [B, H, W, C]
        orig_image = torch.rand(1, 80, 80, 3, dtype=torch.float32)
        mask = torch.zeros(1, 80, 80, dtype=torch.float32)
        mask[0, 20:50, 20:50] = 1.0

        args = self._default_crop_args(orig_image, mask=mask)
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]

        h, w = crop_im.shape[1], crop_im.shape[2]
        # Inpainted image passed as [1, 3, H, W]
        bchw_inpaint = torch.rand(1, 3, h, w, dtype=torch.float32)

        (stitched,) = self.stitch_node.inpaint_stitch(stitcher, bchw_inpaint)
        self.assertEqual(stitched.shape, (1, 80, 80, 3))

    def test_stitch_unbatched_inpainted_image(self):
        # Upstream node outputs 3D unbatched [H, W, C]
        orig_image = torch.rand(1, 80, 80, 3, dtype=torch.float32)
        mask = torch.zeros(1, 80, 80, dtype=torch.float32)
        mask[0, 20:50, 20:50] = 1.0

        args = self._default_crop_args(orig_image, mask=mask)
        stitcher, crop_im, crop_m = self.crop_node.inpaint_crop(**args)[:3]

        h, w = crop_im.shape[1], crop_im.shape[2]
        hwc_inpaint = torch.rand(h, w, 3, dtype=torch.float32)

        (stitched,) = self.stitch_node.inpaint_stitch(stitcher, hwc_inpaint)
        self.assertEqual(stitched.shape, (1, 80, 80, 3))

    def test_stitch_invalid_stitcher_types_raise(self):
        inpaint = torch.rand(1, 30, 30, 3)
        with self.assertRaises(TypeError):
            self.stitch_node.inpaint_stitch("not_a_dict", inpaint)
        with self.assertRaises(TypeError):
            self.stitch_node.inpaint_stitch(None, inpaint)
        with self.assertRaises(ValueError):
            self.stitch_node.inpaint_stitch({}, inpaint)
        with self.assertRaises(ValueError):
            self.stitch_node.inpaint_stitch({"cropped_to_canvas_x": []}, inpaint)


if __name__ == "__main__":
    unittest.main()
