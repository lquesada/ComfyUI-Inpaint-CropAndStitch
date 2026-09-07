import unittest
import torch
from inpaint_cropandstitch import InpaintCropImproved, InpaintStitchImproved


class TestNodesPipeline(unittest.TestCase):
    def setUp(self):
        self.crop_node = InpaintCropImproved()
        self.stitch_node = InpaintStitchImproved()

    def _default_crop_args(self, image, mask=None, optional_context_mask=None, **kwargs):
        args = {
            "image": image,
            "mask": mask,
            "optional_context_mask": optional_context_mask,
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

    def test_node_metadata(self):
        crop_inputs = InpaintCropImproved.INPUT_TYPES()
        self.assertIn("required", crop_inputs)
        self.assertIn("image", crop_inputs["required"])
        self.assertEqual(InpaintCropImproved.FUNCTION, "inpaint_crop")

        stitch_inputs = InpaintStitchImproved.INPUT_TYPES()
        self.assertIn("required", stitch_inputs)
        self.assertIn("stitcher", stitch_inputs["required"])
        self.assertIn("inpainted_image", stitch_inputs["required"])
        self.assertEqual(InpaintStitchImproved.FUNCTION, "inpaint_stitch")

    def test_roundtrip_rgb_pipeline(self):
        # Standard RGB image [1, 100, 100, 3]
        orig_img = torch.rand(1, 100, 100, 3, dtype=torch.float32)
        mask = torch.zeros(1, 100, 100, dtype=torch.float32)
        mask[0, 30:50, 30:50] = 1.0

        args = self._default_crop_args(orig_img, mask=mask)
        crop_results = self.crop_node.inpaint_crop(**args)

        stitcher, cropped_image, cropped_mask = crop_results[:3]

        self.assertEqual(cropped_image.ndim, 4)
        self.assertEqual(cropped_mask.ndim, 3)

        # Simulate inpainting: invert color in crop
        inpainted_crop = 1.0 - cropped_image
        stitch_results = self.stitch_node.inpaint_stitch(stitcher, inpainted_crop)
        output_image = stitch_results[0]

        self.assertEqual(output_image.shape, orig_img.shape)
        # Inside inpaint area, output should reflect inverted crop
        self.assertNotEqual(output_image[0, 40, 40, 0].item(), orig_img[0, 40, 40, 0].item())
        # Outside inpaint area, output should match original image
        self.assertAlmostEqual(output_image[0, 5, 5, 0].item(), orig_img[0, 5, 5, 0].item(), places=5)

    def test_roundtrip_rgba_canvas_rgb_inpaint_pipeline(self):
        """
        End-to-end integration test of the reported bug:
        RGBA canvas input into InpaintCropImproved, standard RGB output from VAE into InpaintStitchImproved.
        """
        # Canvas has 4 channels (RGBA)
        rgba_img = torch.rand(1, 80, 80, 4, dtype=torch.float32)
        rgba_img[:, :, :, 3] = 0.8  # Specific alpha channel
        mask = torch.zeros(1, 80, 80, dtype=torch.float32)
        mask[0, 20:40, 20:40] = 1.0

        args = self._default_crop_args(rgba_img, mask=mask)
        crop_results = self.crop_node.inpaint_crop(**args)

        stitcher, cropped_image, _ = crop_results[:3]

        # Simulate VAE decoding only 3 channels (RGB)
        rgb_inpainted_crop = cropped_image[..., :3].clone() * 0.5

        # Stitch back together
        stitch_results = self.stitch_node.inpaint_stitch(stitcher, rgb_inpainted_crop)
        final_image = stitch_results[0]

        # Final image must be 4 channels (RGBA preserved)
        self.assertEqual(final_image.shape, (1, 80, 80, 4))
        # Alpha channel must be preserved from original canvas
        self.assertAlmostEqual(final_image[0, 30, 30, 3].item(), 0.8, places=5)

    def test_roundtrip_rgb_canvas_rgba_inpaint_pipeline(self):
        # Canvas has 3 channels (RGB), Inpainted crop has 4 channels (RGBA)
        rgb_img = torch.rand(1, 80, 80, 3, dtype=torch.float32)
        mask = torch.zeros(1, 80, 80, dtype=torch.float32)
        mask[0, 20:40, 20:40] = 1.0

        args = self._default_crop_args(rgb_img, mask=mask)
        crop_results = self.crop_node.inpaint_crop(**args)

        stitcher, cropped_image, _ = crop_results[:3]

        # Inpainted crop has an extra alpha channel
        rgba_inpainted_crop = torch.cat([cropped_image, torch.ones_like(cropped_image[..., :1])], dim=-1)

        stitch_results = self.stitch_node.inpaint_stitch(stitcher, rgba_inpainted_crop)
        final_image = stitch_results[0]

        # Final image must be 3 channels
        self.assertEqual(final_image.shape, (1, 80, 80, 3))

    def test_mask_dimensions_normalization(self):
        orig_img = torch.rand(1, 60, 60, 3, dtype=torch.float32)

        # 2D mask [H, W]
        mask_2d = torch.zeros(60, 60, dtype=torch.float32)
        mask_2d[20:30, 20:30] = 1.0
        res_2d = self.crop_node.inpaint_crop(**self._default_crop_args(orig_img, mask=mask_2d))
        self.assertEqual(res_2d[1].ndim, 4)

        # 4D mask [B, 1, H, W]
        mask_4d_b1hw = mask_2d.unsqueeze(0).unsqueeze(1)
        res_4d_1 = self.crop_node.inpaint_crop(**self._default_crop_args(orig_img, mask=mask_4d_b1hw))
        self.assertEqual(res_4d_1[1].ndim, 4)

        # 4D mask [B, H, W, 1]
        mask_4d_bhw1 = mask_2d.unsqueeze(0).unsqueeze(-1)
        res_4d_2 = self.crop_node.inpaint_crop(**self._default_crop_args(orig_img, mask=mask_4d_bhw1))
        self.assertEqual(res_4d_2[1].ndim, 4)

    def test_unbatched_image_input(self):
        # 3D image [H, W, C]
        img_3d = torch.rand(60, 60, 3, dtype=torch.float32)
        mask_2d = torch.zeros(60, 60, dtype=torch.float32)
        mask_2d[20:30, 20:30] = 1.0

        res = self.crop_node.inpaint_crop(**self._default_crop_args(img_3d, mask=mask_2d))
        self.assertEqual(res[1].ndim, 4)

    def test_outpainting_extension_pipeline(self):
        orig_img = torch.rand(1, 60, 60, 3, dtype=torch.float32)
        mask = torch.zeros(1, 60, 60, dtype=torch.float32)

        args = self._default_crop_args(
            orig_img, mask=mask,
            extend_for_outpainting=True,
            extend_up_factor=1.5,
            extend_down_factor=1.2,
            extend_left_factor=1.0,
            extend_right_factor=1.0
        )
        crop_results = self.crop_node.inpaint_crop(**args)
        stitcher, cropped_image, _ = crop_results[:3]

        stitch_results = self.stitch_node.inpaint_stitch(stitcher, cropped_image)
        # Stitched image has the extended (outpainted) image size: 60 * 1.7 = 102 height
        self.assertEqual(stitch_results[0].shape, (1, 102, 60, 3))

    def test_preresize_modes_pipeline(self):
        orig_img = torch.rand(1, 50, 50, 3, dtype=torch.float32)
        mask = torch.zeros(1, 50, 50, dtype=torch.float32)
        mask[0, 10:20, 10:20] = 1.0

        # ensure minimum resolution
        args_min = self._default_crop_args(
            orig_img, mask=mask,
            preresize=True,
            preresize_mode="ensure minimum resolution",
            preresize_min_width=100,
            preresize_min_height=100
        )
        res_min = self.crop_node.inpaint_crop(**args_min)
        stitcher = res_min[0]
        # canvas_image is stored as a list of images [img_1, img_2, ...]
        self.assertGreaterEqual(stitcher['canvas_image'][0].shape[2], 100)

    def test_device_mode_gpu(self):
        orig_img = torch.rand(1, 40, 40, 3, dtype=torch.float32)
        mask = torch.zeros(1, 40, 40, dtype=torch.float32)
        mask[0, 10:20, 10:20] = 1.0

        args = self._default_crop_args(orig_img, mask=mask, device_mode="gpu (much faster)")
        crop_results = self.crop_node.inpaint_crop(**args)
        stitcher, cropped_image, _ = crop_results[:3]
        stitch_results = self.stitch_node.inpaint_stitch(stitcher, cropped_image)
        self.assertEqual(stitch_results[0].shape, orig_img.shape)

    def test_invalid_stitcher_validation(self):
        # Missing required keys in stitcher dict
        corrupted_stitcher = {"canvas_image": [torch.zeros(1, 10, 10, 3)]}
        with self.assertRaises(ValueError):
            self.stitch_node.inpaint_stitch(corrupted_stitcher, torch.zeros(1, 10, 10, 3))

    def test_mask_none_default(self):
        # No mask provided: node creates default mask
        img = torch.rand(1, 40, 40, 3, dtype=torch.float32)
        res = self.crop_node.inpaint_crop(**self._default_crop_args(img, mask=None))
        stitcher, crop_im, crop_m = res[:3]
        self.assertEqual(crop_im.shape[-1], 3)
        self.assertIsNotNone(stitcher)

    def test_mask_mismatched_spatial_resolution(self):
        img = torch.rand(1, 64, 64, 3, dtype=torch.float32)
        # Empty mismatched mask (32x32)
        empty_mask = torch.zeros(1, 32, 32, dtype=torch.float32)
        res_empty = self.crop_node.inpaint_crop(**self._default_crop_args(img, mask=empty_mask))
        self.assertIsNotNone(res_empty[0])

        # Non-empty mismatched mask (32x32 with content)
        content_mask = torch.zeros(1, 32, 32, dtype=torch.float32)
        content_mask[0, 10:20, 10:20] = 1.0
        res_content = self.crop_node.inpaint_crop(**self._default_crop_args(img, mask=content_mask))
        self.assertIsNotNone(res_content[0])

    def test_output_resize_to_target_size(self):
        img = torch.rand(1, 100, 100, 3, dtype=torch.float32)
        mask = torch.zeros(1, 100, 100, dtype=torch.float32)
        mask[0, 20:40, 20:40] = 1.0

        args = self._default_crop_args(
            img, mask=mask,
            output_resize_to_target_size=True,
            output_target_width=128,
            output_target_height=128
        )
        res = self.crop_node.inpaint_crop(**args)
        stitcher, crop_im, crop_m = res[:3]
        self.assertEqual(crop_im.shape[1], 128)
        self.assertEqual(crop_im.shape[2], 128)

    def test_batch_processing(self):
        img_batch = torch.rand(2, 64, 64, 3, dtype=torch.float32)
        mask_batch = torch.zeros(2, 64, 64, dtype=torch.float32)
        mask_batch[0, 10:30, 10:30] = 1.0
        mask_batch[1, 20:40, 20:40] = 1.0

        args = self._default_crop_args(
            img_batch, mask=mask_batch,
            output_resize_to_target_size=True,
            output_target_width=64,
            output_target_height=64
        )
        res = self.crop_node.inpaint_crop(**args)
        stitcher, crop_im, crop_m = res[:3]
        self.assertEqual(crop_im.shape[0], 2)

        # Stitch
    def test_inpaint_crop_grayscale_1channel(self):
        # Grayscale 1-channel image input
        img_gray = torch.rand(1, 40, 40, 1, dtype=torch.float32)
        mask = torch.zeros(1, 40, 40, dtype=torch.float32)
        mask[0, 10:20, 10:20] = 1.0
        res = self.crop_node.inpaint_crop(**self._default_crop_args(img_gray, mask=mask))
        stitcher, crop_im, crop_m = res[:3]
        # Auto-converted to 3 channels RGB
        self.assertEqual(crop_im.shape[-1], 3)

    def test_inpaint_crop_2channel_grayscale_alpha(self):
        # 2-channel (grayscale + alpha)
        img_ga = torch.rand(1, 40, 40, 2, dtype=torch.float32)
        mask = torch.zeros(1, 40, 40, dtype=torch.float32)
        mask[0, 10:20, 10:20] = 1.0
        res = self.crop_node.inpaint_crop(**self._default_crop_args(img_ga, mask=mask))
        stitcher, crop_im, crop_m = res[:3]
        # Auto-converted to 4 channels RGBA
        self.assertEqual(crop_im.shape[-1], 4)

    def test_mask_range_255_normalization(self):
        # Mask with 0-255 values
        img = torch.rand(1, 50, 50, 3, dtype=torch.float32)
        mask_255 = torch.zeros(1, 50, 50, dtype=torch.float32)
        mask_255[0, 15:35, 15:35] = 255.0
        res = self.crop_node.inpaint_crop(**self._default_crop_args(img, mask=mask_255))
        stitcher, crop_im, crop_m = res[:3]
        # Mask is clamped and normalized into [0.0, 1.0]
        self.assertLessEqual(crop_m.max().item(), 1.0)

    def test_preresize_min_greater_than_max_swap(self):
        # Inverted min/max resolutions (min=800, max=400)
        img = torch.rand(1, 60, 60, 3, dtype=torch.float32)
        mask = torch.zeros(1, 60, 60, dtype=torch.float32)
        args = self._default_crop_args(
            img, mask=mask,
            preresize=True,
            preresize_mode="ensure minimum and maximum resolution",
            preresize_min_width=800,
            preresize_max_width=400,
            preresize_min_height=800,
            preresize_max_height=400
        )
        # Should gracefully swap min and max instead of crashing with AssertionError
        res = self.crop_node.inpaint_crop(**args)
        self.assertIsNotNone(res[0])

    def test_batch_without_target_resize_auto_handled(self):
        # Batch of 2 images with output_resize_to_target_size=False
        img_batch = torch.rand(2, 60, 60, 3, dtype=torch.float32)
        mask_batch = torch.zeros(2, 60, 60, dtype=torch.float32)
        mask_batch[0, 10:20, 10:20] = 1.0
        mask_batch[1, 20:30, 20:30] = 1.0
        args = self._default_crop_args(img_batch, mask=mask_batch, output_resize_to_target_size=False)
        # Should auto-enable target resize and process batch cleanly
        res = self.crop_node.inpaint_crop(**args)
        self.assertEqual(res[1].shape[0], 2)

    def test_flexible_batch_ratio_inpaint_stitch(self):
        # 1 stitcher with 4 inpainted candidate images
        img_1 = torch.rand(1, 60, 60, 3, dtype=torch.float32)
        mask_1 = torch.zeros(1, 60, 60, dtype=torch.float32)
        mask_1[0, 15:35, 15:35] = 1.0
        crop_res = self.crop_node.inpaint_crop(**self._default_crop_args(img_1, mask=mask_1))
        stitcher, crop_im, _ = crop_res[:3]

        # 4 variations generated from sampler
        variations_4 = crop_im.repeat(4, 1, 1, 1)
        stitch_res = self.stitch_node.inpaint_stitch(stitcher, variations_4)
        self.assertEqual(stitch_res[0].shape, (4, 60, 60, 3))

    def test_bool_and_uint8_inputs(self):
        # uint8 image (0-255) and boolean mask
        img_u8 = torch.randint(0, 256, (1, 60, 60, 3), dtype=torch.uint8)
        mask_bool = torch.zeros(1, 60, 60, dtype=torch.bool)
        mask_bool[0, 15:35, 15:35] = True

        for device_mode in ["cpu (compatible)", "gpu (much faster)"]:
            with self.subTest(device_mode=device_mode):
                args = self._default_crop_args(img_u8, mask=mask_bool, device_mode=device_mode)
                res = self.crop_node.inpaint_crop(**args)
                stitcher, crop_im, crop_m = res[:3]
                self.assertEqual(crop_im.dtype, torch.float32)
                self.assertEqual(crop_m.dtype, torch.float32)

                # uint8 inpainted image
                inpaint_u8 = (crop_im * 255).to(torch.uint8)
                stitch_res = self.stitch_node.inpaint_stitch(stitcher, inpaint_u8)
                self.assertEqual(stitch_res[0].shape, (1, 60, 60, 3))
                self.assertEqual(stitch_res[0].dtype, torch.float32)

    def test_empty_tensor_masks(self):
        img = torch.rand(1, 40, 40, 3, dtype=torch.float32)
        args = self._default_crop_args(img, mask=torch.empty(0), optional_context_mask=torch.empty(0))
        res = self.crop_node.inpaint_crop(**args)
        stitcher, crop_im, crop_m = res[:3]
        self.assertIsNotNone(stitcher)
        self.assertEqual(crop_im.shape[-1], 3)


if __name__ == '__main__':
    unittest.main()
