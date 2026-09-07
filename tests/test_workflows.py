import os
import unittest
import torch
from tests.workflow_runner import (
    WorkflowRunner,
    validate_workflow_run,
    validate_crop_outputs,
    validate_stitch_outputs,
    MockLoadImage,
    MockMaskToImage,
    MockImageInvert,
    MockImpactMakeImageBatch,
    MockImpactMakeMaskBatch,
    MockImageCompositeMasked,
)


class TestMockNodes(unittest.TestCase):
    """Unit tests for the self-contained mock nodes used in workflow execution."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.testimgs_dir = os.path.join(cls.repo_root, "testimgs")

    def test_mock_load_image_rgb(self):
        loader = MockLoadImage(self.testimgs_dir)
        img, mask = loader.load_image("example.png")
        self.assertEqual(img.ndim, 4)
        self.assertEqual(img.shape[0], 1)
        self.assertEqual(img.shape[-1], 3)
        self.assertEqual(mask.ndim, 3)
        self.assertEqual(mask.shape, (1, 64, 64))  # No alpha -> 64x64 zero mask
        self.assertEqual(mask.sum().item(), 0.0)

    def test_mock_load_image_clipspace_rgba(self):
        loader = MockLoadImage(self.testimgs_dir)
        img, mask = loader.load_image("clipspace/clipspace-mask-105444.59999999404.png [input]")
        self.assertEqual(img.ndim, 4)
        self.assertEqual(img.shape[0], 1)
        self.assertEqual(img.shape[-1], 3)
        self.assertEqual(mask.ndim, 3)
        self.assertEqual(mask.shape[1:], img.shape[1:3])
        self.assertTrue((mask >= 0.0).all() and (mask <= 1.0).all())

    def test_mock_load_image_not_found(self):
        loader = MockLoadImage(self.testimgs_dir)
        with self.assertRaises(FileNotFoundError):
            loader.load_image("non_existent_image_12345.png")

    def test_mock_mask_to_image(self):
        node = MockMaskToImage()
        mask_3d = torch.rand(2, 32, 48)
        (img_4d,) = node.mask_to_image(mask_3d)
        self.assertEqual(img_4d.shape, (2, 32, 48, 3))
        # Channels should be identical grayscale replicated
        self.assertTrue(torch.equal(img_4d[..., 0], img_4d[..., 1]))
        self.assertTrue(torch.equal(img_4d[..., 1], img_4d[..., 2]))

        mask_2d = torch.rand(32, 48)
        (img_from_2d,) = node.mask_to_image(mask_2d)
        self.assertEqual(img_from_2d.shape, (1, 32, 48, 3))

    def test_mock_image_invert(self):
        node = MockImageInvert()
        img = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])
        (inv,) = node.invert(img)
        self.assertTrue(torch.allclose(inv, 1.0 - img))

    def test_mock_impact_image_batch(self):
        node = MockImpactMakeImageBatch()
        img1 = torch.rand(1, 16, 16, 3)
        img2 = torch.rand(2, 16, 16, 3)
        (batched,) = node.make_batch(image1=img1, image2=img2, image3=None)
        self.assertEqual(batched.shape, (3, 16, 16, 3))

    def test_mock_impact_mask_batch(self):
        node = MockImpactMakeMaskBatch()
        m1 = torch.rand(1, 16, 16)
        m2 = torch.rand(16, 16)  # 2D mask
        (batched,) = node.make_batch(mask1=m1, mask2=m2, mask3=None)
        self.assertEqual(batched.shape, (2, 16, 16))

    def test_mock_image_composite_masked(self):
        node = MockImageCompositeMasked()
        dest = torch.zeros(1, 40, 40, 3)
        src = torch.ones(1, 20, 20, 3)
        mask = torch.ones(1, 20, 20)
        (res,) = node.composite(dest, src, x=10, y=10, mask=mask)
        self.assertEqual(res.shape, (1, 40, 40, 3))
        # Inner region (10:30, 10:30) should be 1.0, outer should be 0.0
        self.assertAlmostEqual(res[0, 15, 15, 0].item(), 1.0)
        self.assertAlmostEqual(res[0, 0, 0, 0].item(), 0.0)


class TestWorkflowExecution(unittest.TestCase):
    """Executes testscpu.json and testsgpu.json end-to-end and validates outputs."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.cpu_path = os.path.join(cls.repo_root, "testscpu.json")
        cls.gpu_path = os.path.join(cls.repo_root, "testsgpu.json")
        cls.testimgs_dir = os.path.join(cls.repo_root, "testimgs")
        cls.runner = WorkflowRunner(testimgs_dir=cls.testimgs_dir, verbose=False)
        # Run each workflow once and cache for all assertion methods
        cls.cpu_result = cls.runner.run_file(cls.cpu_path)
        cls.gpu_result = cls.runner.run_file(cls.gpu_path)

    def test_testscpu_workflow_execution_and_outputs(self):
        """Execute all 793 nodes in testscpu.json and validate outputs."""
        result = self.cpu_result
        self.assertEqual(len(result.nodes), 793)
        self.assertEqual(len(result.outputs), 793)
        self.assertEqual(len(result.crop_node_ids), 105)
        self.assertEqual(len(result.stitch_node_ids), 34)
        self.assertEqual(len(result.preview_node_ids), 349)
        self.assertEqual(len(result.load_node_ids), 78)

        # Full validation across all crop, stitch, and preview nodes
        validate_workflow_run(result)

        # Specific sample checks:
        # Check node 15 (InpaintCropImproved)
        crop_15_outs = result.outputs[15]
        self.assertEqual(len(crop_15_outs), 26)
        stitcher_15 = crop_15_outs[0]
        self.assertEqual(stitcher_15["device_mode"], "cpu (compatible)")
        self.assertIn("canvas_image", stitcher_15)
        cropped_img_15 = crop_15_outs[1]
        self.assertEqual(cropped_img_15.ndim, 4)
        self.assertEqual(cropped_img_15.shape[-1], 3)

        # Check node 478 (InpaintStitchImproved)
        stitch_478_out = result.outputs[478][0]
        self.assertEqual(stitch_478_out.ndim, 4)
        self.assertEqual(stitch_478_out.shape[-1], 3)
        self.assertTrue((stitch_478_out >= 0.0).all() and (stitch_478_out <= 1.0).all())

    def test_testsgpu_workflow_execution_and_outputs(self):
        """Execute all 793 nodes in testsgpu.json and validate outputs."""
        result = self.gpu_result
        self.assertEqual(len(result.nodes), 793)
        self.assertEqual(len(result.outputs), 793)
        self.assertEqual(len(result.crop_node_ids), 105)
        self.assertEqual(len(result.stitch_node_ids), 34)
        self.assertEqual(len(result.preview_node_ids), 349)

        # Full validation across all crop, stitch, and preview nodes
        validate_workflow_run(result)

        # Verify device mode was passed through stitcher
        sample_crop_id = result.crop_node_ids[0]
        stitcher = result.outputs[sample_crop_id][0]
        self.assertEqual(stitcher["device_mode"], "gpu (much faster)")

    def test_cpu_gpu_workflow_consistency(self):
        """Compare CPU vs GPU workflows: output shapes and coordinate consistency."""
        cpu_res = self.cpu_result
        gpu_res = self.gpu_result

        # Verify all PreviewImage nodes have identical output shapes
        for nid in cpu_res.preview_node_ids:
            cpu_tensor = cpu_res.outputs[nid][0]
            gpu_tensor = gpu_res.outputs[nid][0]
            self.assertEqual(
                cpu_tensor.shape,
                gpu_tensor.shape,
                f"Preview node {nid} shape mismatch between CPU and GPU",
            )

        # Verify all InpaintStitchImproved nodes have identical shapes
        for nid in cpu_res.stitch_node_ids:
            cpu_stitched = cpu_res.outputs[nid][0]
            gpu_stitched = gpu_res.outputs[nid][0]
            self.assertEqual(
                cpu_stitched.shape,
                gpu_stitched.shape,
                f"Stitch node {nid} shape mismatch between CPU and GPU",
            )


if __name__ == "__main__":
    unittest.main()
