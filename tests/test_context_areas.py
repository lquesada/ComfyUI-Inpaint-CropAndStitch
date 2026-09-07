import unittest
import torch
from inpaint_cropandstitch import CPUProcessorLogic, GPUProcessorLogic


class TestContextAreas(unittest.TestCase):
    def setUp(self):
        self.processors = [
            ("cpu", CPUProcessorLogic()),
            ("gpu", GPUProcessorLogic()),
        ]

    def test_findcontextarea_empty_mask(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                # All zeros mask should return -1 indicator
                mask = torch.zeros(1, 40, 40, dtype=torch.float32)
                _, bx, by, bw, bh = proc.batched_findcontextarea_m(mask)
                self.assertEqual(bx[0].item(), -1)
                self.assertEqual(by[0].item(), -1)
                self.assertEqual(bw[0].item(), -1)
                self.assertEqual(bh[0].item(), -1)

    def test_findcontextarea_single_pixel_center(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                mask = torch.zeros(1, 50, 50, dtype=torch.float32)
                mask[0, 25, 25] = 1.0
                _, bx, by, bw, bh = proc.batched_findcontextarea_m(mask)
                self.assertEqual(bx[0].item(), 25)
                self.assertEqual(by[0].item(), 25)
                self.assertEqual(bw[0].item(), 1)
                self.assertEqual(bh[0].item(), 1)

    def test_findcontextarea_borders(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                # Pixel at top-left (0, 0)
                mask1 = torch.zeros(1, 50, 50, dtype=torch.float32)
                mask1[0, 0, 0] = 1.0
                _, bx1, by1, bw1, bh1 = proc.batched_findcontextarea_m(mask1)
                self.assertEqual(bx1[0].item(), 0)
                self.assertEqual(by1[0].item(), 0)

                # Pixel at bottom-right (49, 49)
                mask2 = torch.zeros(1, 50, 50, dtype=torch.float32)
                mask2[0, 49, 49] = 1.0
                _, bx2, by2, bw2, bh2 = proc.batched_findcontextarea_m(mask2)
                self.assertEqual(bx2[0].item(), 49)
                self.assertEqual(by2[0].item(), 49)

    def test_findcontextarea_multi_blob(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                mask = torch.zeros(1, 100, 100, dtype=torch.float32)
                # Two blobs
                mask[0, 10:20, 10:20] = 1.0
                mask[0, 60:80, 50:70] = 1.0
                _, bx, by, bw, bh = proc.batched_findcontextarea_m(mask)
                self.assertEqual(bx[0].item(), 10)
                self.assertEqual(by[0].item(), 10)
                self.assertEqual(bw[0].item(), 60)  # 69 - 10 + 1
                self.assertEqual(bh[0].item(), 70)  # 79 - 10 + 1

    def test_findcontextarea_batch_dimension(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                # Batch of 2 different masks
                mask = torch.zeros(2, 60, 60, dtype=torch.float32)
                mask[0, 10:20, 10:20] = 1.0
                mask[1, 30:50, 30:50] = 1.0
                _, bx, by, bw, bh = proc.batched_findcontextarea_m(mask)
                self.assertEqual(len(bx), 2)
                self.assertEqual(bx[0].item(), 10)
                self.assertEqual(bw[0].item(), 10)
                self.assertEqual(bx[1].item(), 30)
                self.assertEqual(bw[1].item(), 20)

    def test_growcontextarea(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                mask = torch.zeros(1, 100, 100, dtype=torch.float32)
                mask[0, 40:60, 40:60] = 1.0
                _, x, y, w, h = proc.batched_findcontextarea_m(mask)

                # Grow with factor 1.0 (no change)
                _, gx1, gy1, gw1, gh1 = proc.batched_growcontextarea_m(mask, x, y, w, h, extend_factor=1.0)
                self.assertEqual(gw1[0].item(), 20)
                self.assertEqual(gh1[0].item(), 20)

                # Grow with factor 2.0 (expand around center)
                _, gx2, gy2, gw2, gh2 = proc.batched_growcontextarea_m(mask, x, y, w, h, extend_factor=2.0)
                self.assertGreater(gw2[0].item(), 20)
                self.assertGreater(gh2[0].item(), 20)
                self.assertLess(gx2[0].item(), 40)
                self.assertLess(gy2[0].item(), 40)

                # Empty mask (w == -1) -> should fill entire image
                empty_w = torch.tensor([-1])
                empty_h = torch.tensor([-1])
                empty_x = torch.tensor([-1])
                empty_y = torch.tensor([-1])
                _, egx, egy, egw, egh = proc.batched_growcontextarea_m(mask, empty_x, empty_y, empty_w, empty_h, extend_factor=1.5)
                self.assertEqual(egx[0].item(), 0)
                self.assertEqual(egy[0].item(), 0)
                self.assertEqual(egw[0].item(), 100)
                self.assertEqual(egh[0].item(), 100)

    def test_combinecontextmask(self):
        for name, proc in self.processors:
            with self.subTest(processor=name):
                mask = torch.zeros(1, 100, 100, dtype=torch.float32)
                mask[0, 40:60, 40:60] = 1.0
                _, x, y, w, h = proc.batched_findcontextarea_m(mask)

                opt_mask = torch.zeros(1, 100, 100, dtype=torch.float32)
                opt_mask[0, 10:20, 10:20] = 1.0

                _, cx, cy, cw, ch = proc.batched_combinecontextmask_m(mask, x, y, w, h, opt_mask)
                self.assertEqual(cx[0].item(), 10)
                self.assertEqual(cy[0].item(), 10)
                self.assertGreaterEqual(cx[0].item() + cw[0].item(), 60)
                self.assertGreaterEqual(cy[0].item() + ch[0].item(), 60)


if __name__ == '__main__':
    unittest.main()
