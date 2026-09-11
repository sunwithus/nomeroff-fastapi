# -*- coding: utf-8 -*-
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_prep import bbox_looks_two_line  # noqa: E402
from infer_batch import yolo_chunk_size  # noqa: E402


class YoloChunkSizeTests(unittest.TestCase):
    def test_env_overrides(self):
        self.assertEqual(yolo_chunk_size("cpu", env_batch="12"), 12)
        self.assertEqual(yolo_chunk_size("cuda", free_vram_mb=8000, env_batch="3"), 3)
        self.assertEqual(yolo_chunk_size("cpu", env_batch="0"), 1)

    def test_cpu_stays_one_frame(self):
        self.assertEqual(yolo_chunk_size("cpu"), 5)
        self.assertEqual(yolo_chunk_size("cpu", free_vram_mb=8000), 5)

    def test_cuda_scales_with_free_vram(self):
        self.assertEqual(yolo_chunk_size("cuda", free_vram_mb=500), 2)
        self.assertEqual(yolo_chunk_size("cuda", free_vram_mb=1200), 4)
        self.assertEqual(yolo_chunk_size("cuda", free_vram_mb=2000), 10)
        self.assertEqual(yolo_chunk_size("cuda", free_vram_mb=4000), 20)
        self.assertEqual(yolo_chunk_size("cuda", free_vram_mb=6500), 32)


class TwoLineBboxTests(unittest.TestCase):
    def test_one_line_plate_is_wide(self):
        self.assertFalse(bbox_looks_two_line([100, 200, 260, 235]))  # 160×35 ≈ 0.22

    def test_square_two_line_plate(self):
        # Н909НР125: примерно квадратная рамка до warp
        self.assertTrue(bbox_looks_two_line([800, 500, 920, 580]))  # 120×80 ≈ 0.67

    def test_rejects_tiny_or_empty(self):
        self.assertFalse(bbox_looks_two_line(None))
        self.assertFalse(bbox_looks_two_line([1, 1, 4, 4]))


if __name__ == "__main__":
    unittest.main()
