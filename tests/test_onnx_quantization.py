from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from fall_detection.onnx_preprocess import letterbox, prepare_onnx_input
from scripts.model.quantize_onnx import build_calibration_entries, evenly_select


class OnnxPreprocessTests(unittest.TestCase):
    def test_letterbox_produces_fixed_square(self) -> None:
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = letterbox(image, 384)
        self.assertEqual(result.shape, (384, 384, 3))

    def test_prepare_input_is_normalized_rgb_nchw(self) -> None:
        bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
        result = prepare_onnx_input(bgr, size=1)
        self.assertEqual(result.shape, (1, 3, 1, 1))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[0, :, 0, 0], [1.0, 0.0, 0.0])

    def test_pseudo_infrared_channels_match(self) -> None:
        image = np.random.default_rng(2026).integers(
            0,
            256,
            size=(32, 48, 3),
            dtype=np.uint8,
        )
        result = prepare_onnx_input(image, size=64, infrared=True)
        np.testing.assert_array_equal(result[:, 0], result[:, 1])
        np.testing.assert_array_equal(result[:, 1], result[:, 2])

    def test_invalid_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            letterbox(np.zeros((2, 2, 3), dtype=np.uint8), 0)


class CalibrationSelectionTests(unittest.TestCase):
    def test_even_selection_includes_ends(self) -> None:
        paths = [Path(f"{index}.jpg") for index in range(10)]
        selected = evenly_select(paths, 3)
        self.assertEqual(selected, [paths[0], paths[4], paths[9]])

    def test_visible_and_infrared_counts(self) -> None:
        paths = [Path(f"{index}.jpg") for index in range(100)]
        entries = build_calibration_entries(paths, 64, 0.5)
        self.assertEqual(len(entries), 64)
        self.assertEqual(sum(not infrared for _, infrared in entries), 32)
        self.assertEqual(sum(infrared for _, infrared in entries), 32)

    def test_invalid_ratio_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_calibration_entries([Path("image.jpg")], 1, 1.1)


if __name__ == "__main__":
    unittest.main()
