import unittest

import numpy as np

from fall_detection.image_utils import ensure_3_channels, resize_keep_aspect, to_infrared


class ToInfraredTest(unittest.TestCase):
    def test_bgr_to_infrared(self) -> None:
        bgr = np.array([[[100, 150, 200]]], dtype=np.uint8)
        result = to_infrared(bgr)
        self.assertEqual(result.shape, (1, 1, 3))
        # All three channels should be the same grayscale value
        self.assertEqual(result[0, 0, 0], result[0, 0, 1])
        self.assertEqual(result[0, 0, 1], result[0, 0, 2])

    def test_gray_to_infrared(self) -> None:
        gray = np.array([[128]], dtype=np.uint8)
        result = to_infrared(gray)
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertEqual(result[0, 0, 0], 128)
        self.assertEqual(result[0, 0, 1], 128)
        self.assertEqual(result[0, 0, 2], 128)

    def test_bgra_to_infrared(self) -> None:
        bgra = np.array([[[100, 150, 200, 255]]], dtype=np.uint8)
        result = to_infrared(bgra)
        self.assertEqual(result.shape, (1, 1, 3))
        self.assertEqual(result[0, 0, 0], result[0, 0, 1])
        self.assertEqual(result[0, 0, 1], result[0, 0, 2])

    def test_infrared_is_grayscale_replicated(self) -> None:
        bgr = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        result = to_infrared(bgr)
        np.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 1], result[:, :, 2])

    def test_unsupported_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            to_infrared(np.zeros((10, 10, 5), dtype=np.uint8))

    def test_output_dtype(self) -> None:
        bgr = np.array([[[100, 150, 200]]], dtype=np.uint8)
        result = to_infrared(bgr)
        self.assertEqual(result.dtype, np.uint8)


class Ensure3ChannelsTest(unittest.TestCase):
    def test_3_channel_passthrough(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = ensure_3_channels(img)
        self.assertEqual(result.shape, (100, 200, 3))
        np.testing.assert_array_equal(result, img)

    def test_2d_expands_to_3_channels(self) -> None:
        gray = np.full((100, 200), 128, dtype=np.uint8)
        result = ensure_3_channels(gray)
        self.assertEqual(result.shape, (100, 200, 3))
        np.testing.assert_array_equal(result[:, :, 0], gray)
        np.testing.assert_array_equal(result[:, :, 1], gray)
        np.testing.assert_array_equal(result[:, :, 2], gray)

    def test_1_channel_expands(self) -> None:
        img = np.full((100, 200, 1), 64, dtype=np.uint8)
        result = ensure_3_channels(img)
        self.assertEqual(result.shape, (100, 200, 3))

    def test_4_channel_converts(self) -> None:
        bgra = np.zeros((100, 200, 4), dtype=np.uint8)
        bgra[:, :, 3] = 255
        result = ensure_3_channels(bgra)
        self.assertEqual(result.shape, (100, 200, 3))

    def test_unsupported_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            ensure_3_channels(np.zeros((10, 10, 5), dtype=np.uint8))


class ResizeKeepAspectTest(unittest.TestCase):
    def test_same_size_noop(self) -> None:
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = resize_keep_aspect(img, 640)
        self.assertEqual(result.shape[0], 640)
        self.assertEqual(result.shape[1], 640)

    def test_downscale_landscape(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = resize_keep_aspect(img, 320)
        self.assertLessEqual(result.shape[0], 320)
        self.assertLessEqual(result.shape[1], 320)
        self.assertGreater(result.shape[0], 0)
        self.assertGreater(result.shape[1], 0)

    def test_downscale_portrait(self) -> None:
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result = resize_keep_aspect(img, 320)
        self.assertLessEqual(result.shape[0], 320)
        self.assertLessEqual(result.shape[1], 320)

    def test_preserves_channel_count(self) -> None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = resize_keep_aspect(img, 320)
        self.assertEqual(result.shape[2], 3)

    def test_2d_image(self) -> None:
        img = np.zeros((480, 640), dtype=np.uint8)
        result = resize_keep_aspect(img, 320)
        self.assertEqual(len(result.shape), 2)
        self.assertLessEqual(result.shape[0], 320)


if __name__ == "__main__":
    unittest.main()