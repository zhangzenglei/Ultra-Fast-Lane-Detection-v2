import argparse
import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import batch_infer_ufldv2 as batch
from utils.lane_comparison import (
    CameraCalibration, RecordedContourIndex, decode_recorded_contours,
    draw_recorded_contours,
)


class LaneComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.param = self.root / "camera.json"
        self.param.write_text(json.dumps({
            "cam_id": 5, "image_width": 1280, "image_height": 720,
            "focal_u": 800, "focal_v": 810, "center_u": 640, "center_v": 360,
            "distort": {"param": [0.15, -0.02, 0.001, -0.001, 0.01, 0, 0, 0] + [0] * 8},
        }))
        self.calibration = CameraCalibration(self.param)
        self.contours = self.root / "contours"
        self.contours.mkdir()
        # Nonzero third field, an off-screen point, and an empty record.
        self.payload = (struct.pack("<IIIII", 0x10005, 2, 8, 1, 1)
                        + struct.pack("<ffff", 100, 200, -20, 250)
                        + struct.pack("<III", 0, 0, 0))
        (self.contours / "record_0.json").write_text(json.dumps({
            "camera_id": 5, "timestamp": 1000, "data": list(self.payload),
        }))

    def test_recorded_coordinates_are_only_scaled_and_offscreen_points_preserved(self):
        lanes = decode_recorded_contours(self.payload, [0.5, 0.5])
        self.assertEqual(lanes[0]["points"], [[50, 100], [-10, 125]])
        self.assertEqual(lanes[0]["record_header"], [8, 1, 1])
        self.assertEqual(lanes[1]["points"], [])
        image = np.zeros((180, 320, 3), dtype=np.uint8)
        draw_recorded_contours(image, {"lanes": lanes})
        self.assertEqual(image[100, 50].tolist(), [255, 255, 0])
        self.assertFalse(image[:, 0].any())

    def test_malformed_contours_fail_instead_of_drawing_partial_results(self):
        for payload in (b"", self.payload[:-1], self.payload + b"\x00",
                        struct.pack("<II", 123, 0)):
            with self.subTest(payload=payload[:8]), self.assertRaises(ValueError):
                decode_recorded_contours(payload, [0.5, 0.5])

    def test_timestamp_units_tolerance_and_valid_empty_frame(self):
        index = RecordedContourIndex(self.contours, 5, tolerance_ms=1)
        matched = index.match("image_1000094.jpg", [0.5, 0.5])
        self.assertTrue(matched["matched"])
        self.assertEqual(matched["nearest_delta_us"], 94)
        self.assertTrue(index.match("image_1001000.jpg", [0.5, 0.5])["matched"])
        self.assertFalse(index.match("image_1001001.jpg", [0.5, 0.5])["matched"])
        with self.assertRaises(ValueError):
            index.match("image_without_timestamp.jpg", [0.5, 0.5])
        with self.assertRaises(ValueError):
            RecordedContourIndex(self.contours, 4)
        (self.contours / "record_1.json").write_text(json.dumps({
            "camera_id": 5, "timestamp": 2000,
            "data": list(struct.pack("<II", 0x10005, 0)),
        }))
        empty = RecordedContourIndex(self.contours, 5).match("image_2000094.jpg", [1, 1])
        self.assertTrue(empty["matched"])
        self.assertEqual(empty["lanes"], [])

    def test_remapping_uses_scaled_original_K_without_cropping(self):
        rgb = np.random.default_rng(42).integers(0, 256, (180, 320, 3), dtype=np.uint8)
        matrix = np.array([[200, 0, 160], [0, 202.5, 90], [0, 0, 1]])
        np.testing.assert_allclose(self.calibration.scaled_matrix(320, 180), matrix)
        expected = cv2.undistort(rgb, matrix, self.calibration.distortion, None, matrix)
        actual = self.calibration.undistort(rgb)
        # OpenCV's tiled undistort implementation may round interpolation differently.
        self.assertLess(np.abs(actual.astype(float) - expected.astype(float)).mean(), 0.1)
        self.assertEqual(actual.shape, rgb.shape)
        self.calibration.undistort(rgb)
        self.assertEqual(len(self.calibration._maps), 1)

    def test_pipeline_uses_undistorted_model_input_and_saves_only_final_pairs(self):
        source = self.root / "images"
        source.mkdir()
        rgb = np.random.default_rng(4).integers(0, 256, (360, 640, 3), dtype=np.uint8)
        for timestamp in (1000094, 2000094):
            Image.fromarray(rgb).save(source / f"image_{timestamp}.png")
        args = argparse.Namespace(
            config="configs/culane_res34.py", weights="unused.pth", input=str(source),
            output=str(self.root / "output"), camera_param=str(self.param),
            recorded_contours=str(self.contours), match_tolerance_ms=1, limit=None)
        model = Mock(return_value={})
        expected = transforms.Compose([
            transforms.Resize((int(320 / 0.6), 1600)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])(Image.fromarray(self.calibration.undistort(rgb)))[:, -320:, :].unsqueeze(0)
        with (patch.object(batch.torch.cuda, "is_available", return_value=True),
              patch.object(batch.torch.cuda, "synchronize"),
              patch.object(torch.Tensor, "cuda", lambda tensor: tensor),
              patch.object(batch, "get_culane_model", return_value=model),
              patch.object(batch, "load_checkpoint", return_value={}),
              patch.object(batch, "decode", return_value=[]),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(batch.infer(args), 0)
        torch.testing.assert_close(model.call_args_list[0].args[0], expected)
        output = Path(args.output)
        self.assertEqual(sorted(p.suffix for p in output.iterdir()),
                         [".jpg", ".jpg", ".json", ".json"])
        saved = json.loads((output / "image_1000094_lanes.json").read_text())
        self.assertEqual(saved["coordinate_space"], "undistorted")
        self.assertEqual(saved["calibration"]["camera_matrix"][0], [400, 0, 320])
        self.assertEqual(saved["recorded"]["lanes"][0]["points"][0], [50, 100])
        missing = json.loads((output / "image_2000094_lanes.json").read_text())
        self.assertFalse(missing["recorded"]["matched"])
        result = cv2.imread(str(output / "image_1000094_lanes.jpg"))
        # Away from the legend and markers, the saved background is undistorted.
        background = cv2.cvtColor(self.calibration.undistort(rgb), cv2.COLOR_RGB2BGR)
        encoded = cv2.imencode(".jpg", background, [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
        expected_background = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(result[200:, 200:], expected_background[200:, 200:])


if __name__ == "__main__":
    unittest.main()
