"""Calibration and recorded-contour support for the front-mono image topic."""
import bisect
import json
import struct
from pathlib import Path

import cv2
import numpy as np


class CameraCalibration:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.width = int(data["image_width"])
        self.height = int(data["image_height"])
        self.camera_id = data["cam_id"]
        self.matrix = np.array([
            [data["focal_u"], 0, data["center_u"]],
            [0, data["focal_v"], data["center_v"]],
            [0, 0, 1],
        ], dtype=np.float64)
        params = np.asarray(data["distort"]["param"], dtype=np.float64)
        if params.ndim != 1 or params.size < 8 or np.any(params[8:] != 0):
            raise ValueError("Expected 8 distortion coefficients with optional zero padding")
        self.distortion = params[:8]  # k1, k2, p1, p2, k3, k4, k5, k6
        if (self.width <= 0 or self.height <= 0
                or not np.isfinite(self.matrix).all()
                or not np.isfinite(self.distortion).all()
                or self.matrix[0, 0] <= 0 or self.matrix[1, 1] <= 0):
            raise ValueError("Invalid camera calibration")
        self._maps = {}

    def scale(self, width, height):
        return np.array([width / self.width, height / self.height])

    def scaled_matrix(self, width, height):
        # Full-frame resizing only: no ROI crop or principal-point translation.
        return np.diag([*self.scale(width, height), 1.0]) @ self.matrix

    def undistort(self, rgb):
        height, width = rgb.shape[:2]
        size = (width, height)
        if size not in self._maps:
            matrix = self.scaled_matrix(width, height)
            self._maps[size] = cv2.initUndistortRectifyMap(
                matrix, self.distortion, None, matrix, size, cv2.CV_32FC1)
        return cv2.remap(rgb, *self._maps[size], interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def metadata(self, width, height):
        return {
            "source": str(self.path),
            "calibration_size": [self.width, self.height],
            "camera_matrix": self.scaled_matrix(width, height).tolist(),
            "distortion": self.distortion.tolist(),
            "distortion_order": ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"],
            "cropped": False,
        }


def decode_recorded_contours(values, scale):
    """Decode the observed little-endian front-mono contour byte layout.

    The topic header is followed by a record count. Each record has three u32
    fields, then (field_1 + field_2) float32 coordinate pairs. Preserve the raw
    fields: their semantic names and point connectivity are not assumed here.
    Recorded points are ALREADY undistorted under the original calibration K.
    """
    payload = bytes(values)
    if len(payload) < 8:
        raise ValueError("Truncated contour topic header")
    header, count = struct.unpack_from("<II", payload)
    if header != 0x00010005:
        raise ValueError(f"Unsupported contour topic header: {header:#x}")
    offset = 8
    lanes = []
    for record_id in range(count):
        if offset + 12 > len(payload):
            raise ValueError("Truncated contour record header")
        fields = struct.unpack_from("<III", payload, offset)
        offset += 12
        point_count = fields[1] + fields[2]
        end = offset + point_count * 8
        if end > len(payload):
            raise ValueError("Truncated contour points")
        points = np.frombuffer(payload, dtype="<f4", count=point_count * 2,
                               offset=offset).reshape(-1, 2).astype(np.float64)
        if not np.isfinite(points).all():
            raise ValueError("Non-finite recorded contour coordinate")
        points *= scale
        lanes.append({"record_id": record_id, "record_header": list(fields),
                      "points": points.tolist()})
        offset = end
    if offset != len(payload):
        raise ValueError("Unexpected trailing contour bytes")
    return lanes


class RecordedContourIndex:
    def __init__(self, directory, camera_id, tolerance_ms=1.0):
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        if not np.isfinite(tolerance_ms) or tolerance_ms < 0:
            raise ValueError("Match tolerance must be finite and nonnegative")
        self.tolerance_us = round(tolerance_ms * 1000)
        entries = []
        for path in sorted(directory.rglob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            timestamp = data["timestamp"]
            if type(timestamp) is not int or timestamp <= 0:
                raise ValueError(f"Expected a positive millisecond timestamp: {path}")
            if data["camera_id"] != camera_id:
                raise ValueError(f"Camera ID does not match calibration: {path}")
            entries.append((timestamp * 1000, path))
        if not entries:
            raise RuntimeError(f"No recorded contour JSON files found: {directory}")
        entries.sort()
        self.timestamps, self.paths = map(list, zip(*entries))

    def match(self, image_path, scale):
        try:
            timestamp = int(Path(image_path).stem.rsplit("_", 1)[-1])
        except ValueError as exc:
            raise ValueError(f"Image name must end in a microsecond timestamp: {image_path}") from exc
        index = bisect.bisect_left(self.timestamps, timestamp)
        candidates = range(max(0, index - 1), min(len(self.timestamps), index + 1))
        nearest = min(candidates, key=lambda i: (abs(timestamp - self.timestamps[i]), i))
        delta = timestamp - self.timestamps[nearest]
        result = {"matched": abs(delta) <= self.tolerance_us,
                  "image_timestamp_us": timestamp,
                  "nearest_delta_us": delta, "tolerance_us": self.tolerance_us,
                  "lanes": []}
        if result["matched"]:
            path = self.paths[nearest]
            data = json.loads(path.read_text(encoding="utf-8"))
            result.update(source=str(path), timestamp_ms=self.timestamps[nearest] // 1000,
                          lanes=decode_recorded_contours(data["data"], scale))
        return result


def draw_recorded_contours(image, recorded, color=(255, 255, 0)):
    # Draw sampled points only; do not invent connectivity across unknown groups.
    height, width = image.shape[:2]
    for lane in recorded["lanes"]:
        for x, y in lane["points"]:
            # Keep off-screen coordinates in JSON, but do not clamp them to edges.
            if 0 <= x < width and 0 <= y < height:
                point = (min(width - 1, round(x)), min(height - 1, round(y)))
                cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)


def draw_comparison_legend(image, recorded):
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (min(image.shape[1] - 1, 950), 78), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, dst=image)
    cv2.putText(image, "UFLDv2: orange | Recorded: cyan", (15, 29),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if recorded["matched"]:
        text = (f"Recorded: MATCH | image - contour = {recorded['nearest_delta_us']:+d} us"
                f" | {len(recorded['lanes'])} contours")
    else:
        text = f"Recorded: NO MATCH | nearest delta = {recorded['nearest_delta_us']:+d} us"
    cv2.putText(image, text, (15, 61), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
