from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from openvino import Core

from ..config import Config
from ..jobs import PipelineResult
from ..logging_utils import debug_trace, get_logger
from ..schemas import ArtifactPaths, JobStats

logger = get_logger("pipeline")


@dataclass
class FrameMeasurement:
    frame_idx: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2
    detection_conf: float = 0.0
    filtered_xy: Optional[Tuple[float, float]] = None


class OpenVINOYoloDetector:
    """Performs YOLOv8 inference via OpenVINO."""

    def __init__(self, config: Config, device: Optional[str] = None) -> None:
        self.config = config
        self.core = Core()
        self.device = (device or config.runtime.device).upper()
        xml_path = str(config.model.xml)
        bin_path = str(config.model.bin) if config.model.bin else None
        if bin_path:
            model = self.core.read_model(model=xml_path, weights=bin_path)
        else:
            model = self.core.read_model(model=xml_path)
        properties = {}
        if config.runtime.num_streams > 0:
            properties["NUM_STREAMS"] = str(config.runtime.num_streams)
        self.compiled = self.core.compile_model(model=model, device_name=self.device, config=properties)
        self.input_tensor = self.compiled.input(0)
        self.output_tensors = [self.compiled.output(i) for i in range(len(self.compiled.outputs))]
        _, _, self.input_height, self.input_width = self.input_tensor.shape
        self.conf_thres = config.model.conf_thres
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Detector initialised xml=%s bin=%s device=%s streams=%s conf_thres=%.2f",
                config.model.xml,
                config.model.bin,
                self.device,
                properties.get("NUM_STREAMS", "default"),
                self.conf_thres,
            )

    @debug_trace(name="OpenVINOYoloDetector.detect")
    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float]]:
        """Returns list of (bbox_xyxy, confidence) detections for class 0."""
        blob, ratio, pad = self._preprocess(frame)
        predictions = self.compiled.infer_new_request({self.input_tensor.any_name: blob})
        detections: List[Tuple[Tuple[float, float, float, float], float]] = []

        if len(self.output_tensors) == 2:
            boxes = predictions[self.output_tensors[0]]
            scores = predictions[self.output_tensors[1]]
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            if boxes.ndim == 1:
                boxes = np.expand_dims(boxes, 0)
            if scores.ndim == 1:
                scores = np.expand_dims(scores, 0)
            for box, cls_scores in zip(boxes, scores):
                conf = float(np.max(cls_scores))
                if conf < self.conf_thres:
                    continue
                xyxy = self._scale_coords(box, ratio, pad, frame.shape[:2])
                detections.append((xyxy, conf))
        else:
            output = predictions[self.output_tensors[0]]
            output = np.squeeze(output)
            if output.ndim == 2:
                preds = output
            elif output.ndim == 3:
                preds = output.reshape(output.shape[1], output.shape[2])
            else:
                preds = output.reshape(-1, output.shape[-1])
            if preds.shape[0] == 84 and preds.shape[1] != 84:
                preds = preds.T
            for row in preds:
                if row.shape[0] < 6:
                    continue
                scores = row[4:]
                conf = float(np.max(scores))
                if conf < self.conf_thres:
                    continue
                xyxy = self._scale_coords(row[:4], ratio, pad, frame.shape[:2])
                detections.append((xyxy, conf))

        detections.sort(key=lambda item: item[1], reverse=True)
        return detections

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        img = frame.copy()
        h, w = img.shape[:2]
        r = min(self.input_width / w, self.input_height / h)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_w = self.input_width - new_w
        pad_h = self.input_height - new_h
        top = pad_h // 2
        left = pad_w // 2
        bottom = pad_h - top
        right = pad_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        blob = padded.transpose((2, 0, 1))[np.newaxis, ...].astype(np.float32) / 255.0
        return blob, r, (left, top)

    def _scale_coords(
        self,
        box: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
        original_shape: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        if box.shape[0] >= 6:
            # Some exports provide x1,y1,x2,y2 directly
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        else:
            x_center, y_center, width, height = box[0], box[1], box[2], box[3]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

        x1 -= pad[0]
        y1 -= pad[1]
        x2 -= pad[0]
        y2 -= pad[1]
        x1 /= ratio
        y1 /= ratio
        x2 /= ratio
        y2 /= ratio
        h, w = original_shape
        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y2 = float(np.clip(y2, 0, h - 1))
        return (x1, y1, x2, y2)


class KalmanPointTracker:
    """Single-object constant-velocity Kalman tracker."""

    def __init__(self, max_age: int = 15) -> None:
        self.F = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.R = np.eye(2, dtype=np.float32) * 0.5
        self.state: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.last_update: Optional[int] = None
        self.max_age = max_age

    @debug_trace(name="KalmanPointTracker.update")
    def update(self, frame_idx: int, measurement: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if self.state is None:
            if measurement is None:
                return None
            self.state = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)
            self.P = np.eye(4, dtype=np.float32)
            self.last_update = frame_idx
            return measurement

        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        result = self.state[:2].copy()
        if measurement is not None:
            z = np.array([measurement[0], measurement[1]], dtype=np.float32)
            y = z - (self.H @ self.state)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.state = self.state + K @ y
            self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
            result = self.state[:2].copy()
            self.last_update = frame_idx
        elif self.last_update is not None and frame_idx - self.last_update > self.max_age:
            self.state = None
            self.P = None
            return None

        return float(result[0]), float(result[1])


def moving_average(series: List[float], window: int) -> List[float]:
    if not series:
        return []
    window = max(window, 1)
    if window % 2 == 0:
        window += 1
    if window == 1 or len(series) < window:
        return series.copy()
    arr = np.array(series, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="same")
    # Remove padding
    smoothed = smoothed[window // 2 : -window // 2]
    return smoothed.tolist()


class AnalysisPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.detectors: Dict[str, OpenVINOYoloDetector] = {}

    @debug_trace(name="AnalysisPipeline.run")
    def run(
        self,
        job_id: str,
        input_video: Path,
        output_dir: Path,
        device: Optional[str],
        progress_cb,
    ) -> PipelineResult:
        def emit(stage: str, value: float, message: Optional[str] = None) -> None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Stage=%s progress=%.2f message=%s", stage, value, message)
            progress_cb(stage, value, message)

        requested_device = (device or self.config.runtime.device).upper()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Starting analysis job_id=%s requested_device=%s input=%s",
                job_id,
                requested_device,
                input_video,
            )

        tracker = KalmanPointTracker(max_age=self.config.pipeline.max_age)
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {input_video}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        measurements: List[FrameMeasurement] = []
        frame_idx = 0

        emit("detecting", 0.1)

        detector, device_name = self._initialise_detector(requested_device, emit)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            best_detection = detections[0] if detections else None

            measurement_xy: Optional[Tuple[float, float]] = None
            detection_conf = 0.0
            if best_detection:
                bbox, detection_conf = best_detection
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                measurement_xy = (cx, cy)
            filtered = tracker.update(frame_idx, measurement_xy)

            measurements.append(
                FrameMeasurement(
                    frame_idx=frame_idx,
                    bbox=best_detection[0] if best_detection else None,
                    detection_conf=detection_conf,
                    filtered_xy=filtered,
                )
            )

            frame_idx += 1
            if total_frames:
                emit("detecting", min(0.1 + 0.6 * (frame_idx / total_frames), 0.7))

        cap.release()

        if not any(m.filtered_xy for m in measurements):
            raise RuntimeError("No club head detections found in the provided video.")

        measurement_map: Dict[int, FrameMeasurement] = {m.frame_idx: m for m in measurements}

        # Smooth trajectories
        filtered_points: List[Tuple[int, Tuple[float, float]]] = [
            (m.frame_idx, m.filtered_xy) for m in measurements if m.filtered_xy
        ]  # filtered_xy guaranteed not None
        frame_numbers = [frame for frame, _ in filtered_points]
        xs = [pt[0] for _, pt in filtered_points]
        ys = [pt[1] for _, pt in filtered_points]

        smoothed_xs = moving_average(xs, self.config.pipeline.smoothing_window)
        smoothed_ys = moving_average(ys, self.config.pipeline.smoothing_window)

        if (
            len(smoothed_xs) != len(frame_numbers)
            or len(smoothed_ys) != len(frame_numbers)
        ):
            logger.warning(
                "Smoothing output length mismatch (frames=%s xs=%s ys=%s); "
                "using raw filtered points for missing entries",
                len(frame_numbers),
                len(smoothed_xs),
                len(smoothed_ys),
            )

        smoothed_map: Dict[int, Tuple[float, float]] = {}
        for idx, frame_no in enumerate(frame_numbers):
            base_point = filtered_points[idx][1]
            x_val = smoothed_xs[idx] if idx < len(smoothed_xs) else base_point[0]
            y_val = smoothed_ys[idx] if idx < len(smoothed_ys) else base_point[1]
            smoothed_map[frame_no] = (x_val, y_val)

        # Compute speed metrics
        scale_mm_per_px = self.config.calibration.scale_mm_per_px
        speeds: Dict[int, float] = {}
        max_speed = 0.0
        impact_frame: Optional[int] = None
        prev_pt: Optional[Tuple[float, float]] = None
        prev_frame: Optional[int] = None
        for frame_no in frame_numbers:
            point = smoothed_map.get(frame_no)
            if point is None:
                continue
            if prev_pt is None or prev_frame is None:
                speeds[frame_no] = 0.0
                prev_pt = point
                prev_frame = frame_no
                continue
            dt = (frame_no - prev_frame) / fps
            if dt <= 0:
                speeds[frame_no] = speeds.get(prev_frame, 0.0)
                prev_frame = frame_no
                prev_pt = point
                continue
            distance_px = math.dist(point, prev_pt)
            distance_mm = distance_px * scale_mm_per_px
            speed_mps = (distance_mm / 1000.0) / dt
            speeds[frame_no] = speed_mps
            if speed_mps > max_speed:
                max_speed = speed_mps
                impact_frame = frame_no
            prev_pt = point
            prev_frame = frame_no

        avg_speed = float(np.mean(list(speeds.values()))) if speeds else 0.0
        duration_s = frame_numbers[-1] / fps if frame_numbers else 0.0

        emit("exporting", 0.75)

        # Export artifacts
        job_video_path = output_dir / "result.mp4"
        trajectory_csv_path = output_dir / "trajectory.csv"
        trajectory_json_path = output_dir / "trajectory.json"
        stats_json_path = output_dir / "stats.json"
        snapshots: List[str] = []

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError("Failed to re-open video for overlay export.")

        # Prefer H.264 for web playback; try common FourCCs then fall back.
        chosen_fourcc = None
        writer = None
        for cc in ["avc1", "H264", "X264", "h264", "mp4v"]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*cc)
                candidate = cv2.VideoWriter(str(job_video_path), fourcc, fps, (width, height))
                if candidate.isOpened():
                    writer = candidate
                    chosen_fourcc = cc
                    break
                else:
                    candidate.release()
            except Exception:  # noqa: BLE001
                pass
        if writer is None:
            raise RuntimeError("Failed to initialise VideoWriter for output video.")

        frame_idx = 0

        impact_snapshot = output_dir / "impact.png"
        first_snapshot = output_dir / "address.png"

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            measurement = measurement_map.get(frame_idx)
            bbox_to_draw: Optional[Tuple[float, float, float, float]] = None
            if measurement and measurement.bbox:
                bbox_to_draw = measurement.bbox

            point = smoothed_map.get(frame_idx)
            if bbox_to_draw is None and point:
                # Fallback: draw a fixed-size box around the tracked point
                half_size = 25
                cx, cy = point
                bbox_to_draw = (cx - half_size, cy - half_size, cx + half_size, cy + half_size)

            if bbox_to_draw:
                x1, y1, x2, y2 = bbox_to_draw
                x1_i = max(0, min(frame_w - 1, int(round(x1))))
                y1_i = max(0, min(frame_h - 1, int(round(y1))))
                x2_i = max(0, min(frame_w - 1, int(round(x2))))
                y2_i = max(0, min(frame_h - 1, int(round(y2))))
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (255, 0, 0), 2)
                confidence = measurement.detection_conf if measurement else 0.0
                label = f"{confidence:.2f}"
                text_y = max(0, min(frame_h - 1, y1_i - 10))
                cv2.putText(
                    frame,
                    label,
                    (x1_i, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if point:
                # Highlight the club head position with a marker
                cv2.circle(frame, tuple(map(int, point)), 6, (0, 0, 255), -1)
            writer.write(frame)

            if frame_numbers and frame_idx == frame_numbers[0] and first_snapshot.name not in snapshots and point:
                cv2.imwrite(str(first_snapshot), frame)
                snapshots.append(first_snapshot.name)
            if impact_frame is not None and frame_idx == impact_frame and impact_snapshot.name not in snapshots and point:
                cv2.imwrite(str(impact_snapshot), frame)
                snapshots.append(impact_snapshot.name)

            frame_idx += 1

        cap.release()
        writer.release()

        # If we could not use an H.264 FourCC directly, try ffmpeg re-encode to H.264.
        try:
            if chosen_fourcc not in {"avc1", "H264", "X264", "h264"}:
                import shutil
                import subprocess

                ffmpeg = shutil.which("ffmpeg")
                if ffmpeg:
                    h264_path = output_dir / "result_h264.mp4"
                    cmd = [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(job_video_path),
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(h264_path),
                    ]
                    subprocess.run(cmd, check=True)
                    # Replace artifact path to point to H.264 output
                    job_video_path = h264_path
        except Exception as _exc:  # noqa: BLE001
            # Best-effort re-encode; keep original if re-encode fails
            pass

        # Export CSV
        with trajectory_csv_path.open("w", encoding="utf-8") as csv_file:
            csv_file.write("frame,x,y,speed_mps\n")
            for frame_no in frame_numbers:
                x, y = smoothed_map.get(frame_no, (0.0, 0.0))
                csv_file.write(f"{frame_no},{x:.4f},{y:.4f},{speeds.get(frame_no, 0.0):.6f}\n")

        # Export JSON
        trajectory_payload = []
        for frame_no in frame_numbers:
            x, y = smoothed_map.get(frame_no, (0.0, 0.0))
            measurement = measurement_map.get(frame_no)
            bbox = None
            confidence = 0.0
            if measurement:
                confidence = measurement.detection_conf
                if measurement.bbox:
                    bbox = [float(measurement.bbox[0]), float(measurement.bbox[1]), float(measurement.bbox[2]), float(measurement.bbox[3])]

            trajectory_payload.append(
                {
                    "frame": frame_no,
                    "x": float(x),
                    "y": float(y),
                    "speed_mps": speeds.get(frame_no, 0.0),
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

        with trajectory_json_path.open("w", encoding="utf-8") as json_file:
            json.dump(trajectory_payload, json_file, indent=2)

        stats = JobStats(
            max_speed_mps=max_speed,
            avg_speed_mps=avg_speed,
            impact_frame=impact_frame,
            duration_s=duration_s,
        )

        stats_payload = {
            "max_speed_mps": max_speed,
            "avg_speed_mps": avg_speed,
            "impact_frame": impact_frame,
            "duration_s": duration_s,
            "frames": len(frame_numbers),
            "fps": fps,
            "width": width,
            "height": height,
        }
        with stats_json_path.open("w", encoding="utf-8") as stats_file:
            json.dump(stats_payload, stats_file, indent=2)

        artifacts = ArtifactPaths(
            video_mp4=job_video_path.name,
            trajectory_csv=trajectory_csv_path.name,
            trajectory_json=trajectory_json_path.name,
            snapshots=snapshots,
            stats_json=stats_json_path.name,
        )

        emit("completed", 0.95)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Analysis complete job_id=%s max_speed=%.2f avg_speed=%.2f impact_frame=%s duration=%.2fs",
                job_id,
                max_speed,
                avg_speed,
                impact_frame,
                duration_s,
            )

        return PipelineResult(artifacts=artifacts, stats=stats, device=device_name)

    def _initialise_detector(
        self,
        requested_device: str,
        emit,
    ) -> Tuple[OpenVINOYoloDetector, str]:
        candidates = self._device_candidates(requested_device)
        last_error: Optional[Exception] = None

        for candidate in candidates:
            try:
                detector = self._get_detector(candidate)
                if requested_device == "AUTO":
                    emit("detecting", 0.1, f"Using {candidate}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Detector selected requested=%s chosen=%s",
                        requested_device,
                        candidate,
                    )
                return detector, candidate
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        "Detector initialisation failed requested=%s candidate=%s error=%s",
                        requested_device,
                        candidate,
                        exc,
                    )

        device_list = ", ".join(candidates) or requested_device
        raise RuntimeError(f"Failed to initialise detector for devices: {device_list}") from last_error

    def _device_candidates(self, requested_device: str) -> List[str]:
        if requested_device == "AUTO":
            return ["GPU", "NPU", "CPU"]
        return [requested_device]

    def _get_detector(self, device: str) -> OpenVINOYoloDetector:
        key = device.upper()
        detector = self.detectors.get(key)
        if detector is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Initialising detector for device=%s", key)
            detector = OpenVINOYoloDetector(self.config, device=key)
            self.detectors[key] = detector
        return detector
