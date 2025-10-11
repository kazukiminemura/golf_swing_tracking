from __future__ import annotations

import json
import logging
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque

import cv2
import numpy as np
from openvino import Core
import yaml

from ..config import Config, PipelineConfig
from ..jobs import PipelineResult
from ..logging_utils import debug_trace, get_logger
from ..schemas import ArtifactPaths, JobStats

logger = get_logger("pipeline")


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
        self.iou_thres = config.model.iou_thres
        self.class_names = self._load_metadata_names(Path(xml_path))
        self.target_class_idx = self._resolve_target_class_index(self.class_names)

    @debug_trace(name="OpenVINOYoloDetector.detect")
    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float]]:
        """Returns list of (bbox_xyxy, confidence) detections for the target class."""
        blob, ratio, pad = self._preprocess(frame)
        predictions = self.compiled.infer_new_request({self.input_tensor.any_name: blob})
        detections: List[Tuple[Tuple[float, float, float, float], float]] = []

        if len(self.output_tensors) == 2:
            boxes = predictions[self.output_tensors[0]]
            scores = predictions[self.output_tensors[1]]
            boxes = self._reshape_boxes(np.squeeze(boxes))
            scores = self._reshape_scores(np.squeeze(scores), boxes.shape[0])
            detections = self._collect_detections(boxes, scores, ratio, pad, frame.shape[:2])
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
            preds = self._reshape_predictions(preds)
            boxes = preds[:, :4]
            scores = preds[:, 4:]
            detections = self._collect_detections(boxes, scores, ratio, pad, frame.shape[:2])

        return detections

    def _collect_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        ratio: float,
        pad: Tuple[float, float],
        frame_shape: Tuple[int, int],
    ) -> List[Tuple[Tuple[float, float, float, float], float]]:
        if boxes.size == 0 or scores.size == 0:
            return []

        boxes = boxes.astype(np.float32, copy=False)
        scores = self._sigmoid(scores.astype(np.float32, copy=False))

        if scores.ndim == 1:
            scores = scores[:, np.newaxis]

        if self.target_class_idx is not None and self.target_class_idx < scores.shape[1]:
            confs = scores[:, self.target_class_idx]
        else:
            confs = np.max(scores, axis=1)

        keep = confs >= self.conf_thres
        if not np.any(keep):
            return []

        detections: List[Tuple[Tuple[float, float, float, float], float]] = []
        for box, conf in zip(boxes[keep], confs[keep]):
            xyxy = self._scale_coords(box, ratio, pad, frame_shape)
            detections.append((xyxy, float(conf)))

        detections.sort(key=lambda item: item[1], reverse=True)
        return detections

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _reshape_boxes(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.reshape(0, 4)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] != 4:
            arr = arr.reshape(-1, 4)
        return arr

    @staticmethod
    def _reshape_scores(arr: np.ndarray, box_count: int) -> np.ndarray:
        if arr.size == 0:
            return arr.reshape(box_count, 0)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[0] != box_count:
            arr = arr.reshape(box_count, -1)
        return arr

    @staticmethod
    def _reshape_predictions(preds: np.ndarray) -> np.ndarray:
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]
        if preds.ndim == 2 and preds.shape[0] < preds.shape[1] and preds.shape[1] > 8:
            preds = preds.T
        return preds.reshape(-1, preds.shape[-1])

    @staticmethod
    def _load_metadata_names(xml_path: Path) -> Dict[int, str]:
        metadata_path = xml_path.with_name("metadata.yaml")
        if not metadata_path.exists():
            return {}
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                payload = yaml.safe_load(fh) or {}
        except Exception:
            return {}

        names = payload.get("names", {})
        mapping: Dict[int, str] = {}
        if isinstance(names, dict):
            for key, value in names.items():
                try:
                    mapping[int(key)] = str(value)
                except (TypeError, ValueError):
                    continue
        elif isinstance(names, (list, tuple)):
            mapping = {idx: str(name) for idx, name in enumerate(names)}
        return mapping

    @staticmethod
    def _resolve_target_class_index(names: Dict[int, str]) -> Optional[int]:
        for idx, name in names.items():
            if str(name).lower() == "head":
                return idx
        if names:
            return max(names.keys())
        return None

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
        x1 = y1 = x2 = y2 = 0.0
        if box.shape[0] >= 4 and box[2] > box[0] and box[3] > box[1]:
            # Already in x1, y1, x2, y2 form
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
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


class ClubHeadTracker:
    """Single-object tracker using smoothed detections with velocity-based prediction."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        window = max(1, getattr(cfg, "smoothing_window", 1))
        self.centers: deque[Tuple[float, float]] = deque(maxlen=window)
        self.sizes: deque[Tuple[float, float]] = deque(maxlen=window)
        self.reset()

    def reset(self) -> None:
        self.centers.clear()
        self.sizes.clear()
        self.velocity: Tuple[float, float] = (0.0, 0.0)
        self.last_center: Optional[Tuple[float, float]] = None
        self.last_size: Optional[Tuple[float, float]] = None
        self.last_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_frame_idx: Optional[int] = None
        self.last_area: Optional[float] = None
        self.confidence: float = 0.0
        self.misses: int = 0

    @staticmethod
    def _clip_bbox(bbox: Tuple[float, float, float, float], frame_shape: Tuple[int, int, int]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        x1 = float(np.clip(x1, 0, w - 1))
        y1 = float(np.clip(y1, 0, h - 1))
        x2 = float(np.clip(x2, 0, w - 1))
        y2 = float(np.clip(y2, 0, h - 1))
        if x2 <= x1:
            x2 = min(float(w - 1), x1 + 4.0)
        if y2 <= y1:
            y2 = min(float(h - 1), y1 + 4.0)
        return (x1, y1, x2, y2)

    @staticmethod
    def _mean_pair(values: deque[Tuple[float, float]]) -> Tuple[float, float]:
        if not values:
            return (0.0, 0.0)
        sx = sum(v[0] for v in values)
        sy = sum(v[1] for v in values)
        inv = 1.0 / len(values)
        return (sx * inv, sy * inv)

    def _update_with_detection(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: Optional[float],
        frame_idx: int,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        width = max(4.0, x2 - x1)
        height = max(4.0, y2 - y1)
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        size = (width, height)

        if self.last_center is not None and self.last_frame_idx is not None:
            dt = max(1, frame_idx - self.last_frame_idx)
            if dt > 0:
                vx = (center[0] - self.last_center[0]) / dt
                vy = (center[1] - self.last_center[1]) / dt
                self.velocity = (
                    0.7 * self.velocity[0] + 0.3 * vx,
                    0.7 * self.velocity[1] + 0.3 * vy,
                )

        self.centers.append(center)
        self.sizes.append(size)

        avg_center = self._mean_pair(self.centers)
        avg_size = self._mean_pair(self.sizes)

        smoothed_bbox = (
            avg_center[0] - avg_size[0] / 2.0,
            avg_center[1] - avg_size[1] / 2.0,
            avg_center[0] + avg_size[0] / 2.0,
            avg_center[1] + avg_size[1] / 2.0,
        )
        smoothed_bbox = self._clip_bbox(smoothed_bbox, frame_shape)

        self.last_center = (
            (smoothed_bbox[0] + smoothed_bbox[2]) / 2.0,
            (smoothed_bbox[1] + smoothed_bbox[3]) / 2.0,
        )
        self.last_size = (
            max(4.0, smoothed_bbox[2] - smoothed_bbox[0]),
            max(4.0, smoothed_bbox[3] - smoothed_bbox[1]),
        )
        self.last_bbox = smoothed_bbox
        self.last_area = self.last_size[0] * self.last_size[1]
        self.last_frame_idx = frame_idx
        self.confidence = float(confidence or 0.0)
        self.misses = 0
        return smoothed_bbox

    def _predict(self, frame_idx: int, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[float, float, float, float]]:
        if (
            self.last_bbox is None
            or self.last_center is None
            or self.last_size is None
            or self.last_frame_idx is None
        ):
            return None

        dt = frame_idx - self.last_frame_idx
        if dt <= 0:
            return self._clip_bbox(self.last_bbox, frame_shape)

        vx, vy = self.velocity
        decay = 0.92 ** dt
        vx *= decay
        vy *= decay

        pred_center = (
            self.last_center[0] + vx * dt,
            self.last_center[1] + vy * dt,
        )
        width, height = self.last_size
        predicted_bbox = (
            pred_center[0] - width / 2.0,
            pred_center[1] - height / 2.0,
            pred_center[0] + width / 2.0,
            pred_center[1] + height / 2.0,
        )
        predicted_bbox = self._clip_bbox(predicted_bbox, frame_shape)

        self.velocity = (vx, vy)
        self.last_center = (
            (predicted_bbox[0] + predicted_bbox[2]) / 2.0,
            (predicted_bbox[1] + predicted_bbox[3]) / 2.0,
        )
        self.last_size = (
            max(4.0, predicted_bbox[2] - predicted_bbox[0]),
            max(4.0, predicted_bbox[3] - predicted_bbox[1]),
        )
        self.last_bbox = predicted_bbox
        self.last_area = self.last_size[0] * self.last_size[1]
        self.last_frame_idx = frame_idx
        self.confidence *= 0.9
        return predicted_bbox

    def step(
        self,
        detection_bbox: Optional[Tuple[float, float, float, float]],
        confidence: Optional[float],
        frame_idx: int,
        frame_shape: Tuple[int, int, int],
    ) -> Optional[Tuple[Tuple[float, float, float, float], float, str]]:
        if detection_bbox is not None:
            bbox = self._update_with_detection(detection_bbox, confidence, frame_idx, frame_shape)
            return bbox, self.confidence, "detect"

        if self.last_bbox is None:
            return None

        bbox = self._predict(frame_idx, frame_shape)
        if bbox is None:
            self.reset()
            return None

        self.misses += 1
        if self.misses > max(1, getattr(self.cfg, "max_age", 1)):
            self.reset()
            return None

        return bbox, max(self.confidence, 0.0), "track"


class AnalysisPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.detectors: Dict[str, OpenVINOYoloDetector] = {}

    def _is_plausible_detection(
        self,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int, int],
        frame_idx: int,
        last_center: Optional[Tuple[float, float]],
        last_area: Optional[float],
        last_frame_idx: Optional[int],
    ) -> Tuple[bool, Optional[str]]:
        cfg = self.config.pipeline

        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width <= 0.0 or height <= 0.0:
            return False, "degenerate bbox dimensions"

        area = width * height
        if cfg.min_bbox_area_px > 0.0 and area < cfg.min_bbox_area_px:
            return False, f"area {area:.1f} below minimum {cfg.min_bbox_area_px}"

        frame_h, frame_w = frame_shape[:2]
        frame_area = float(frame_h * frame_w)
        if cfg.max_bbox_area_ratio > 0.0 and area > frame_area * cfg.max_bbox_area_ratio:
            return False, f"area ratio {area / frame_area:.4f} exceeds limit {cfg.max_bbox_area_ratio}"

        aspect_ratio = width / max(height, 1e-6)
        if cfg.min_bbox_aspect_ratio > 0.0 and aspect_ratio < cfg.min_bbox_aspect_ratio:
            return False, f"aspect {aspect_ratio:.2f} below minimum {cfg.min_bbox_aspect_ratio}"
        if cfg.max_bbox_aspect_ratio > 0.0 and aspect_ratio > cfg.max_bbox_aspect_ratio:
            return False, f"aspect {aspect_ratio:.2f} above maximum {cfg.max_bbox_aspect_ratio}"

        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        if last_center is not None and last_frame_idx is not None:
            frame_gap = frame_idx - last_frame_idx
            if frame_gap <= max(1, cfg.max_age):
                if cfg.max_center_jump_px > 0.0:
                    dist = math.hypot(center[0] - last_center[0], center[1] - last_center[1])
                    if dist > cfg.max_center_jump_px:
                        return False, f"center jump {dist:.1f}px exceeds limit {cfg.max_center_jump_px}"
                if last_area and cfg.max_area_change_ratio > 0.0:
                    ratio = max(area, last_area) / max(min(area, last_area), 1e-6)
                    if ratio > cfg.max_area_change_ratio:
                        return False, f"area change {ratio:.2f} exceeds limit {cfg.max_area_change_ratio}"

        return True, None

    def _initial_detection_score(
        self,
        bbox: Tuple[float, float, float, float],
        frame_shape: Tuple[int, int, int],
        confidence: float,
    ) -> float:
        frame_h, frame_w = frame_shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return confidence

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        mid_x = frame_w / 2.0
        mid_y = frame_h / 2.0
        if center_x >= mid_x and center_y >= mid_y:
            denom_x = max(frame_w - mid_x, 1.0)
            denom_y = max(frame_h - mid_y, 1.0)
            rightness = max(0.0, min(1.0, (center_x - mid_x) / denom_x))
            bottomness = max(0.0, min(1.0, (center_y - mid_y) / denom_y))
            location_score = 0.5 * (rightness + bottomness)
        else:
            location_score = -1.0
        return location_score + 0.05 * confidence

    def _smooth_swing_path(
        self,
        payload: List[Dict[str, float]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        if len(payload) < 2:
            return

        frames = np.array([entry["frame"] for entry in payload], dtype=np.float64)
        xs = np.array([entry["x"] for entry in payload], dtype=np.float64)
        ys = np.array([entry["y"] for entry in payload], dtype=np.float64)

        t = frames - frames[0]
        duration = float(np.max(t))
        if duration > 0.0:
            t /= duration
        else:
            t = np.zeros_like(t)

        degree_options = [1, 2]
        best_fit = None
        best_error = float("inf")

        rank_warning = getattr(np, "RankWarning", RuntimeWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rank_warning)
            for degree in degree_options:
                if len(payload) <= degree:
                    continue
                try:
                    coeffs_x = np.polyfit(t, xs, degree)
                    coeffs_y = np.polyfit(t, ys, degree)
                except Exception:  # noqa: BLE001
                    continue
                fit_x = np.polyval(coeffs_x, t)
                fit_y = np.polyval(coeffs_y, t)
                residual = np.mean((fit_x - xs) ** 2 + (fit_y - ys) ** 2)
                penalty = 1e-3 * degree
                score = residual + penalty
                if score < best_error:
                    best_error = score
                    best_fit = (fit_x, fit_y, degree)

        if best_fit is None:
            return

        fit_x, fit_y, selected_degree = best_fit
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Swing path smoothing applied degree=%s residual=%.5f",
                selected_degree,
                best_error,
            )

        max_x = max(float(frame_width) - 1.0, 0.0)
        max_y = max(float(frame_height) - 1.0, 0.0)

        for entry, new_x, new_y in zip(payload, fit_x, fit_y):
            bbox = entry.get("bbox", [entry["x"], entry["y"], entry["x"], entry["y"]])
            orig_cx = float(entry["x"])
            orig_cy = float(entry["y"])
            x1, y1, x2, y2 = bbox
            width = max(2.0, x2 - x1)
            height = max(2.0, y2 - y1)
            half_w = width / 2.0
            half_h = height / 2.0

            cx = float(np.clip(new_x, 0.0, max_x))
            cy = float(np.clip(new_y, 0.0, max_y))

            shift_limit_x = max(10.0, 0.3 * width)
            shift_limit_y = max(10.0, 0.3 * height)
            delta_x = max(-shift_limit_x, min(shift_limit_x, cx - orig_cx))
            delta_y = max(-shift_limit_y, min(shift_limit_y, cy - orig_cy))
            cx = float(np.clip(orig_cx + delta_x, 0.0, max_x))
            cy = float(np.clip(orig_cy + delta_y, 0.0, max_y))

            left = cx - half_w
            right = cx + half_w
            if left < 0.0:
                right -= left
                left = 0.0
            if right > max_x:
                shift = right - max_x
                left -= shift
                right = max_x
                if left < 0.0:
                    left = 0.0
            top = cy - half_h
            bottom = cy + half_h
            if top < 0.0:
                bottom -= top
                top = 0.0
            if bottom > max_y:
                shift = bottom - max_y
                top -= shift
                bottom = max_y
                if top < 0.0:
                    top = 0.0

            entry["bbox"] = [float(left), float(top), float(right), float(bottom)]
            entry["x"] = float((left + right) / 2.0)
            entry["y"] = float((top + bottom) / 2.0)

    @debug_trace(name="AnalysisPipeline.run")
    def run(
        self,
        job_id: str,
        input_video: Path,
        output_dir: Path,
        device: Optional[str],
        progress_cb,
        params: Optional[Dict[str, object]] = None,
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

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open input video: {input_video}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        emit("detecting", 0.1)

        detector, device_name = self._initialise_detector(requested_device, emit)

        job_video_path = output_dir / "result.mp4"
        trajectory_json_path = output_dir / "trajectory.json"
        stats_json_path = output_dir / "stats.json"

        # Prefer H.264 for web playback; try common FourCCs then fall back.
        chosen_fourcc = None
        writer = None
        writer_path = job_video_path
        codec_groups = [
            (Path("result.mp4"), ["avc1", "H264", "X264", "h264", "mp4v"]),
            (Path("result.avi"), ["MP4V", "MJPG", "XVID"]),
        ]
        for target_name, codes in codec_groups:
            candidate_path = output_dir / target_name
            for cc in codes:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*cc)
                    candidate = cv2.VideoWriter(str(candidate_path), fourcc, fps, (width, height))
                    if candidate.isOpened():
                        writer = candidate
                        chosen_fourcc = cc
                        writer_path = candidate_path
                        break
                    candidate.release()
                except Exception:  # noqa: BLE001
                    continue
            if writer is not None:
                break

        if writer is None:
            cap.release()
            raise RuntimeError("Failed to initialise VideoWriter for output video.")
        job_video_path = writer_path

        # Resolve tracking mode: per-job override via params takes precedence over config
        use_tracking = bool(getattr(self.config.pipeline, "enable_tracking", True))
        try:
            if params and "enable_tracking" in params:
                use_tracking = bool(params.get("enable_tracking"))
        except Exception:
            pass
        tracker = ClubHeadTracker(self.config.pipeline) if use_tracking else None
        detection_payload: List[Dict[str, float]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not use_tracking:
                detections = detector.detect(frame)
                detection_bbox: Optional[Tuple[float, float, float, float]] = None
                detection_confidence: Optional[float] = None

                if detections:
                    ordered_detections = sorted(
                        detections,
                        key=lambda det: self._initial_detection_score(det[0], frame.shape, det[1]),
                        reverse=True,
                    )

                    for candidate_bbox, candidate_conf in ordered_detections:
                        plausible, reason = self._is_plausible_detection(
                            candidate_bbox,
                            frame.shape,
                            frame_idx,
                            None,
                            None,
                            None,
                        )
                        if plausible:
                            detection_bbox = candidate_bbox
                            detection_confidence = candidate_conf
                            break
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "Frame %s detection rejected: %s (bbox=%s conf=%.3f)",
                                frame_idx,
                                reason,
                                candidate_bbox,
                                candidate_conf,
                            )

                if detection_bbox is not None and detection_confidence is not None:
                    bbox, confidence, source = detection_bbox, float(detection_confidence), "detect"
                    x1, y1, x2, y2 = bbox
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Frame %s club-head %s bbox=(%.2f, %.2f, %.2f, %.2f) conf=%.3f",
                            frame_idx,
                            "detected",
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence,
                        )
                    x1, y1, x2, y2 = bbox
                    detection_payload.append(
                        {
                            "frame": frame_idx,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(confidence),
                            "x": float((x1 + x2) / 2.0),
                            "y": float((y1 + y2) / 2.0),
                            "source": source,
                        }
                    )
                    frame_h, frame_w = frame.shape[:2]
                    x1_i = max(0, min(frame_w - 1, int(round(x1))))
                    y1_i = max(0, min(frame_h - 1, int(round(y1))))
                    x2_i = max(0, min(frame_w - 1, int(round(x2))))
                    y2_i = max(0, min(frame_h - 1, int(round(y2))))
                    color = (255, 0, 0)
                    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                    label = f"D {confidence:.2f}"
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

                    cv2.circle(
                        frame,
                        (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
                        6,
                        (0, 0, 255),
                        -1,
                    )
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Frame %s club-head detection: none", frame_idx)
            else:
                # Use tracker-assisted path
                detections = detector.detect(frame)
                detection_bbox: Optional[Tuple[float, float, float, float]] = None
                detection_confidence: Optional[float] = None

                if detections:
                    if tracker and tracker.last_center is None:
                        ordered_detections = sorted(
                            detections,
                            key=lambda det: self._initial_detection_score(det[0], frame.shape, det[1]),
                            reverse=True,
                        )
                    else:
                        ordered_detections = detections

                    for candidate_bbox, candidate_conf in ordered_detections:
                        plausible, reason = self._is_plausible_detection(
                            candidate_bbox,
                            frame.shape,
                            frame_idx,
                            tracker.last_center if tracker else None,
                            tracker.last_area if tracker else None,
                            tracker.last_frame_idx if tracker else None,
                        )
                        if plausible:
                            detection_bbox = candidate_bbox
                            detection_confidence = candidate_conf
                            break
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "Frame %s detection rejected: %s (bbox=%s conf=%.3f)",
                                frame_idx,
                                reason,
                                candidate_bbox,
                                candidate_conf,
                            )

                track_result = tracker.step(detection_bbox, detection_confidence, frame_idx, frame.shape) if tracker else None

                if track_result:
                    bbox, confidence, source = track_result
                    x1, y1, x2, y2 = bbox
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Frame %s club-head %s bbox=(%.2f, %.2f, %.2f, %.2f) conf=%.3f",
                            frame_idx,
                            "tracked" if source == "track" else "detected",
                            x1,
                            y1,
                            x2,
                            y2,
                            confidence,
                        )
                    x1, y1, x2, y2 = bbox
                    detection_payload.append(
                        {
                            "frame": frame_idx,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(confidence),
                            "x": float((x1 + x2) / 2.0),
                            "y": float((y1 + y2) / 2.0),
                            "source": source,
                        }
                    )
                    frame_h, frame_w = frame.shape[:2]
                    x1_i = max(0, min(frame_w - 1, int(round(x1))))
                    y1_i = max(0, min(frame_h - 1, int(round(y1))))
                    x2_i = max(0, min(frame_w - 1, int(round(x2))))
                    y2_i = max(0, min(frame_h - 1, int(round(y2))))
                    color = (255, 0, 0) if source == "detect" else (0, 128, 255)
                    cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                    label = f"{'T' if source == 'track' else 'D'} {confidence:.2f}"
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

                    cv2.circle(
                        frame,
                        (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
                        6,
                        (0, 0, 255),
                        -1,
                    )
                elif logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Frame %s club-head detection: none", frame_idx)

            writer.write(frame)

            frame_idx += 1
            if total_frames:
                emit("detecting", min(0.1 + 0.75 * (frame_idx / total_frames), 0.85))

        cap.release()
        writer.release()

        if not detection_payload:
            raise RuntimeError("No club head detections found in the provided video.")

        self._smooth_swing_path(detection_payload, width, height)

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
                    job_video_path = h264_path
        except Exception as _exc:  # noqa: BLE001
            pass

        emit("exporting", 0.9)

        with trajectory_json_path.open("w", encoding="utf-8") as json_file:
            json.dump(detection_payload, json_file, indent=2)

        stats_payload = {
            "detections": len(detection_payload),
            "frames": frame_idx,
            "fps": fps,
            "width": width,
            "height": height,
        }
        with stats_json_path.open("w", encoding="utf-8") as stats_file:
            json.dump(stats_payload, stats_file, indent=2)

        stats = JobStats()

        artifacts = ArtifactPaths(
            video_mp4=job_video_path.name,
            trajectory_csv=None,
            trajectory_json=trajectory_json_path.name,
            snapshots=[],
            stats_json=stats_json_path.name,
        )

        emit("completed", 0.95)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Analysis complete job_id=%s detections=%s fps=%.2f",
                job_id,
                len(detection_payload),
                fps,
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
