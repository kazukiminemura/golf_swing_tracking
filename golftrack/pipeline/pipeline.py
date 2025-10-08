from __future__ import annotations

import json
import logging
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
        for cc in ["avc1", "H264", "X264", "h264", "mp4v"]:
            try:
                fourcc = cv2.VideoWriter_fourcc(*cc)
                candidate = cv2.VideoWriter(str(job_video_path), fourcc, fps, (width, height))
                if candidate.isOpened():
                    writer = candidate
                    chosen_fourcc = cc
                    break
                candidate.release()
            except Exception:  # noqa: BLE001
                pass
        if writer is None:
            cap.release()
            raise RuntimeError("Failed to initialise VideoWriter for output video.")

        detection_payload: List[Dict[str, float]] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            best_detection = detections[0] if detections else None
            if best_detection:
                bbox, confidence = best_detection
                x1, y1, x2, y2 = bbox
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Frame %s club-head detection bbox=(%.2f, %.2f, %.2f, %.2f) conf=%.3f",
                        frame_idx,
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                    )
                detection_payload.append(
                    {
                        "frame": frame_idx,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidence),
                        "x": float((x1 + x2) / 2.0),
                        "y": float((y1 + y2) / 2.0),
                    }
                )

                frame_h, frame_w = frame.shape[:2]
                x1_i = max(0, min(frame_w - 1, int(round(x1))))
                y1_i = max(0, min(frame_h - 1, int(round(y1))))
                x2_i = max(0, min(frame_w - 1, int(round(x2))))
                y2_i = max(0, min(frame_h - 1, int(round(y2))))
                cv2.rectangle(frame, (x1_i, y1_i), (x2_i, y2_i), (255, 0, 0), 2)
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
