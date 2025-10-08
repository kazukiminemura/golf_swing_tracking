"""
Minimal YOLOv8 object detection script that draws only bounding boxes.

Usage examples (PyTorch .pt):

- Webcam:   python detect_yolov8_simple.py --model yolov8n.pt --source 0
- Video:    python detect_yolov8_simple.py --model runs/club_head/yolov8_finetune_cpu/weights/best.pt --source side_reference1.mp4
- Image:    python detect_yolov8_simple.py --model yolov8n.pt --source static/example.jpg --save out.jpg

Usage examples (OpenVINO .xml exported by Ultralytics):

- Webcam:   python detect_yolov8_simple.py --model runs/club_head/yolov8_finetune_cpu/weights/best_openvino_model/best.xml --source 0
- Video:    python detect_yolov8_simple.py --model path/to/best.xml --source side_reference1.mp4 --save out.mp4
- Image:    python detect_yolov8_simple.py --model path/to/best.xml --source static/example.jpg --save out.jpg

Notes:
- When you pass a .xml model, Ultralytics uses the OpenVINO backend automatically.
- Shows a window by default when a GUI is available (use --no-show to disable).
- Saves output when --save is provided (image path or video file). If --save is a directory for images,
  the script writes a file alongside the input image name.
 - Class filtering: by default this script detects only the "head" class. Override with `--classes`.
"""

import argparse
from pathlib import Path
import sys
import platform
import importlib
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # noqa: F401
        return "0" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def draw_boxes_only(frame: np.ndarray, boxes_xyxy: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    for x1, y1, x2, y2 in boxes_xyxy.astype(int):
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def is_video_source(path_or_num: str) -> bool:
    # Numeric string (webcam index) or common video file extensions
    if path_or_num.isdigit():
        return True
    ext = Path(path_or_num).suffix.lower()
    return ext in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def run_image(model: YOLO, source: str, imgsz: int, device: str, conf: float, show: bool, save: Optional[str], classes: Optional[list]):
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {source}")
    results = model.predict(source=img, imgsz=imgsz, conf=conf, device=device, classes=classes, verbose=False)
    if not results:
        print("[WARN] No results returned by model.")
        return
    res = results[0]
    boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res, "boxes") and res.boxes is not None else np.empty((0, 4))
    out = draw_boxes_only(img.copy(), boxes)

    if save:
        out_path = Path(save)
        if out_path.is_dir():
            out_path = out_path / (Path(source).stem + "_det" + Path(source).suffix)
        cv2.imwrite(str(out_path), out)
        print(f"[SAVE] {out_path}")
    if show:
        cv2.imshow("Detections (boxes only)", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(model: YOLO, source: Union[int, str], imgsz: int, device: str, conf: float, show: bool, save: Optional[str], classes: Optional[list]):
    cap_index: Optional[int] = None
    if isinstance(source, str) and source.isdigit():
        cap_index = int(source)
    cap = cv2.VideoCapture(cap_index if cap_index is not None else source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video/webcam source: {source}")

    writer = None
    if save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") if str(save).lower().endswith(".mp4") else cv2.VideoWriter_fourcc(*"XVID")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(save), fourcc, fps, (w, h))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results = model.predict(source=frame, imgsz=imgsz, conf=conf, device=device, classes=classes, verbose=False)
            res = results[0] if results else None
            boxes = res.boxes.xyxy.cpu().numpy() if res is not None and hasattr(res, "boxes") and res.boxes is not None else np.empty((0, 4))
            out = draw_boxes_only(frame.copy(), boxes)

            if writer is not None:
                writer.write(out)
            if show:
                cv2.imshow("Detections (boxes only)", out)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Simple YOLOv8 detection: draw bounding boxes only")
    ap.add_argument("--model", default="yolov8n.pt", help="Path to YOLOv8 .pt weights")
    ap.add_argument("--source", default="0", help="Image/Video path or webcam index (e.g., 0)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto", help="'auto'|'cpu'|'0'|'0,1'|'cuda:0'")
    ap.add_argument("--show", dest="show", action="store_true", help="Show window with detections")
    ap.add_argument("--no-show", dest="show", action="store_false", help="Disable window display")
    ap.set_defaults(show=True)
    ap.add_argument("--save", default=None, help="Output file (video/image) or directory for images")
    ap.add_argument(
        "--classes",
        default="head",
        help="Comma-separated class names or indices to detect (default: 'head'). Use 'all' to disable filtering.")
    args = ap.parse_args()

    print('[ENV] python:', sys.version.replace('\n', ' '))
    print('[ENV] executable:', sys.executable)
    print('[ENV] platform:', platform.platform())
    for pkg in ('torch', 'ultralytics', 'opencv-python', 'openvino'):
        try:
            m = importlib.import_module(
                'cv2' if pkg == 'opencv-python' else ('openvino.runtime' if pkg == 'openvino' else pkg)
            )
            print(f"[ENV] {pkg} version: {getattr(m, '__version__', 'unknown')}")
        except Exception as e:
            print(f"[ENV] {pkg} import error: {e}")

    # Prefer CPU when an OpenVINO model (.xml) is provided unless user overrides.
    model_path = Path(args.model)
    is_openvino_model = model_path.suffix.lower() == '.xml'
    device = pick_device('cpu' if (args.device == 'auto' and is_openvino_model) else args.device)
    if is_openvino_model and args.device == 'auto':
        print("[INFO] .xml model detected -> using OpenVINO backend on CPU")
    print(f"[INFO] using device: {device}")

    model = YOLO(args.model)

    # Resolve class filter to indices
    def get_model_names(m: YOLO) -> Optional[dict]:
        names = getattr(m, 'names', None)
        if names is None:
            names = getattr(getattr(m, 'model', None), 'names', None)
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: n for i, n in enumerate(names)}
        return None

    def parse_classes_arg(arg: str, names_map: Optional[dict]) -> Optional[list]:
        if not arg or arg.lower() in {"all", "none", "*"}:
            return None
        tokens = [t.strip() for t in arg.split(',') if t.strip()]
        out = []
        for t in tokens:
            if t.isdigit():
                out.append(int(t))
                continue
            if names_map:
                # case-insensitive exact match first
                matched = [i for i, n in names_map.items() if str(n).lower() == t.lower()]
                if not matched:
                    # fallback: contains
                    matched = [i for i, n in names_map.items() if t.lower() in str(n).lower()]
                if matched:
                    out.extend(matched)
                    continue
            # final fallback for common golf classes
            if t.lower() == 'head':
                out.append(1)
            elif t.lower() == 'club':
                out.append(0)
        # de-duplicate while preserving order
        seen = set()
        uniq = []
        for i in out:
            if i not in seen:
                uniq.append(i); seen.add(i)
        return uniq or None

    names_map = get_model_names(model)
    class_filter = parse_classes_arg(args.classes, names_map)
    if class_filter is not None:
        print(f"[INFO] class filter: {class_filter} ({[names_map.get(i, i) if names_map else i for i in class_filter]})")
    else:
        print("[INFO] class filter: disabled (all classes)")

    if is_video_source(args.source):
        run_video(model, args.source, args.imgsz, device, args.conf, args.show, args.save, class_filter)
    else:
        run_image(model, args.source, args.imgsz, device, args.conf, args.show, args.save, class_filter)


if __name__ == "__main__":
    main()
