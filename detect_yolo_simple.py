"""
Super simple YOLO detection that draws boxes for the head class only.

Usage (.pt or .xml):

- Webcam:   python detect_yolo_simple.py --source 0
- Video:    python detect_yolo_simple.py --source videos/side_reference1.mp4 --save out.mp4
- Image:    python detect_yolo_simple.py --source static/example.jpg --save out.jpg

Notes:
- Defaults to `best.pt` in this repo if `--model` is not given.
- If `--model` is an OpenVINO `.xml`, it runs on CPU automatically.
- Shows a window by default (use `--no-show` to disable). If `--save` is provided, writes the result.
"""

import argparse
from pathlib import Path
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


def _filtered_boxes(res, allow_classes: Optional[list]) -> np.ndarray:
    if res is None or not hasattr(res, "boxes") or res.boxes is None:
        return np.empty((0, 4))
    boxes = res.boxes.xyxy.cpu().numpy()
    if allow_classes is None:
        return boxes
    try:
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        mask = np.isin(cls_ids, np.array(allow_classes, dtype=int))
        return boxes[mask]
    except Exception:
        return boxes


def run_image(model: YOLO, source: str, imgsz: int, device: str, conf: float, show: bool, save: Optional[str], classes: Optional[list]):
    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {source}")
    results = model.predict(source=img, imgsz=imgsz, conf=conf, device=device, classes=classes, verbose=False)
    if not results:
        print("[WARN] No results returned by model.")
        return
    res = results[0]
    boxes = _filtered_boxes(res, classes)
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
            boxes = _filtered_boxes(res, classes)
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
    ap = argparse.ArgumentParser(description="YOLO detection (head only): draw bounding boxes")
    ap.add_argument("--model", default="best.pt", help="Path to YOLO weights (.pt or OpenVINO .xml)")
    ap.add_argument("--source", default="0", help="Image/Video path or webcam index (e.g., 0)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="auto", help="'auto'|'cpu'|'0'|'0,1'|'cuda:0'")
    ap.add_argument("--show", dest="show", action="store_true", help="Show window with detections")
    ap.add_argument("--no-show", dest="show", action="store_false", help="Disable window display")
    ap.set_defaults(show=True)
    ap.add_argument("--save", default=None, help="Output file (video/image) or directory for images")
    args = ap.parse_args()

    model_path = Path(args.model)
    is_openvino_model = model_path.suffix.lower() == ".xml"
    device = pick_device("cpu" if (args.device == "auto" and is_openvino_model) else args.device)

    model = YOLO(args.model)

    # Determine the index for the 'head' class
    names = getattr(model, "names", None)
    if names is None:
        names = getattr(getattr(model, "model", None), "names", None)

    class_filter = [0]

    if is_video_source(args.source):
        run_video(model, args.source, args.imgsz, device, args.conf, args.show, args.save, class_filter)
    else:
        run_image(model, args.source, args.imgsz, device, args.conf, args.show, args.save, class_filter)


if __name__ == "__main__":
    main()
