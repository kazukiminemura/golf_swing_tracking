# train_golfclubhead_yolov11_cpu.py
import argparse
from pathlib import Path
import sys, platform, importlib, traceback
from ultralytics import YOLO
import yaml

# --- 参老E��報の標準�E力（任意！E---
print('[ENV] python:', sys.version.replace('\n', ' '))
print('[ENV] executable:', sys.executable)
print('[ENV] platform:', platform.platform())
for pkg in ('torch', 'ultralytics'):
    try:
        m = importlib.import_module(pkg)
        print(f"[ENV] {pkg} version: {getattr(m, '__version__', 'unknown')}")
    except Exception as e:
        print(f"[ENV] {pkg} import error: {e}")
        traceback.print_exc()

YAML_TPL = """\
# Auto-generated for golf club head detection
path: {root}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  1: club
  0: head
"""

def main():
    ap = argparse.ArgumentParser(description="YOLOv11 fine-tune (club head, GPU-first)")
    ap.add_argument("--data_root", required=True, help="root containing {train,valid}/{images,labels}")
    ap.add_argument("--model", default="yolov11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=200)
    # GPU-first defaults
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="Ultralytics device: 'auto'|'cpu'|'0'|'0,1'|'cuda:0'")
    ap.add_argument("--project", default="runs/club_head")
    ap.add_argument("--name", default="yolov11_finetune_cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use_existing_yaml", action="store_true", help="Use and rewrite an existing data.yaml in-place")
    ap.add_argument("--export_ov", action="store_true")
    ap.add_argument("--export_onnx", action="store_true")
    args = ap.parse_args()

    # --- チE�Eタルート解決 ---
    root = Path(args.data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"[ERR] data_root does not exist: {root}")

    # 忁E��フォルダ確誁E
    for p in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        if not (root / p).exists():
            raise AssertionError(f"[ERR] missing: {root / p}")

    # --- dataset YAML 準備 ---
    if args.use_existing_yaml and (root / "data.yaml").exists():
        yaml_path = root / "data.yaml"
        with open(yaml_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f) or {}
        data_cfg.update({
            "path": str(root).replace("\\", "/"),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
        })
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_cfg, f, sort_keys=False)
        print(f"[INFO] using & updated existing -> {yaml_path}")
    else:
        yaml_path = root / "club_head.yaml"
        yaml_path.write_text(YAML_TPL.format(root=str(root).replace("\\", "/")), encoding="utf-8")
        print(f"[INFO] generated -> {yaml_path}")

    # --- デバイス決定 ---
    device_arg = args.device
    if device_arg == "auto":
        try:
            import torch  # noqa: F401
            device_arg = "0" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
        except Exception:
            device_arg = "cpu"
    else:
        # If a GPU was requested but CUDA is unavailable, fall back gracefully.
        if device_arg != "cpu":
            try:
                import torch  # noqa: F401
                if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
                    print("[WARN] CUDA is not available. Falling back to CPU.")
                    device_arg = "cpu"
            except Exception:
                device_arg = "cpu"
    print(f"[INFO] using device: {device_arg}")

    # --- 学習（指定デバイス） ---
    model = YOLO(args.model)
    print(f"[INFO] start training on {device_arg}...")
    train_res = model.train(
        data=str(yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device_arg,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # workers=0,   # Windowsでの安定
        # cache=True,
    )
    print(f"[INFO] weights saved in: {train_res.save_dir}")

    # --- 検証（指定デバイス） ---
    print(f"[INFO] validating on {device_arg}...")
    val_res = model.val(data=str(yaml_path), imgsz=args.imgsz, device=device_arg)
    try:
        print(f"[VAL] mAP50={val_res.box.map50:.4f}  mAP50-95={val_res.box.map:.4f}")
    except Exception:
        print("[VAL] finished. (metrics structure may vary by version)")

    # --- エクスポ�Eト（任意、CPU固定！E---
    best_pt = Path(train_res.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        if args.export_ov:
            print("[INFO] export OpenVINO IR (CPU)...")
            model.export(model=str(best_pt), format="openvino", imgsz=args.imgsz, device="cpu")
        if args.export_onnx:
            print("[INFO] export ONNX (CPU)...")
            model.export(model=str(best_pt), format="onnx", imgsz=args.imgsz, opset=12, device="cpu")
    else:
        print(f"[WARN] best.pt not found: {best_pt}")

    # --- 予測侁E---
    print("\n[HINT] predict example:")
    print(f'yolo task=detect mode=predict model="{best_pt}" source=/path/to/test.mp4 imgsz={args.imgsz} conf=0.25 device={device_arg}')

if __name__ == "__main__":
    main()

