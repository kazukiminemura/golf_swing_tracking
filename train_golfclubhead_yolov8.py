# train_golfclubhead_yolov8_cpu.py
import argparse
from pathlib import Path
import sys, platform, importlib, traceback
from ultralytics import YOLO
import yaml

# --- 参考情報の標準出力（任意） ---
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
  0: club
  1: head
"""

def main():
    ap = argparse.ArgumentParser(description="YOLOv8 fine-tune (club head, CPU only)")
    ap.add_argument("--data_root", required=True, help="root containing {train,valid}/{images,labels}")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=-1)   # -1:auto
    ap.add_argument("--project", default="runs/club_head")
    ap.add_argument("--name", default="yolov8_finetune_cpu")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--use_existing_yaml", action="store_true", help="Use and rewrite an existing data.yaml in-place")
    ap.add_argument("--export_ov", action="store_true")
    ap.add_argument("--export_onnx", action="store_true")
    args = ap.parse_args()

    # --- データルート解決 ---
    root = Path(args.data_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"[ERR] data_root does not exist: {root}")

    # 必須フォルダ確認
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

    # --- 学習（CPU固定） ---
    model = YOLO(args.model)
    print("[INFO] start training on CPU...")
    train_res = model.train(
        data=str(yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device="cpu",                 # ★ CPU固定
        project=args.project,
        name=args.name,
        resume=args.resume,
        # 便利系（必要に応じて有効化）
        # workers=0,   # Windowsでの安全策
        # cache=True,
    )
    print(f"[INFO] weights saved in: {train_res.save_dir}")

    # --- 検証（CPU固定） ---
    print("[INFO] validating on CPU...")
    val_res = model.val(data=str(yaml_path), imgsz=args.imgsz, device="cpu")
    try:
        print(f"[VAL] mAP50={val_res.box.map50:.4f}  mAP50-95={val_res.box.map:.4f}")
    except Exception:
        print("[VAL] finished. (metrics structure may vary by version)")

    # --- エクスポート（任意、CPU固定） ---
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

    # --- 予測例 ---
    print("\n[HINT] predict example:")
    print(f'yolo task=detect mode=predict model="{best_pt}" source=/path/to/test.mp4 imgsz={args.imgsz} conf=0.25 device=cpu')

if __name__ == "__main__":
    main()
