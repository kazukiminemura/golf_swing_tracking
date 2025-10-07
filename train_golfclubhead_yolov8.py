# train_golfclubhead_yolov8_min.py
import argparse
from pathlib import Path
from ultralytics import YOLO

YAML_TPL = """\
# Auto-generated for golf club head detection
path: {root}
train: club_head/train
val:   club_head/val
names:
  0: club_head
"""

def main():
    ap = argparse.ArgumentParser(description="YOLOv8 fine-tune (club head)")
    ap.add_argument("--data_root", required=True, help="root containing club_head/{train,val}/{images,labels}")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=-1)   # -1:auto
    ap.add_argument("--device", default="")            # "0" / "cpu"
    ap.add_argument("--project", default="runs/club_head")
    ap.add_argument("--name", default="yolov8_finetune")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export_ov", action="store_true")
    ap.add_argument("--export_onnx", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    # 必要フォルダ確認
    for p in ["club_head/train/images", "club_head/train/labels", "club_head/val/images", "club_head/val/labels"]:
        assert (root / p).exists(), f"[ERR] missing: {root/p}"

    # YAML自動生成
    yaml_path = root / "club_head.yaml"
    yaml_path.write_text(YAML_TPL.format(root=str(root).replace("\\", "/")), encoding="utf-8")
    print(f"[INFO] dataset yaml -> {yaml_path}")

    # 学習
    model = YOLO(args.model)
    print("[INFO] training...")
    results = model.train(
        data=str(yaml_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=(args.device or None),
        project=args.project,
        name=args.name,
        mosaic=0.5, mixup=0.10, close_mosaic=10,  # 小物体・高速運動向けの控えめ拡張
        box=7.0, cls=1.0, dfl=1.0,               # 位置精度寄り
        patience=50, cos_lr=True,                 # 収束安定
        seed=42, workers=8,
        exist_ok=True, verbose=True,
        resume=args.resume)

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[INFO] best weights: {best}")

    # 検証
    print("[INFO] validating...")
    val = model.val(model=str(best), data=str(yaml_path), imgsz=args.imgsz)
    print(f"[VAL] mAP50={val.box.map50:.4f}  mAP50-95={val.box.map:.4f}")

    # エクスポート（任意）
    if args.export_ov:
        print("[INFO] export OpenVINO IR...")
        model.export(model=str(best), format="openvino", imgsz=args.imgsz)
    if args.export_onnx:
        print("[INFO] export ONNX...")
        model.export(model=str(best), format="onnx", imgsz=args.imgsz, opset=12)

    # ワンライナー推論ヒント
    print("\n[HINT] predict:")
    print(f'yolo task=detect mode=predict model="{best}" source=/path/to/test.mp4 imgsz={args.imgsz} conf=0.25')

if __name__ == "__main__":
    main()
