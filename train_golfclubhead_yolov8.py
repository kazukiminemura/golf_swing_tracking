# train_golfclubhead_yolov8_min.py
import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml

YAML_TPL = """\
# Auto-generated for golf club head detection
path: {root}
train: club_head/train/images
val: club_head/valid/images
test: club_head/test/images

nc: 2
names:
  0: club
  1: head
"""

def main():
    ap = argparse.ArgumentParser(description="YOLOv8 fine-tune (club head)")
    ap.add_argument("--data_root", required=True, help="root containing club_head/{train,valid}/{images,labels}")
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
    ap.add_argument("--use_existing_yaml", action="store_true", help="Use existing data.yaml file")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    # 必要フォルダ確認
    for p in ["club_head/train/images", "club_head/train/labels", "club_head/valid/images", "club_head/valid/labels"]:
        assert (root / p).exists(), f"[ERR] missing: {root/p}"

    # データセット設定ファイルの選択
    if args.use_existing_yaml and (root / "club_head" / "data.yaml").exists():
        yaml_path = root / "club_head" / "data.yaml"
        print(f"[INFO] using existing dataset yaml -> {yaml_path}")
        
        # 既存のYAMLファイルのパスを修正
        import yaml
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # 相対パスを絶対パスに変更
        club_head_dir = root / "club_head"
        data_config['path'] = str(club_head_dir)
        data_config['train'] = 'train/images'
        data_config['val'] = 'valid/images'
        data_config['test'] = 'test/images'
        
        # 修正されたYAMLファイルを保存
        yaml_path = root / "club_head_modified.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        print(f"[INFO] modified dataset yaml -> {yaml_path}")
    else:
        # YAML自動生成
        yaml_path = root / "club_head.yaml"
        yaml_path.write_text(YAML_TPL.format(root=str(root).replace("\\", "/")), encoding="utf-8")
        print(f"[INFO] generated dataset yaml -> {yaml_path}")

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
        mosaic=0.8, mixup=0.15, close_mosaic=20,  # 2クラス検出向けの強めの拡張
        box=7.5, cls=0.5, dfl=1.5,               # multi-class向けの調整
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
    print("\n[CLASSES] detected:")
    print("  0: club (ゴルフクラブのシャフト部分)")
    print("  1: head (ゴルフクラブのヘッド部分)")

if __name__ == "__main__":
    main()
