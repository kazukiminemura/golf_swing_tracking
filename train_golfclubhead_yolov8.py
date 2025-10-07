# train_golfclubhead_yolov8_xpu.py
import argparse
from pathlib import Path
import sys
import platform
import importlib
import traceback
import torch

# --- 環境情報出力 ---
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

from ultralytics import YOLO
import yaml
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
    ap = argparse.ArgumentParser(description="YOLOv8 fine-tune (club head, GPU/CPU support)")
    ap.add_argument("--data_root", required=True, help="root containing club_head/{train,valid}/{images,labels}")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=-1)   # -1:auto
    ap.add_argument("--device", default="", help='"0"/"cpu"/"cuda"')
    ap.add_argument("--project", default="runs/club_head")
    ap.add_argument("--name", default="yolov8_finetune")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--export_ov", action="store_true")
    ap.add_argument("--export_onnx", action="store_true")
    ap.add_argument("--use_existing_yaml", action="store_true", help="Use existing data.yaml file")
    args = ap.parse_args()

    # --- GPU (CUDA) support ---
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    # --- dataset root 確定 ---
    given = Path(args.data_root)
    if not given.exists():
        raise FileNotFoundError(f"[ERR] data_root does not exist: {given}")
    if (given / "club_head").exists():
        root = (given / "club_head").resolve()
    elif given.name == "club_head":
        root = given.resolve()
    else:
        root = given.resolve()

    # --- 必須フォルダ確認 ---
    required = ["train/images", "train/labels", "valid/images", "valid/labels"]
    missing = [str(root / p) for p in required if not (root / p).exists()]
    if missing:
        raise AssertionError(f"[ERR] missing required dataset folders under {root}:\n  " + "\n  ".join(missing))

    # --- YAML生成 or 既存利用 ---
    if args.use_existing_yaml and (root / "data.yaml").exists():
        yaml_path = root / "club_head_modified.yaml"
        with open(root / "data.yaml", 'r') as f:
            data_config = yaml.safe_load(f)
        data_config['path'] = str(root)
        data_config['train'] = 'train/images'
        data_config['val'] = 'valid/images'
        data_config['test'] = 'test/images'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        print(f"[INFO] modified dataset yaml -> {yaml_path}")
    else:
        yaml_path = root / "club_head.yaml"
        yaml_path.write_text(YAML_TPL.format(root=str(root).replace("\\", "/")), encoding="utf-8")
        print(f"[INFO] generated dataset yaml -> {yaml_path}")

    # --- 学習 ---
    # Device resolution: prefer CUDA when available and requested, otherwise CPU.
    user_dev = (args.device or "").strip().lower()
    if user_dev in {"cpu", "c"}:
        device_for_ultralytics = "cpu"
    elif user_dev in {"gpu", "cuda"} and cuda_available:
        device_for_ultralytics = "cuda"
    elif user_dev == "" and cuda_available:
        # Auto: allow GPU if available
        device_for_ultralytics = "cuda"
    elif any(ch.isdigit() for ch in user_dev) and cuda_available:
        # explicit device ids like '0' or '0,1'
        device_for_ultralytics = user_dev
    else:
        device_for_ultralytics = "cpu"
        if user_dev not in {"", "cpu", "c"}:
            print(f"[WARN] Requested device '{user_dev}' not available; falling back to CPU")

    print(f"[INFO] device resolved: cuda_available={cuda_available}, device_for_ultralytics={device_for_ultralytics}")

    model = YOLO(args.model)
    print("[INFO] preparing PyTorch training loop using Ultralytics model weights...")

    # Try to extract underlying PyTorch module from Ultralytics YOLO wrapper
    try:
        yolo_module = getattr(model, 'model', None) or model
    except Exception:
        yolo_module = model

    if yolo_module is None:
        raise RuntimeError("Could not locate underlying PyTorch model in the Ultralytics YOLO object")

    # Minimal Dataset for YOLO-format labels
    class YoloTextDataset(Dataset):
        def __init__(self, root, imgsz=640, split='train', transform=None):
            self.img_dir = Path(root) / f"{split}/images"
            self.lbl_dir = Path(root) / f"{split}/labels"
            self.imgs = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])
            self.imgsz = imgsz
            self.transform = transform or transforms.Compose([
                transforms.Resize((imgsz, imgsz)),
                transforms.ToTensor(),
            ])

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            img_path = self.imgs[idx]
            img = Image.open(img_path).convert('RGB')
            img_t = self.transform(img)

            # Load labels in YOLO txt format: class x_center y_center w h (normalized)
            lbl_path = self.lbl_dir / (img_path.stem + '.txt')
            targets = []
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            targets.append([cls, x, y, w, h])
            targets = torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0,5), dtype=torch.float32)
            return img_t, targets

    # Build datasets/loaders
    train_ds = YoloTextDataset(root=str(yaml_path.parent), imgsz=args.imgsz, split='train')
    val_ds = YoloTextDataset(root=str(yaml_path.parent), imgsz=args.imgsz, split='valid')

    batch_size = max(1, args.batch if args.batch > 0 else 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda b: list(zip(*b)))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=lambda b: list(zip(*b)))

    # Choose runtime device for tensors
    if device_for_ultralytics == 'cpu':
        device = torch.device('cpu')
    elif device_for_ultralytics in (None, 'cuda'):
        # prefer first CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # explicit device id
        device = torch.device('cuda:' + device_for_ultralytics.split(',')[0]) if torch.cuda.is_available() else torch.device('cpu')
    print(f"[INFO] using device: {device}")
    yolo_module.to(device)

    # Simple optimizer
    optimizer = torch.optim.SGD(yolo_module.parameters(), lr=0.01, momentum=0.9)

    best_val_loss = float('inf')
    results = type('R', (), {'save_dir': args.project})()

    for epoch in range(args.epochs):
        yolo_module.train()
        running_loss = 0.0
        n_batches = 0
        for imgs, targets in train_loader:
            # imgs: tuple of tensors, targets: tuple of tensors
            imgs = torch.stack(imgs).to(device)
            # Prepare targets as list of tensors per image for many detection models
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            # Many Ultralytics models accept (imgs, targets) in training and return a dict or loss tensor
            try:
                out = yolo_module(imgs, targets)
            except TypeError:
                # Fallback: try calling the wrapper model
                out = model.model(imgs, targets)

            # Interpret result as loss
            if isinstance(out, dict) and 'loss' in out:
                loss = out['loss']
            elif isinstance(out, torch.Tensor):
                loss = out
            else:
                raise RuntimeError("Unable to interpret model output as a loss. Ultralytics internal API may have changed.")

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_train_loss = running_loss / max(1, n_batches)
        print(f"[EPOCH {epoch+1}/{args.epochs}] train_loss={avg_train_loss:.4f}")

        # Validation loop (simple loss evaluation)
        yolo_module.eval()
        val_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = torch.stack(imgs).to(device)
                targets = [t.to(device) for t in targets]
                out = yolo_module(imgs, targets)
                if isinstance(out, dict) and 'loss' in out:
                    loss = out['loss']
                elif isinstance(out, torch.Tensor):
                    loss = out
                else:
                    raise RuntimeError("Unable to interpret model output as a loss during validation.")
                val_loss += float(loss.item())
                v_batches += 1

        avg_val_loss = val_loss / max(1, v_batches)
        print(f"[EPOCH {epoch+1}] val_loss={avg_val_loss:.4f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_dir = Path(args.project)
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / 'best.pt'
            torch.save({'model_state_dict': yolo_module.state_dict(), 'epoch': epoch+1}, ckpt_path)
            results.save_dir = str(save_dir)


    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"[INFO] best weights: {best}")

    # --- 検証 ---
    print("[INFO] validating...")
    val = model.val(model=str(best), data=str(yaml_path), imgsz=args.imgsz)
    print(f"[VAL] mAP50={val.box.map50:.4f}  mAP50-95={val.box.map:.4f}")

    # --- エクスポート ---
    if args.export_ov:
        print("[INFO] export OpenVINO IR...")
        model.export(model=str(best), format="openvino", imgsz=args.imgsz)
    if args.export_onnx:
        print("[INFO] export ONNX...")
        model.export(model=str(best), format="onnx", imgsz=args.imgsz, opset=12)

    print("\n[HINT] predict:")
    print(f'yolo task=detect mode=predict model="{best}" source=/path/to/test.mp4 imgsz={args.imgsz} conf=0.25')

if __name__ == "__main__":
    main()
