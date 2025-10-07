# ultra2ov_track.py
import os, time, csv, argparse
import cv2, numpy as np
from collections import deque
from ultralytics import YOLO
from openvino.runtime import Core
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------- 画像前処理（letterbox） ----------
def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
    h, w = im.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    nh, nw = int(round(h*r)), int(round(w*r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0]-nh)//2; bottom = new_shape[0]-nh-top
    left = (new_shape[1]-nw)//2; right = new_shape[1]-nw-left
    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, r, left, top

def scale_boxes(xyxy, r, padx, pady):
    x1, y1, x2, y2 = xyxy.T
    x1 = (x1 - padx) / r; y1 = (y1 - pady) / r
    x2 = (x2 - padx) / r; y2 = (y2 - pady) / r
    return np.stack([x1, y1, x2, y2], axis=1)

# ---------- NMS (生出力用) ----------
def iou(box, boxes):
    xx1 = np.maximum(box[0], boxes[:,0]); yy1 = np.maximum(box[1], boxes[:,1])
    xx2 = np.minimum(box[2], boxes[:,2]); yy2 = np.minimum(box[3], boxes[:,3])
    w = np.maximum(0, xx2-xx1); h = np.maximum(0, yy2-yy1)
    inter = w*h
    a1 = (box[2]-box[0])*(box[3]-box[1]); a2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter / np.maximum(a1 + a2 - inter, 1e-6)

def nms(boxes, scores, thr=0.45):
    idxs = scores.argsort()[::-1]; keep=[]
    while idxs.size>0:
        i = idxs[0]; keep.append(i)
        if idxs.size==1: break
        ovr = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ovr < thr]
    return np.array(keep, int)

# ---------- OpenVINO 出力デコード（NMS済/生出力 両対応） ----------
def decode_outputs(outputs, conf_thr=0.25):
    dets=[]
    outs = [np.array(o.data if hasattr(o,"data") else o) for o in outputs]
    # 単一出力 [1,1,N,7] (x1,y1,x2,y2,score,class_id,batch)
    if len(outs)==1 and outs[0].ndim==4 and outs[0].shape[-1]==7:
        arr = outs[0][0,0]  # [N,7]
        for x1,y1,x2,y2,sc,cls,_ in arr:
            if sc >= conf_thr: dets.append([float(x1),float(y1),float(x2),float(y2),float(sc),int(cls)])
        return dets
    # 4出力（num, boxes, scores, classes）の変種
    if len(outs)>=3:
        try:
            flat=[a.reshape(-1, a.shape[-1]) if a.ndim>2 else a.reshape(-1) for a in outs]
            num=None; boxes=None; scores=None; classes=None
            for a in flat:
                if a.ndim==1 and num is None: num=int(a[0])
                elif a.ndim==2 and a.shape[1]==4 and boxes is None: boxes=a
                elif a.ndim==1 and scores is None: scores=a
                elif a.ndim==1 and classes is None: classes=a
            if boxes is not None and scores is not None and classes is not None:
                for (x1,y1,x2,y2), s, c in zip(boxes, scores, classes):
                    if s>=conf_thr: dets.append([float(x1),float(y1),float(x2),float(y2),float(s),int(c)])
                return dets
        except Exception:
            pass
    # 生出力 [1, no, 85] (xywh + obj + cls…)
    big = max(outs, key=lambda a: a.size)
    if big.ndim==3 and big.shape[-1]>=6:
        out = big[0]
        xywh = out[:, :4]; obj = out[:, 4:5]; cls = out[:, 5:]
        probs = (obj*cls); scores = probs.max(axis=1); cls_ids = probs.argmax(axis=1)
        keep = scores >= conf_thr
        xywh, scores, cls_ids = xywh[keep], scores[keep], cls_ids[keep]
        xyxy = np.empty_like(xywh); xyxy[:,0:2] = xywh[:,0:2]-xywh[:,2:4]/2; xyxy[:,2:4] = xywh[:,0:2]+xywh[:,2:4]/2
        if len(xyxy)>0:
            k = nms(xyxy, scores, 0.45)
            for i in k:
                x1,y1,x2,y2 = xyxy[i]
                dets.append([float(x1),float(y1),float(x2),float(y2),float(scores[i]),int(cls_ids[i])])
    return dets

# ---------- 速度 ----------
def speed(prev, now, dt, ppm):
    if prev is None or dt<=0: return (0.0, None, None)
    vpx = float(np.hypot(now[0]-prev[0], now[1]-prev[1]) / dt)
    if ppm>0: ms=vpx/ppm; return vpx, ms, ms*3.6
    return vpx, None, None

# ---------- 引数 ----------
def parse_args():
    p=argparse.ArgumentParser(description="Ultralytics auto-download -> OpenVINO -> DeepSORT")
    p.add_argument("--video", type=str, default="0", help="Video path or cam index (e.g., 0)")
    p.add_argument("--ultra_model", type=str, required=True,
                   help="Ultralytics model spec: e.g. yolov8n.pt or HUB URL/ID")
    p.add_argument("--device", type=str, default="AUTO", help="AUTO/GPU/CPU/NPU")
    p.add_argument("--resize_width", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--class_id", type=int, default=-1, help="Filter by this class id (optional)")
    p.add_argument("--ppm", type=float, default=0.0, help="Pixels per meter (0: px/sのみ)")
    p.add_argument("--trail", type=int, default=80)
    p.add_argument("--out_video", type=str, default="")
    p.add_argument("--out_csv", type=str, default="trajectory.csv")
    return p.parse_args()

# ---------- Ultralytics から自動DL → OpenVINO へ ----------
def prepare_ir_from_ultra(model_spec):
    """
    model_spec に yolov8n.pt のような公開モデル名、または
    Ultralytics HUB の URL/ID を渡すと、内部で自動DLして OpenVINO に export。
    戻り値は model.xml のパス。
    """
    # 例:
    # - "yolov8n.pt" → 自動DL
    # - "https://hub.ultralytics.com/models/XXXX"（or "XXXX"）→ APIキー必要
    if model_spec and not model_spec.endswith(".pt") and "yolov8" in model_spec and ".pt" not in model_spec:
        # "yolov8n" のような短縮記法に .pt を補完
        model_spec = model_spec + ".pt"
    m = YOLO(model_spec)  # ここで自動DL/HUB解決
    # OpenVINO へエクスポート（出力ディレクトリは ultralytics の既定）
    print("[INFO] Exporting to OpenVINO IR...")
    res = m.export(format="openvino")  # best_openvino_model/
    # res はパス文字列 or dict（Ultralytics版により差異あり）なので、両対応で探す
    candidates = [
        os.path.join(os.getcwd(), "best_openvino_model", "model.xml"),
        os.path.join(os.path.dirname(getattr(m, "ckpt_path", "") or "."), "best_openvino_model", "model.xml"),
    ]
    for c in candidates:
        if os.path.exists(c): return c
    # fallback: 直近に作られた model.xml を探索
    found = None
    for root, _, files in os.walk(os.getcwd()):
        if "model.xml" in files and "openvino" in root.lower():
            found = os.path.join(root, "model.xml"); break
    if not found:
        raise RuntimeError("OpenVINO IR export failed. Check Ultralytics logs.")
    return found

def open_video(src):
    cap = cv2.VideoCapture(0 if src.isdigit() else src)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video source: {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, w, h, fps

def writer(path, fps, w, h):
    if not path: return None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (w, h))

def main():
    args = parse_args()
    # 1) Ultralytics から自動DL → OpenVINO へ
    xml_path = prepare_ir_from_ultra(args.ultra_model)
    print(f"[INFO] IR ready: {xml_path}")

    # 2) OpenVINO 推論セットアップ
    core = Core()
    compiled = core.compile_model(xml_path, args.device)
    inp = compiled.input(0); H, W = int(inp.shape[2]), int(inp.shape[3])
    out_tensors = [compiled.output(i) for i in range(len(compiled.outputs))]

    # 3) 動画 & 出力
    cap, ow, oh, fps = open_video(args.video)
    pw = args.resize_width if args.resize_width>0 else ow
    scale = pw/float(ow); ph = int(round(oh*scale))
    vw = writer(args.out_video, fps, pw, ph)

    tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)
    trails = {}
    csvf = open(args.out_csv, "w", newline="", encoding="utf-8")
    csvw = csv.writer(csvf); csvw.writerow(["frame","time","track_id","cx","cy","v_px_s","v_m_s","v_km_h"])

    last = time.time(); idx=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if scale!=1.0: frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)

        # 前処理
        img_l, r, px, py = letterbox(frame, (H, W))
        blob = img_l[:, :, ::-1].transpose(2,0,1).astype(np.float32)/255.0
        blob = np.ascontiguousarray(blob)[None, ...]

        # 推論
        res = compiled({inp.get_any_name(): blob})
        outputs = [res[o] for o in res]
        dets = decode_outputs(outputs, conf_thr=args.conf)

        # クラスフィルタ
        if args.class_id >= 0:
            dets = [d for d in dets if d[5]==args.class_id]

        # 元座標へ
        if dets:
            boxes = np.array([d[:4] for d in dets], np.float32)
            scores = np.array([d[4] for d in dets], np.float32)
            clses  = np.array([d[5] for d in dets], np.int32)
            boxes = scale_boxes(boxes, r, px, py)
        else:
            boxes = np.zeros((0,4), np.float32); scores=np.zeros((0,),np.float32); clses=np.zeros((0,),np.int32)

        # DeepSORT 入力
        ds_in = []
        for (x1,y1,x2,y2), sc, c in zip(boxes, scores, clses):
            ds_in.append([[float(x1),float(y1),float(x2-x1),float(y2-y1)], float(sc), int(c)])
        tracks = tracker.update_tracks(ds_in, frame=frame)

        # 可視化 & CSV
        now = time.time(); dt = now - last; last = now
        for trk in tracks:
            if not trk.is_confirmed(): continue
            tid = trk.track_id
            l,t,r_,b = map(int, trk.to_ltrb()); cx,cy = (l+r_)//2, (t+b)//2
            if tid not in trails: trails[tid]=deque(maxlen=args.trail)
            prev = trails[tid][-1] if len(trails[tid]) else None
            trails[tid].append((cx,cy))
            vpx, vms, vkmh = speed(prev,(cx,cy),dt,args.ppm)

            cv2.rectangle(frame,(l,t),(r_,b),(0,255,0),2)
            label=f"id {tid} "+(f"{vkmh:5.1f} km/h" if vkmh is not None else (f"{vms:5.2f} m/s" if vms is not None else f"{vpx:6.1f} px/s"))
            cv2.putText(frame,label,(l,max(0,t-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA)

            pts=list(trails[tid])
            for i in range(1,len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0,255,255), 2)

            t_rel = idx/float(fps)
            csvw.writerow([idx, f"{t_rel:.3f}", tid, cx, cy,
                           f"{vpx:.3f}", f"{vms:.3f}" if vms is not None else "", f"{vkmh:.3f}" if vkmh is not None else ""])

        cv2.putText(frame, f"Ultralytics->OpenVINO ({args.device}) + DeepSORT", (10,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("tracking", frame)
        if vw is not None: vw.write(frame)
        if (cv2.waitKey(1)&0xFF) in (27, ord('q')): break
        idx += 1

    cap.release(); csvf.close()
    if vw is not None: vw.release()
    cv2.destroyAllWindows()
    print(f"[INFO] CSV saved to {args.out_csv}")
    if args.out_video: print(f"[INFO] Video saved to {args.out_video}")

if __name__ == "__main__":
    main()
