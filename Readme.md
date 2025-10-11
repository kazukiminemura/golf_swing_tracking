## Golf Swing Tracking

Local-first golf swing analysis with OpenVINO inference and a simple FastAPI web UI. Upload a swing video to get an annotated overlay, trajectory, and speed metrics. A CLI is included for batch/offline runs.

### Features

- Web UI: upload videos, live job progress via WebSocket, result playback with trajectory overlay.
- Detector: OpenVINO-accelerated YOLO model (IR XML/BIN) with a lightweight tracker and smoothing.
- Metrics: per-swing trajectory, max/avg speed, impact frame, duration.
- Artifacts: MP4 overlay video, trajectory CSV/JSON, stats JSON, optional snapshots.
- Devices: choose `CPU`, `GPU`, or `NPU` per job; compiled models are cached per device.

### Quickstart

- Prereqs: Python 3.10+, pip; for GPU/NPU, install the proper OpenVINO drivers/toolkit for your platform.
- Install deps:
  - `pip install -r requirements.txt`

Run the web server:

- `uvicorn main:app --reload --port 8000`
- Open http://127.0.0.1:8000 and upload an MP4/MOV/AVI. Pick a device from the dropdown. When finished, use the download links under Results.

### CLI Usage

- Analyze a single video:
  - `python -m golftrack.cli analyze path\to\video.mp4 --device CPU`

- Optional flags:
  - `--config path\to\config.yaml` set an explicit config file
  - `--output path\to\out_dir` override the output directory
  - `--device CPU|GPU|NPU` pick an inference device
  - `--job-id custom123` set a custom job id

Exit code is non-zero on failure; file paths to artifacts are printed on success.

### Configuration

You can use a `config.yaml` in the project root (auto-detected) or environment variables. Defaults point to a bundled OpenVINO IR model at `yolov8_finetune_cpu/weights/best_openvino_model/best.xml`.

Example `config.yaml`:

```yaml
runtime:
  device: CPU         # CPU | GPU | NPU
  num_streams: 1
model:
  xml: yolov8_finetune_cpu/weights/best_openvino_model/best.xml
  conf_thres: 0.35
  iou_thres: 0.5
  input_size: 640
export:
  out_dir: outputs
calibration:
  scale_mm_per_px: 0.45
pipeline:
  smoothing_window: 5
debug:
  enabled: false
  log_level: INFO
  # log_file: logs/golftrack.log
```

Environment overrides:

- `GOLFTRACK_CONFIG` path to alternate YAML config
- `GOLFTRACK_DEVICE` override device (e.g., `CPU`, `GPU`)
- `GOLFTRACK_OUTPUT` output directory
- `GOLFTRACK_DEBUG` set `true` to enable debug tracing
- `GOLFTRACK_LOG_LEVEL` e.g., `DEBUG`, `INFO`
- `GOLFTRACK_LOG_FILE` write logs to this path

### Artifacts & Layout

- Default output root: `outputs/`
- Inputs are copied to `outputs/inputs/`.
- Per-job artifacts in `outputs/artifacts/<job_id>/`, including:
  - `result.mp4` or `result_h264.mp4`
  - `trajectory.csv`, `trajectory.json`
  - `stats.json`
  - `*.png` snapshots (optional)

### API Endpoints

- `GET /` serve the web UI
- `GET /healthz` health check
- `POST /api/jobs` upload a video (form field `file`, optional `device`)
- `GET /api/jobs` list jobs
- `GET /api/jobs/{job_id}` job detail
- `GET /api/jobs/{job_id}/results` job summary + artifacts map
- `GET /api/jobs/{job_id}/video` annotated MP4
- `GET /api/jobs/{job_id}/trajectory.csv|json` trajectory data
- `GET /api/jobs/{job_id}/stats.json` metrics
- `WS /ws/jobs/{job_id}` live progress events
- `GET /api/system-usage` CPU/GPU/NPU usage snapshot (best-effort)

### Models

- Default model: OpenVINO IR at `original_model`.
- To use a different IR, update `model.xml` (and `model.bin` if needed) in config.
- If you prefer Ultralytics weights for training or export, YOLO11 family weights are hosted by Ultralytics; for example, `yolo11n.pt` is available at:
  - `https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt`

### GPU/NPU Notes

- CPU runs work with just `pip install openvino` (already in `requirements.txt`).
- For GPU/NPU, install the appropriate OpenVINO runtime/drivers for your OS and hardware. Then set `--device GPU` or `GOLFTRACK_DEVICE=GPU`.

### Troubleshooting

- OpenVINO plugin not found: ensure OpenVINO runtime is installed for your device; try `CPU` first.
- PyTorch GPU wheels: if you need CUDA, install the matching `torch`/`torchvision` from the official PyTorch site.
- Large/long videos: processing is offline; keep the tab open or use the CLI.

### Repository Layout

- `main.py` FastAPI app entry point
- `golftrack/` core package (config, jobs, pipeline, CLI, server)
- `static/` web UI assets
- `docs/` notes and references
- `yolov8_finetune_cpu/weights/` default OpenVINO IR weights
- `outputs/` created at runtime for artifacts
