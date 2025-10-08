## Golf Swing Tracking

This project implements a local-first golf swing analysis workflow powered by OpenVINO and YOLOv8. The system provides both a FastAPI-based web UI and a CLI for batch processing golf swing videos.

### Features

- Upload swing videos via browser or CLI and receive an annotated overlay video.
- YOLOv8 (OpenVINO) based club head detection with a Kalman filter trajectory tracker.
- Velocity estimation using configurable millimetre-per-pixel calibration.
- Exported artifacts: overlay MP4, trajectory CSV/JSON, stats JSON, and impact snapshots.
- Web UI with live job progress via WebSocket and interactive trajectory canvas overlay.

### Prerequisites

- Python 3.10 or later
- OpenVINO runtime dependencies for your device (CPU/GPU/NPU)
- (Optional) GPU/NPU drivers if targeting those devices

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the API Server

```bash
uvicorn main:app --reload --port 8000
```

Open a browser at `http://127.0.0.1:8000/` to access the UI. Upload a swing video, select a target device, and monitor progress in real time. Completed jobs expose download links for all generated artifacts.

API endpoints (partial list):

- `POST /api/jobs` – upload a video for analysis
- `GET /api/jobs` – list jobs and statuses
- `GET /api/jobs/{id}/results` – retrieve job summary/artifacts
- `GET /api/jobs/{id}/video` – download the annotated video
- `WS /ws/jobs/{id}` – subscribe to live job progress

### CLI Usage

```bash
python -m golftrack.cli analyze path/to/video.mp4 --device CPU
```

Optional flags:

- `--config path/to/config.yaml`: load custom configuration
- `--output /path/to/output_dir`: override artifact directory
- `--device GPU|CPU|NPU`: target inference device

### Configuration

Runtime options can be customised via `config.yaml` (auto-detected from the project root) or environment variables:

```yaml
runtime:
  device: GPU
  num_streams: 2
model:
  xml: yolov8_finetune_cpu/weights/best_openvino_model/best.xml
  conf_thres: 0.35
export:
  out_dir: outputs
calibration:
  scale_mm_per_px: 0.45
pipeline:
  smoothing_window: 5
debug:
  enabled: false
  log_level: INFO
  log_file: logs/golftrack.log
```

Environment overrides:

- `GOLFTRACK_CONFIG`: alternate config file
- `GOLFTRACK_DEVICE`: force device (e.g., `CPU`, `GPU`)
- `GOLFTRACK_OUTPUT`: output directory
- `GOLFTRACK_DEBUG`: set to `true` to enable verbose tracing
- `GOLFTRACK_LOG_LEVEL`: override log level (e.g., `DEBUG`)
- `GOLFTRACK_LOG_FILE`: write logs to a specific file

### GPU / NPU Execution

- Ensure the appropriate OpenVINO GPU/NPU drivers are installed (see Intel® OpenVINO™ Toolkit setup guides).
- Choose the target device per job from the Web UI drop-down or via CLI `--device GPU`.
- For a default device, set `runtime.device: GPU` in `config.yaml` or export `GOLFTRACK_DEVICE=GPU`.
- The backend now keeps a dedicated compiled model per device, so you can mix CPU/GPU jobs without restarts.

### Repository Layout

- `main.py`: FastAPI application entrypoint
- `golftrack/`: core package (config, job manager, pipeline, CLI)
- `static/`: web UI assets
- `docs/`: requirements and architecture references
- `outputs/`: generated at runtime for artifacts

### Debugging Failures

Enable `debug.enabled: true` (or export `GOLFTRACK_DEBUG=true`) to capture detailed function-level traces. When active, the service logs entry/exit for key pipeline stages, job workers, and CLI runs, and records stack traces for any failing function. Configure `debug.log_file` to persist logs for later review.

### Next Steps

- Integrate system usage telemetry for GPU/NPU backends
- Extend tracker to multi-object scenarios and ByteTrack compatibility
- Add automated tests and synthetic video fixtures
