from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import Config
from .jobs import JobManager
from .logging_utils import configure_logging, get_logger
from .pipeline.pipeline import AnalysisPipeline
from .schemas import JobCreateResponse, JobDetail, JobListResponse, JobProgressEvent, JobStatus, SystemUsage
from .storage import Storage

logger = get_logger("server")

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


class AppState:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.storage = Storage(config.export.out_dir)
        self.pipeline = AnalysisPipeline(config)
        self.job_manager = JobManager(
            storage=self.storage,
            pipeline_runner=self.pipeline.run,
            device_default=config.runtime.device,
            max_workers=max(1, config.runtime.num_streams or 1),
        )


def create_app(config: Optional[Config] = None) -> FastAPI:
    cfg = config or Config.load()
    configure_logging(cfg)
    state = AppState(cfg)

    app = FastAPI(
        title="Golf Swing Tracking",
        version="0.2.0",
        description="FastAPI service for local golf swing analysis using OpenVINO.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory=str(cfg.server.static_dir)), name="static")

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info("Starting FastAPI app on %s:%s (debug=%s)", cfg.server.host, cfg.server.port, cfg.debug.enabled)
        await state.job_manager.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("Stopping FastAPI app")
        await state.job_manager.stop()

    async def get_manager() -> JobManager:
        return state.job_manager

    async def get_storage() -> Storage:
        return state.storage

    async def get_config() -> Config:
        return cfg

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_path = cfg.server.static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="UI not found")
        with index_path.open("r", encoding="utf-8") as fh:
            return HTMLResponse(fh.read())

    @app.post("/api/jobs", response_model=JobCreateResponse)
    async def create_job(
        request: Request,
        file: UploadFile = File(...),
        device: Optional[str] = Form(None),
        tracking: Optional[str] = Form(None),
        manager: JobManager = Depends(get_manager),
    ) -> JobCreateResponse:
        if file.content_type not in {"video/mp4", "video/x-matroska", "video/quicktime", "video/avi", "application/octet-stream"}:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        form_device = device.strip().upper() if device else None
        query_device_raw = request.query_params.get("device")
        query_device = query_device_raw.strip().upper() if query_device_raw else None
        # tracking: "on"/"off" via form or query
        form_tracking = tracking.strip().lower() if tracking else None
        query_tracking_raw = request.query_params.get("tracking")
        query_tracking = query_tracking_raw.strip().lower() if query_tracking_raw else None
        tracking_choice = form_tracking or query_tracking
        params: Dict[str, Any] = {}
        if tracking_choice in {"on", "off"}:
            params["enable_tracking"] = tracking_choice == "on"

        device_choice = form_device or query_device
        record = await manager.submit(file, device=device_choice, params=params or None)
        logger.info(
            "Job submitted job_id=%s filename=%s device_form=%s device_query=%s resolved=%s tracking=%s",
            record.id,
            file.filename,
            form_device,
            query_device,
            record.device,
            params.get("enable_tracking") if params else None,
        )
        return JobCreateResponse(job_id=record.id, status=record.status)

    @app.get("/api/jobs", response_model=JobListResponse)
    async def list_jobs(manager: JobManager = Depends(get_manager)) -> JobListResponse:
        jobs = await manager.list()
        summaries = [manager.to_summary(job) for job in jobs]
        summaries.sort(key=lambda job: job.created_at, reverse=True)
        return JobListResponse(jobs=summaries)

    @app.get("/api/jobs/{job_id}", response_model=JobDetail)
    async def get_job(job_id: str, manager: JobManager = Depends(get_manager)) -> JobDetail:
        record = await manager.get(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Job not found")
        return manager.to_detail(record)

    @app.get("/api/jobs/{job_id}/results", response_model=JobDetail)
    async def job_results(job_id: str, manager: JobManager = Depends(get_manager)) -> JobDetail:
        record = await manager.get(job_id)
        if not record:
            raise HTTPException(status_code=404, detail="Job not found")
        if record.status != JobStatus.SUCCEEDED:
            raise HTTPException(status_code=409, detail=f"Job is not complete (status={record.status})")
        return manager.to_detail(record)

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str, manager: JobManager = Depends(get_manager)) -> dict:
        deleted = await manager.delete(job_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Job not found")
        logger.info("Job deleted job_id=%s", job_id)
        return {"deleted": True}

    def _artifact_path(job_id: str, filename: str, storage: Storage) -> Path:
        return storage.job_dir(job_id) / filename

    @app.get("/api/jobs/{job_id}/video")
    async def download_video(job_id: str, manager: JobManager = Depends(get_manager), storage: Storage = Depends(get_storage)) -> FileResponse:
        record = await manager.get(job_id)
        if not record or not record.artifacts.video_mp4:
            raise HTTPException(status_code=404, detail="Video not available")
        path = _artifact_path(job_id, record.artifacts.video_mp4, storage)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Video file missing")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/jobs/{job_id}/trajectory.csv")
    async def download_csv(job_id: str, manager: JobManager = Depends(get_manager), storage: Storage = Depends(get_storage)) -> FileResponse:
        record = await manager.get(job_id)
        if not record or not record.artifacts.trajectory_csv:
            raise HTTPException(status_code=404, detail="Trajectory CSV not available")
        path = _artifact_path(job_id, record.artifacts.trajectory_csv, storage)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File missing")
        return FileResponse(path, media_type="text/csv")

    @app.get("/api/jobs/{job_id}/trajectory.json")
    async def download_json(job_id: str, manager: JobManager = Depends(get_manager), storage: Storage = Depends(get_storage)) -> FileResponse:
        record = await manager.get(job_id)
        if not record or not record.artifacts.trajectory_json:
            raise HTTPException(status_code=404, detail="Trajectory JSON not available")
        path = _artifact_path(job_id, record.artifacts.trajectory_json, storage)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File missing")
        return FileResponse(path, media_type="application/json")

    @app.get("/api/jobs/{job_id}/stats.json")
    async def download_stats(job_id: str, manager: JobManager = Depends(get_manager), storage: Storage = Depends(get_storage)) -> FileResponse:
        record = await manager.get(job_id)
        if not record or not record.artifacts.stats_json:
            raise HTTPException(status_code=404, detail="Stats not available")
        path = _artifact_path(job_id, record.artifacts.stats_json, storage)
        if not path.exists():
            raise HTTPException(status_code=404, detail="File missing")
        return FileResponse(path, media_type="application/json")

    @app.websocket("/ws/jobs/{job_id}")
    async def job_progress_websocket(websocket: WebSocket, job_id: str, manager: JobManager = Depends(get_manager)) -> None:
        await websocket.accept()
        queue = await manager.subscribe(job_id)
        try:
            while True:
                event: JobProgressEvent = await queue.get()
                await websocket.send_json(event.dict())
                if event.status in {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED}:
                    break
        except WebSocketDisconnect:
            pass
        finally:
            manager.unsubscribe(job_id, queue)
            with contextlib.suppress(Exception):
                await websocket.close()

    @app.get("/api/system-usage", response_model=SystemUsage)
    async def system_usage() -> SystemUsage:
        if psutil is None:
            return SystemUsage(cpu_percent=0.0, gpu_percent=0.0, npu_percent=0.0)
        cpu = psutil.cpu_percent(interval=0.1)
        gpu = 0.0
        npu = 0.0
        return SystemUsage(cpu_percent=cpu, gpu_percent=gpu, npu_percent=npu)

    return app


__all__ = ["create_app"]
