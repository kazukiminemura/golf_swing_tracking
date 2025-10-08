from __future__ import annotations

import asyncio
from collections import defaultdict
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import UploadFile

from .logging_utils import debug_trace, get_logger
from .schemas import ArtifactPaths, JobDetail, JobProgressEvent, JobStats, JobStatus, JobSummary
from .storage import Storage

ProgressCallback = Callable[[str, float, Optional[str]], None]

logger = get_logger("jobs")


@dataclass
class JobRecord:
    id: str
    status: JobStatus
    device: str
    created_at: datetime
    updated_at: datetime
    input_path: Path
    params: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0
    stage: Optional[str] = None
    error: Optional[str] = None
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)
    stats: JobStats = field(default_factory=JobStats)


@dataclass
class PipelineResult:
    artifacts: ArtifactPaths
    stats: JobStats


class JobManager:
    def __init__(
        self,
        storage: Storage,
        pipeline_runner: Callable[[str, Path, Path, str, ProgressCallback], PipelineResult],
        device_default: str = "CPU",
        max_workers: int = 1,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.storage = storage
        self.pipeline_runner = pipeline_runner
        self.device_default = device_default
        self.max_workers = max_workers
        self.loop = loop or asyncio.get_event_loop()
        self.queue: "asyncio.Queue[str]" = asyncio.Queue()
        self.jobs: Dict[str, JobRecord] = {}
        self.listeners: Dict[str, List[asyncio.Queue[JobProgressEvent]]] = defaultdict(list)
        self.worker_tasks: List[asyncio.Task[Any]] = []
        self._lock = asyncio.Lock()
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        if self.worker_tasks:
            return
        logger.debug("Starting job manager with max_workers=%s", self.max_workers)
        for _ in range(self.max_workers):
            task = asyncio.create_task(self._worker_loop(), name="golftrack-worker")
            self.worker_tasks.append(task)

    async def stop(self) -> None:
        self._shutdown.set()
        logger.debug("Stopping job manager (active_workers=%s)", len(self.worker_tasks))
        for task in self.worker_tasks:
            task.cancel()
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()

    @debug_trace(name="JobManager.submit")
    async def submit(
        self,
        upload: UploadFile,
        device: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        job_id = uuid4().hex
        chosen_device = (device or self.device_default).upper()
        params = params or {}
        created = datetime.utcnow()
        input_path = self.storage.save_upload(job_id, upload)

        record = JobRecord(
            id=job_id,
            status=JobStatus.QUEUED,
            device=chosen_device,
            created_at=created,
            updated_at=created,
            input_path=input_path,
            params=params,
        )
        async with self._lock:
            self.jobs[job_id] = record

        await self.queue.put(job_id)
        self._dispatch(job_id, JobProgressEvent(job_id=job_id, status=JobStatus.QUEUED, progress=0.0, stage="queued"))
        logger.debug("Queued upload job_id=%s device=%s path=%s", job_id, chosen_device, input_path)
        return record

    @debug_trace(name="JobManager.enqueue_existing")
    async def enqueue_existing(
        self,
        source_path: Path,
        device: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> JobRecord:
        job_id = uuid4().hex
        chosen_device = (device or self.device_default).upper()
        params = params or {}
        created = datetime.utcnow()
        input_path = self.storage.link_existing(job_id, source_path)
        record = JobRecord(
            id=job_id,
            status=JobStatus.QUEUED,
            device=chosen_device,
            created_at=created,
            updated_at=created,
            input_path=input_path,
            params=params,
        )
        async with self._lock:
            self.jobs[job_id] = record
        await self.queue.put(job_id)
        self._dispatch(job_id, JobProgressEvent(job_id=job_id, status=JobStatus.QUEUED, progress=0.0, stage="queued"))
        logger.debug("Queued existing job_id=%s device=%s path=%s", job_id, chosen_device, input_path)
        return record

    async def get(self, job_id: str) -> Optional[JobRecord]:
        async with self._lock:
            return self.jobs.get(job_id)

    async def list(self) -> List[JobRecord]:
        async with self._lock:
            return list(self.jobs.values())

    async def subscribe(self, job_id: str) -> asyncio.Queue[JobProgressEvent]:
        queue: "asyncio.Queue[JobProgressEvent]" = asyncio.Queue()
        self.listeners[job_id].append(queue)
        # Push current state immediately if present
        record = await self.get(job_id)
        if record:
            queue.put_nowait(
                JobProgressEvent(
                    job_id=job_id,
                    status=record.status,
                    progress=record.progress,
                    stage=record.stage,
                    message=record.error,
                )
            )
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue[JobProgressEvent]) -> None:
        listeners = self.listeners.get(job_id, [])
        if queue in listeners:
            listeners.remove(queue)

    async def delete(self, job_id: str) -> bool:
        async with self._lock:
            record = self.jobs.pop(job_id, None)
        if not record:
            return False
        logger.debug("Deleting job job_id=%s", job_id)
        # Remove artifacts directory
        job_dir = self.storage.artifacts_dir / job_id
        if job_dir.exists():
            import shutil

            shutil.rmtree(job_dir)
        video = record.input_path
        if video.exists():
            video.unlink()
        return True

    @debug_trace(name="JobManager._worker_loop")
    async def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                job_id = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            record = await self.get(job_id)
            if not record:
                self.queue.task_done()
                continue

            await self._update_status(record, JobStatus.RUNNING, stage="starting", progress=0.05)
            logger.debug("Worker picked job_id=%s", job_id)

            try:
                result: PipelineResult = await asyncio.to_thread(
                    self.pipeline_runner,
                    job_id,
                    record.input_path,
                    self.storage.job_dir(job_id),
                    record.device,
                    self._make_progress_callback(job_id),
                )
                record.artifacts = result.artifacts
                record.stats = result.stats
                await self._update_status(record, JobStatus.SUCCEEDED, stage="completed", progress=1.0)
                logger.debug("Job succeeded job_id=%s", job_id)
            except Exception as exc:  # noqa: BLE001
                await self._fail_job(record, exc)
            finally:
                self.queue.task_done()

    async def _update_status(
        self,
        record: JobRecord,
        status: JobStatus,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        record.status = status
        if stage:
            record.stage = stage
        if progress is not None:
            record.progress = progress
        record.updated_at = datetime.utcnow()
        if message:
            record.error = message
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Job status update job_id=%s status=%s stage=%s progress=%.2f message=%s",
                record.id,
                record.status,
                record.stage,
                record.progress,
                message,
            )
        self._dispatch(
            record.id,
            JobProgressEvent(
                job_id=record.id,
                status=record.status,
                stage=record.stage,
                progress=record.progress,
                message=record.error,
            ),
        )

    async def _fail_job(self, record: JobRecord, exc: Exception) -> None:
        record.error = str(exc)
        logger.exception("Job failed job_id=%s error=%s", record.id, exc)
        await self._update_status(record, JobStatus.FAILED, stage="failed", progress=1.0, message=str(exc))

    def _make_progress_callback(self, job_id: str) -> ProgressCallback:
        def _cb(stage: str, progress: float, message: Optional[str] = None) -> None:
            def _update() -> None:
                record = self.jobs.get(job_id)
                if not record:
                    return
                record.stage = stage
                record.progress = min(max(progress, 0.0), 1.0)
                record.updated_at = datetime.utcnow()
                if message:
                    record.error = message
                self._dispatch(
                    job_id,
                    JobProgressEvent(
                        job_id=job_id,
                        status=record.status,
                        stage=stage,
                        progress=record.progress,
                        message=message,
                    ),
                )

            self.loop.call_soon_threadsafe(_update)

        return _cb

    def _dispatch(self, job_id: str, event: JobProgressEvent) -> None:
        queues = self.listeners.get(job_id, [])
        for queue in list(queues):
            if queue.full():
                continue
            queue.put_nowait(event)

    def to_summary(self, record: JobRecord) -> JobSummary:
        return JobSummary(
            id=record.id,
            status=record.status,
            device=record.device,
            created_at=record.created_at,
            updated_at=record.updated_at,
            progress=record.progress,
            stage=record.stage,
            error=record.error,
        )

    def to_detail(self, record: JobRecord) -> JobDetail:
        return JobDetail(job=self.to_summary(record), artifacts=record.artifacts, stats=record.stats)
