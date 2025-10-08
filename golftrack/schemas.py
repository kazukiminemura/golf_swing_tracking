from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class ArtifactPaths(BaseModel):
    video_mp4: Optional[str] = None
    trajectory_csv: Optional[str] = None
    trajectory_json: Optional[str] = None
    snapshots: List[str] = Field(default_factory=list)
    stats_json: Optional[str] = None


class JobStats(BaseModel):
    max_speed_mps: float = 0.0
    avg_speed_mps: float = 0.0
    impact_frame: Optional[int] = None
    duration_s: float = 0.0


class JobSummary(BaseModel):
    id: str
    status: JobStatus
    device: str
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0
    stage: Optional[str] = None
    error: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class JobDetail(BaseModel):
    job: JobSummary
    artifacts: ArtifactPaths
    stats: JobStats


class JobListResponse(BaseModel):
    jobs: List[JobSummary]


class SystemUsage(BaseModel):
    cpu_percent: float
    gpu_percent: float
    npu_percent: float


class JobProgressEvent(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    stage: Optional[str] = None
    message: Optional[str] = None
