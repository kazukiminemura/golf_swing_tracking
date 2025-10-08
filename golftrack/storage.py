from __future__ import annotations

from pathlib import Path
from typing import Optional

import shutil

from fastapi import UploadFile


class Storage:
    """Manages file system layout for inputs and outputs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.inputs_dir = self.root / "inputs"
        self.artifacts_dir = self.root / "artifacts"
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        path = self.artifacts_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_upload(self, job_id: str, upload: UploadFile) -> Path:
        """Persist an uploaded file under inputs/."""
        suffix = Path(upload.filename or "input.mp4").suffix or ".mp4"
        dest = self.inputs_dir / f"{job_id}{suffix}"
        with dest.open("wb") as buffer:
            shutil.copyfileobj(upload.file, buffer)
        upload.file.close()
        return dest

    def link_existing(self, job_id: str, source: Path) -> Path:
        dest = self.inputs_dir / f"{job_id}{source.suffix}"
        if source.resolve() != dest.resolve():
            shutil.copy2(source, dest)
        return dest

    def artifact_path(self, job_id: str, name: str) -> Path:
        return self.job_dir(job_id) / name

