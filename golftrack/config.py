from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import os
import yaml


@dataclass
class RuntimeConfig:
    """Device/runtime related configuration."""

    device: str = "CPU"  # CPU | GPU | NPU
    num_streams: int = 1


@dataclass
class ModelConfig:
    """Model loading parameters."""

    xml: Path = Path("yolov8_finetune_cpu/weights/best_openvino_model/best.xml")
    bin: Optional[Path] = None
    conf_thres: float = 0.35
    iou_thres: float = 0.5
    input_size: int = 640

    def resolve(self, base_dir: Path) -> None:
        self.xml = (self.xml if self.xml.is_absolute() else base_dir / self.xml).resolve()
        if self.bin is None:
            self.bin = self.xml.with_suffix(".bin")
        elif not self.bin.is_absolute():
            self.bin = (base_dir / self.bin).resolve()


@dataclass
class CalibrationConfig:
    """Physical calibration for velocity estimation."""

    scale_mm_per_px: float = 0.45
    club_length_mm: float = 1100.0


@dataclass
class ExportConfig:
    """Artifacts export settings."""

    out_dir: Path = Path("outputs")
    snapshot_stride: int = 10

    def resolve(self, base_dir: Path) -> None:
        if not self.out_dir.is_absolute():
            self.out_dir = (base_dir / self.out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ServerConfig:
    """FastAPI server configuration."""

    host: str = "127.0.0.1"
    port: int = 8000
    static_dir: Path = Path("static")

    def resolve(self, base_dir: Path) -> None:
        if not self.static_dir.is_absolute():
            self.static_dir = (base_dir / self.static_dir).resolve()
        self.static_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    smoothing_window: int = 5
    min_track_confidence: float = 0.4
    max_age: int = 30
    n_init: int = 3
    min_bbox_area_px: float = 25.0
    max_bbox_area_ratio: float = 0.02
    min_bbox_aspect_ratio: float = 0.4
    max_bbox_aspect_ratio: float = 3.0
    max_center_jump_px: float = 160.0
    max_area_change_ratio: float = 4.0


@dataclass
class DebugConfig:
    """Debug/logging controls."""

    enabled: bool = False
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def resolve(self, base_dir: Path) -> None:
        if self.log_file and not self.log_file.is_absolute():
            self.log_file = (base_dir / self.log_file).resolve()
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        base_dir = Path(os.getenv("GOLFTRACK_BASEDIR", Path.cwd()))
        config = cls()

        raw: Dict[str, Any] = {}
        config_path: Optional[Path] = None

        env_path = os.getenv("GOLFTRACK_CONFIG")
        if env_path:
            config_path = Path(env_path)
        elif path:
            config_path = Path(path)
        else:
            default_path = base_dir / "config.yaml"
            if default_path.exists():
                config_path = default_path

        if config_path and config_path.exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}

        if raw:
            config = cls.from_dict(raw)

        # Apply environment overrides
        device_override = os.getenv("GOLFTRACK_DEVICE")
        if device_override:
            config.runtime.device = device_override

        out_dir_override = os.getenv("GOLFTRACK_OUTPUT")
        if out_dir_override:
            config.export.out_dir = Path(out_dir_override)

        static_dir_override = os.getenv("GOLFTRACK_STATIC")
        if static_dir_override:
            config.server.static_dir = Path(static_dir_override)

        debug_flag = os.getenv("GOLFTRACK_DEBUG")
        if debug_flag:
            config.debug.enabled = debug_flag.lower() in {"1", "true", "yes", "on"}

        debug_level = os.getenv("GOLFTRACK_LOG_LEVEL")
        if debug_level:
            config.debug.log_level = debug_level

        debug_file = os.getenv("GOLFTRACK_LOG_FILE")
        if debug_file:
            config.debug.log_file = Path(debug_file)

        config.model.resolve(base_dir)
        config.export.resolve(base_dir)
        config.server.resolve(base_dir)
        config.debug.resolve(base_dir)
        return config

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        def _build(dataclass_cls, data: Dict[str, Any]):
            field_names = {f.name for f in dataclass_cls.__dataclass_fields__.values()}  # type: ignore
            init_kwargs = {}
            for key, value in data.items():
                if key in field_names:
                    init_kwargs[key] = value
            return dataclass_cls(**init_kwargs)

        runtime = _build(RuntimeConfig, payload.get("runtime", {}))
        model = _build(ModelConfig, payload.get("model", {}))
        calibration = _build(CalibrationConfig, payload.get("calibration", {}))
        export = _build(ExportConfig, payload.get("export", {}))
        server = _build(ServerConfig, payload.get("server", {}))
        pipeline = _build(PipelineConfig, payload.get("pipeline", {}))
        debug = _build(DebugConfig, payload.get("debug", {}))

        return cls(
            runtime=runtime,
            model=model,
            calibration=calibration,
            export=export,
            server=server,
            pipeline=pipeline,
            debug=debug,
        )


__all__ = ["Config"]
