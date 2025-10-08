from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .config import Config
from .logging_utils import configure_logging, get_logger

logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Golf swing tracking CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a video file")
    analyze.add_argument("video", type=Path, help="Path to input video")
    analyze.add_argument("--device", type=str, help="Inference device (CPU/GPU/NPU)")
    analyze.add_argument("--config", type=Path, help="Optional config YAML")
    analyze.add_argument("--output", type=Path, help="Override output directory")
    analyze.add_argument("--job-id", type=str, help="Custom job id")

    return parser


def analyze_command(
    video: Path,
    device: Optional[str],
    config_path: Optional[Path],
    output_override: Optional[Path],
    job_id: Optional[str],
) -> int:
    if not video.exists():
        logger.error("Input video not found: %s", video)
        print(f"[ERROR] Input video not found: {video}", file=sys.stderr)
        return 1

    from .pipeline.pipeline import AnalysisPipeline
    from .storage import Storage

    config = Config.load(config_path)
    configure_logging(config)
    if device:
        config.runtime.device = device.upper()
    if output_override:
        config.export.out_dir = output_override
        config.export.resolve(Path.cwd())

    storage = Storage(config.export.out_dir)
    pipeline = AnalysisPipeline(config)
    job_identifier = job_id or uuid4().hex
    input_path = storage.link_existing(job_identifier, video)

    logger.info("Starting analysis job_id=%s device=%s video=%s", job_identifier, config.runtime.device, video)
    print(f"[INFO] Starting analysis job {job_identifier} on device {config.runtime.device}")

    def progress(stage: str, value: float, message: Optional[str] = None) -> None:
        percent = int(value * 100)
        message_str = f" - {message}" if message else ""
        logger.debug("CLI progress stage=%s progress=%.2f message=%s", stage, value, message)
        print(f"[{percent:3d}%] {stage}{message_str}", file=sys.stderr)

    result = pipeline.run(
        job_identifier,
        input_path,
        storage.job_dir(job_identifier),
        config.runtime.device,
        progress,
    )
    artifacts = result.artifacts

    logger.info("Analysis complete job_id=%s", job_identifier)
    print("[INFO] Analysis complete")
    print(f"  Video: {storage.job_dir(job_identifier) / artifacts.video_mp4}")
    print(f"  Trajectory CSV: {storage.job_dir(job_identifier) / artifacts.trajectory_csv}")
    print(f"  Trajectory JSON: {storage.job_dir(job_identifier) / artifacts.trajectory_json}")
    print(f"  Stats JSON: {storage.job_dir(job_identifier) / artifacts.stats_json}")
    if artifacts.snapshots:
        for snap in artifacts.snapshots:
            print(f"  Snapshot: {storage.job_dir(job_identifier) / snap}")
    print(f"  Max speed: {result.stats.max_speed_mps:.2f} m/s")
    print(f"  Avg speed: {result.stats.avg_speed_mps:.2f} m/s")
    if result.stats.impact_frame is not None:
        print(f"  Impact frame: {result.stats.impact_frame}")
    print(f"  Duration: {result.stats.duration_s:.2f}s")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return analyze_command(args.video, args.device, args.config, args.output, args.job_id)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
