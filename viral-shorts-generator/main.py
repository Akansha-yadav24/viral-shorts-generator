#!/usr/bin/env python3
"""
main.py
=======
CLI entry point for the Viral Shorts Generator.

Usage
-----
Interactive mode (prompts for input):
    python main.py

Single video, scripted:
    python main.py --video "path/to/video.mp4" --output "path/to/output"

Batch mode (all videos in a folder):
    python main.py --batch "path/to/folder" --output "path/to/output"

Custom config file:
    python main.py --video video.mp4 --output out --config my_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import load_config, validate_config
from clip_generator import run_pipeline_for_video, run_batch
from utils import (
    setup_logger, validate_input_video, ensure_output_dir,
    check_ffmpeg_installed, ValidationError,
)


BANNER = r"""
=========================================================
   AI Viral Shorts Generator  (Powered by Gemini + Whisper)
=========================================================
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatically convert long videos into viral short clips using Gemini AI."
    )
    parser.add_argument("--video", type=str, help="Path to a single input video file.")
    parser.add_argument("--batch", type=str, help="Path to a folder of videos to process in batch.")
    parser.add_argument("--output", type=str, help="Output folder for generated clips.")
    parser.add_argument("--config", type=str, default=None, help="Path to a config.json file.")
    parser.add_argument("--env", type=str, default=None, help="Path to a .env file.")
    parser.add_argument("--no-resume", action="store_true", help="Disable resuming from cached transcripts.")
    return parser.parse_args()


def prompt_for_paths() -> tuple[str, str]:
    print(BANNER)
    video_path = input("Enter Video Path: ").strip().strip('"')
    output_path = input("Enter Output Folder: ").strip().strip('"')
    return video_path, output_path


def collect_batch_videos(folder: Path) -> list[Path]:
    valid_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_exts)


def main() -> int:
    args = parse_args()
    logger = setup_logger()

    print(BANNER)

    # --- Pre-flight checks ---
    if not check_ffmpeg_installed():
        logger.error(
            "FFmpeg/ffprobe not found on PATH. Install it first:\n"
            "  Windows : choco install ffmpeg   (or download from https://ffmpeg.org and add to PATH)\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg"
        )
        return 1

    cfg = load_config(config_path=args.config, env_path=args.env)
    problems = validate_config(cfg)
    if problems:
        for p in problems:
            logger.error(f"Config problem: {p}")
        return 1

    resume = not args.no_resume

    # --- Batch mode ---
    if args.batch:
        try:
            batch_folder = Path(args.batch).expanduser().resolve()
            if not batch_folder.is_dir():
                raise ValidationError(f"Not a directory: {batch_folder}")
            videos = collect_batch_videos(batch_folder)
            if not videos:
                logger.error(f"No video files found in {batch_folder}")
                return 1
            output_root = ensure_output_dir(args.output or str(batch_folder / "output"))
            logger.info(f"Batch mode: {len(videos)} video(s) found.")
            results = run_batch(videos, output_root, cfg, logger)
            total_clips = sum(len(r.clips) for r in results)
            print(f"\nBatch complete. Processed {len(results)}/{len(videos)} video(s), "
                  f"{total_clips} total clip(s) generated.")
            return 0
        except ValidationError as e:
            logger.error(str(e))
            return 1

    # --- Single video mode (scripted or interactive) ---
    video_str = args.video
    output_str = args.output
    if not video_str or not output_str:
        prompted_video, prompted_output = prompt_for_paths()
        video_str = video_str or prompted_video
        output_str = output_str or prompted_output

    try:
        video_path = validate_input_video(video_str)
        output_root = ensure_output_dir(output_str)
    except ValidationError as e:
        logger.error(str(e))
        return 1

    print("\nProcessing...")
    try:
        result = run_pipeline_for_video(video_path, output_root, cfg, logger, resume=resume)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1

    print(f"\nDone! Generated {len(result.clips)} clip(s).")
    print(f"Clips saved to: {output_root / cfg.clips_dir_name}")
    print(f"Manifest: {result.clips_json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
