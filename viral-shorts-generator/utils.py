"""
utils.py
========
Shared helper functions used across the pipeline:
  - Logging setup
  - Timestamp parsing/formatting (HH:MM:SS[.ms] <-> seconds)
  - Path validation
  - FFmpeg availability check
  - Safe JSON extraction from LLM text output
  - Temp file cleanup
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logger(name: str = "viral_shorts", log_file: Optional[str] = "pipeline.log") -> logging.Logger:
    """Configure and return a logger that writes to console and (optionally) a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                             datefmt="%H:%M:%S")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError:
            pass  # non-fatal if we can't write logs to disk

    return logger


# --------------------------------------------------------------------------- #
# Timestamp helpers
# --------------------------------------------------------------------------- #

_TS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:\.(\d+))?$")


def timestamp_to_seconds(ts: str) -> float:
    """
    Convert 'HH:MM:SS', 'MM:SS', or 'HH:MM:SS.ms' into float seconds.
    Raises ValueError on malformed input.
    """
    ts = ts.strip()
    match = _TS_RE.match(ts)
    if not match:
        raise ValueError(f"Malformed timestamp: '{ts}'")
    hours, minutes, seconds, millis = match.groups()
    hours = int(hours) if hours else 0
    minutes = int(minutes)
    seconds = int(seconds)
    millis = float(f"0.{millis}") if millis else 0.0
    return hours * 3600 + minutes * 60 + seconds + millis


def seconds_to_timestamp(seconds: float, always_hours: bool = True) -> str:
    """Convert float seconds into 'HH:MM:SS.mmm' (or 'MM:SS.mmm' if always_hours=False)."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if always_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes:02d}:{secs:06.3f}"


def seconds_to_srt_timestamp(seconds: float) -> str:
    """Convert float seconds into SRT format: HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# --------------------------------------------------------------------------- #
# Path / environment validation
# --------------------------------------------------------------------------- #

class ValidationError(Exception):
    """Raised for user-facing input validation failures."""


def validate_input_video(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise ValidationError(f"Video file not found: {path}")
    if not path.is_file():
        raise ValidationError(f"Path is not a file: {path}")
    valid_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
    if path.suffix.lower() not in valid_exts:
        raise ValidationError(
            f"Unsupported video extension '{path.suffix}'. Supported: {sorted(valid_exts)}"
        )
    if path.stat().st_size == 0:
        raise ValidationError(f"Video file is empty (0 bytes): {path}")
    return path


def ensure_output_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_ffmpeg_installed() -> bool:
    """Return True if ffmpeg and ffprobe are available on PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def get_video_duration_seconds(video_path: Path) -> float:
    """Use ffprobe to get duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        raise RuntimeError(f"Could not read video duration via ffprobe: {e}")


# --------------------------------------------------------------------------- #
# JSON extraction (LLMs sometimes wrap JSON in markdown fences despite instructions)
# --------------------------------------------------------------------------- #

def extract_json_array(text: str) -> Any:
    """
    Robustly extract a JSON array from raw LLM text output, stripping
    accidental markdown code fences or leading/trailing prose.
    Raises ValueError if no valid JSON array can be parsed.
    """
    text = text.strip()

    # Strip markdown fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Sometimes models wrap the array in {"clips": [...]}
            for value in data.values():
                if isinstance(value, list):
                    return value
    except json.JSONDecodeError:
        pass

    # Fall back to locating the first '[' and last ']' in the text
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON array from model output: {e}")

    raise ValueError("No JSON array found in model output.")


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #

def cleanup_dir(path: Path, logger: Optional[logging.Logger] = None) -> None:
    """Remove all files inside a temp directory (keeps the directory itself)."""
    if not path.exists():
        return
    for item in path.iterdir():
        try:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except OSError as e:
            if logger:
                logger.warning(f"Could not remove temp item {item}: {e}")


def safe_filename(name: str, max_len: int = 80) -> str:
    """Sanitize a string for use as a filename fragment."""
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[\s]+", "_", name)
    return name[:max_len] if name else "clip"
