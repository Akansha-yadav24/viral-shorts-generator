"""
config.py
=========
Centralized configuration for the Viral Shorts Generator.

Loads:
  - Secrets (GEMINI_API_KEY) from a `.env` file via python-dotenv.
  - Tunable parameters from `config.json` (falls back to sane defaults
    if the file is missing or a key is absent).

All other modules import the `AppConfig` dataclass instance produced by
`load_config()` rather than reading files themselves, so configuration
stays in one place.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass
class AppConfig:
    # --- Secrets / API ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"  # fast + free-tier friendly

    # --- Clip constraints ---
    max_clip_seconds: int = 90
    min_clip_seconds: int = 20
    merge_gap_seconds: float = 1.5   # gaps smaller than this get merged
    max_clips_per_video: int = 20

    # --- Transcription ---
    whisper_model_size: str = "small"   # tiny/base/small/medium/large-v3
    whisper_device: str = "auto"        # auto/cpu/cuda
    whisper_compute_type: str = "auto"
    transcript_chunk_chars: int = 12000  # chars per Gemini request chunk

    # --- Video export ---
    output_width: int = 1080
    output_height: int = 1920           # 9:16 default; set to 0 to keep source AR
    keep_source_resolution: bool = False
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 18                        # quality (lower = better, 18 ~ visually lossless)
    preset: str = "medium"

    # --- Features (bonus) ---
    generate_subtitles: bool = True
    burn_in_captions: bool = False
    generate_thumbnails: bool = True
    resume_supported: bool = True
    max_export_workers: int = 2

    # --- Paths ---
    temp_dir_name: str = "temp"
    clips_dir_name: str = "clips"

    # --- Gemini retry behavior ---
    gemini_max_retries: int = 3
    gemini_timeout_seconds: int = 120

    def to_dict(self) -> dict:
        return asdict(self)


def _load_json_overrides(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[config] Warning: could not parse {path} ({e}). Using defaults.")
        return {}


def load_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> AppConfig:
    """
    Build an AppConfig by layering: dataclass defaults -> config.json -> .env secrets.
    """
    # 1. Load .env for secrets
    if env_path:
        load_dotenv(env_path, override=True)
    else:
        load_dotenv(override=False)

    cfg = AppConfig()

    # 2. Layer config.json overrides
    json_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    overrides = _load_json_overrides(json_path)
    for key, value in overrides.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            print(f"[config] Warning: unknown config key '{key}' in {json_path}, ignoring.")

    # 3. API key always comes from environment (never from config.json)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        cfg.gemini_api_key = api_key

    return cfg


def validate_config(cfg: AppConfig) -> list[str]:
    """Return a list of human-readable problems with the config (empty = OK)."""
    problems = []
    if not cfg.gemini_api_key:
        problems.append(
            "GEMINI_API_KEY is missing. Add it to a .env file (see .env.example)."
        )
    if cfg.min_clip_seconds >= cfg.max_clip_seconds:
        problems.append("min_clip_seconds must be less than max_clip_seconds.")
    if cfg.max_clip_seconds > 90:
        problems.append("max_clip_seconds should not exceed 90 (platform short-form limits).")
    return problems
