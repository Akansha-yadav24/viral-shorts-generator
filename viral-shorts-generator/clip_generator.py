"""
clip_generator.py
==================
Top-level orchestrator that runs the full pipeline for ONE input video:

  extract audio -> transcribe -> chunk transcript -> analyze with Gemini
  -> validate/clean timestamps -> export clips -> (optional) subtitles/thumbnails
  -> write clips.json

Also supports resuming an interrupted run: if a `clips.json` / transcript
cache already exists in the video's temp dir, expensive steps are skipped.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from config import AppConfig
from gemini_client import GeminiClient, ClipCandidate
from transcriber import (
    TranscriptSegment, extract_audio, transcribe_audio,
    chunk_transcript,
)
from utils import (
    get_video_duration_seconds, cleanup_dir, ValidationError,
)
from video_editor import (
    FinalClip, validate_and_clean_candidates, export_all_clips,
    generate_srt_for_clip, burn_in_subtitles, generate_thumbnail,
)


class PipelineResult:
    def __init__(self, video_path: Path, clips: List[FinalClip], clips_json_path: Path):
        self.video_path = video_path
        self.clips = clips
        self.clips_json_path = clips_json_path


def _transcript_cache_path(temp_dir: Path) -> Path:
    return temp_dir / "transcript_cache.json"


def _save_transcript_cache(segments: List[TranscriptSegment], path: Path) -> None:
    data = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    path.write_text(json.dumps(data), encoding="utf-8")


def _load_transcript_cache(path: Path) -> Optional[List[TranscriptSegment]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [TranscriptSegment(**item) for item in data]
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def run_pipeline_for_video(
    video_path: Path,
    output_root: Path,
    cfg: AppConfig,
    logger: logging.Logger,
    resume: bool = True,
) -> PipelineResult:
    """
    Run the complete video -> shorts pipeline for a single video file.
    `output_root` is the destination folder chosen by the user; a `clips/`
    subfolder holds the exported MP4s and `clips.json` sits alongside it.
    """
    temp_dir = output_root / cfg.temp_dir_name / video_path.stem
    clips_dir = output_root / cfg.clips_dir_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Processing: {video_path.name} ===")

    # --- Duration check ---
    duration = get_video_duration_seconds(video_path)
    logger.info(f"Video duration: {duration / 60:.1f} minutes")

    # --- Step 1: Audio extraction + transcription (resumable) ---
    cache_path = _transcript_cache_path(temp_dir)
    segments: Optional[List[TranscriptSegment]] = None
    if resume and cfg.resume_supported:
        segments = _load_transcript_cache(cache_path)
        if segments:
            logger.info(f"Resume: loaded cached transcript ({len(segments)} segments), skipping re-transcription.")

    if not segments:
        with tqdm(total=2, desc="Audio + Transcription", unit="step") as pbar:
            audio_path = extract_audio(video_path, temp_dir, logger)
            pbar.update(1)
            segments = transcribe_audio(
                audio_path,
                model_size=cfg.whisper_model_size,
                device=cfg.whisper_device,
                compute_type=cfg.whisper_compute_type,
                logger=logger,
            )
            pbar.update(1)
        if not segments:
            raise RuntimeError("Transcription produced no text. The video may be silent or corrupted.")
        _save_transcript_cache(segments, cache_path)

    # --- Step 2: Chunk transcript for Gemini ---
    chunks = chunk_transcript(segments, max_chars=cfg.transcript_chunk_chars)
    logger.info(f"Transcript split into {len(chunks)} chunk(s) for Gemini analysis.")

    # --- Step 3: Gemini analysis ---
    client = GeminiClient(
        api_key=cfg.gemini_api_key,
        model_name=cfg.gemini_model,
        max_retries=cfg.gemini_max_retries,
        timeout_seconds=cfg.gemini_timeout_seconds,
        logger=logger,
    )
    candidates: List[ClipCandidate] = client.analyze_transcript_chunks(
        chunks, min_sec=cfg.min_clip_seconds, max_sec=cfg.max_clip_seconds
    )
    if not candidates:
        raise RuntimeError("Gemini did not return any viable clip candidates for this video.")

    # --- Step 4: Validate, dedupe, merge timestamps ---
    final_clips = validate_and_clean_candidates(
        candidates,
        video_duration=duration,
        min_sec=cfg.min_clip_seconds,
        max_sec=cfg.max_clip_seconds,
        merge_gap=cfg.merge_gap_seconds,
        max_clips=cfg.max_clips_per_video,
        logger=logger,
    )
    if not final_clips:
        raise RuntimeError("No clips survived validation (all were too short, overlapping, or invalid).")

    # --- Step 5: Export clips ---
    logger.info(f"Exporting {len(final_clips)} clip(s)...")
    exported = export_all_clips(
        video_path=video_path,
        clips=final_clips,
        output_dir=clips_dir,
        max_workers=cfg.max_export_workers,
        codec=cfg.video_codec,
        audio_codec=cfg.audio_codec,
        crf=cfg.crf,
        preset=cfg.preset,
        target_width=cfg.output_width,
        target_height=cfg.output_height,
        keep_source_resolution=cfg.keep_source_resolution,
        logger=logger,
    )

    # --- Step 6 (bonus): subtitles / burned-in captions / thumbnails ---
    if cfg.generate_subtitles or cfg.generate_thumbnails:
        for clip in tqdm(exported, desc="Post-processing (subs/thumbnails)", unit="clip"):
            clip_path = clips_dir / clip.filename
            if cfg.generate_subtitles:
                srt_path = generate_srt_for_clip(clip, segments, clips_dir)
                if cfg.burn_in_captions:
                    burned_path = clips_dir / f"clip_{clip.index:03d}_captioned.mp4"
                    burn_in_subtitles(clip_path, srt_path, burned_path, logger)
            if cfg.generate_thumbnails:
                generate_thumbnail(clip_path, clips_dir, clip.index)

    # --- Step 7: Write clips.json ---
    clips_json_path = output_root / "clips.json"
    manifest = [
        {
            "title": c.title,
            "start": c.start,
            "end": c.end,
            "duration": round(c.duration, 2),
            "score": c.score,
            "reason": c.reason,
            "filename": c.filename,
        }
        for c in exported
    ]
    clips_json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Wrote manifest -> {clips_json_path}")

    # --- Cleanup temp audio (keep transcript cache for resume safety) ---
    for f in temp_dir.glob("*_audio.wav"):
        try:
            f.unlink()
        except OSError:
            pass

    logger.info(f"Done! Generated {len(exported)} clip(s) in {clips_dir}")
    return PipelineResult(video_path=video_path, clips=exported, clips_json_path=clips_json_path)


def run_batch(
    video_paths: List[Path],
    output_root: Path,
    cfg: AppConfig,
    logger: logging.Logger,
) -> List[PipelineResult]:
    """Process multiple videos sequentially, continuing past individual failures."""
    results = []
    for i, video_path in enumerate(video_paths, start=1):
        logger.info(f"--- Batch item {i}/{len(video_paths)}: {video_path.name} ---")
        try:
            video_output = output_root / video_path.stem
            result = run_pipeline_for_video(video_path, video_output, cfg, logger)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {video_path.name}: {e}")
        time.sleep(1)  # small pause between videos to avoid API rate spikes
    return results
