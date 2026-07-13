"""
video_editor.py
================
Turns Gemini's clip candidates into validated, non-overlapping timestamp
ranges and exports them as MP4 files with ffmpeg. Also handles optional
SRT subtitle generation (burned-in or sidecar) and thumbnail extraction.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import ffmpeg

from gemini_client import ClipCandidate
from transcriber import TranscriptSegment
from utils import seconds_to_srt_timestamp, safe_filename


@dataclass
class FinalClip:
    index: int
    title: str
    start: float
    end: float
    score: int
    reason: str
    filename: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


# --------------------------------------------------------------------------- #
# Timestamp validation / cleanup
# --------------------------------------------------------------------------- #

def validate_and_clean_candidates(
    candidates: List[ClipCandidate],
    video_duration: float,
    min_sec: int,
    max_sec: int,
    merge_gap: float,
    max_clips: int,
    logger: Optional[logging.Logger] = None,
) -> List[FinalClip]:
    """
    - Clamp timestamps to valid video range.
    - Drop clips shorter than min_sec.
    - Trim clips longer than max_sec (keep the start, cut the tail).
    - Sort by start time, merge/deduplicate near-identical or overlapping clips.
    - Cap to the top-N clips by score.
    """
    log = logger or logging.getLogger(__name__)
    cleaned: List[ClipCandidate] = []

    for c in candidates:
        start = max(0.0, min(c.start_seconds, video_duration))
        end = max(0.0, min(c.end_seconds, video_duration))
        if end <= start:
            continue
        if end - start > max_sec:
            end = start + max_sec
        if end - start < min_sec:
            log.debug(f"Discarding too-short clip '{c.title}' ({end - start:.1f}s)")
            continue
        cleaned.append(ClipCandidate(c.title, start, end, c.score, c.reason))

    # Sort by start time to make overlap detection linear
    cleaned.sort(key=lambda c: c.start_seconds)

    merged: List[ClipCandidate] = []
    for c in cleaned:
        if not merged:
            merged.append(c)
            continue

        prev = merged[-1]
        gap = c.start_seconds - prev.end_seconds

        overlaps = c.start_seconds < prev.end_seconds
        near_duplicate = overlaps and (
            min(c.end_seconds, prev.end_seconds) - max(c.start_seconds, prev.start_seconds)
        ) > 0.6 * min(c.duration, prev.duration)

        if near_duplicate:
            # Keep whichever has the higher Gemini score
            if c.score > prev.score:
                merged[-1] = c
            continue

        if overlaps or (0 <= gap <= merge_gap):
            # Merge into one continuous clip if the combined length still fits
            new_start = min(prev.start_seconds, c.start_seconds)
            new_end = max(prev.end_seconds, c.end_seconds)
            if new_end - new_start <= max_sec:
                merged[-1] = ClipCandidate(
                    title=prev.title if prev.score >= c.score else c.title,
                    start_seconds=new_start,
                    end_seconds=new_end,
                    score=max(prev.score, c.score),
                    reason=prev.reason if prev.score >= c.score else c.reason,
                )
                continue

        merged.append(c)

    # Rank by score, keep top N, then re-sort chronologically for output ordering
    merged.sort(key=lambda c: c.score, reverse=True)
    top = merged[:max_clips]
    top.sort(key=lambda c: c.start_seconds)

    final_clips = [
        FinalClip(index=i + 1, title=c.title, start=c.start_seconds, end=c.end_seconds,
                   score=c.score, reason=c.reason)
        for i, c in enumerate(top)
    ]
    log.info(f"Validated {len(candidates)} raw candidates -> {len(final_clips)} final clips.")
    return final_clips


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #

def build_output_filename(clip: FinalClip) -> str:
    return f"clip_{clip.index:03d}.mp4"


def export_clip(
    video_path: Path,
    clip: FinalClip,
    output_dir: Path,
    codec: str = "libx264",
    audio_codec: str = "aac",
    crf: int = 18,
    preset: str = "medium",
    target_width: int = 1080,
    target_height: int = 1920,
    keep_source_resolution: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Cut a single clip from the source video using ffmpeg with re-encoding
    (accurate frame-level cuts; stream-copy trimming can drift on non-keyframes).
    Vertical (9:16) crop-to-fill is applied unless keep_source_resolution=True.
    """
    log = logger or logging.getLogger(__name__)
    filename = build_output_filename(clip)
    out_path = output_dir / filename

    stream = ffmpeg.input(str(video_path), ss=clip.start, t=clip.duration)

    video_stream = stream.video
    if not keep_source_resolution:
        # Scale to fill target box, then center-crop to exact target aspect ratio
        video_stream = (
            video_stream
            .filter("scale", target_width, target_height, force_original_aspect_ratio="increase")
            .filter("crop", target_width, target_height)
        )

    audio_stream = stream.audio

    try:
        (
            ffmpeg
            .output(
                video_stream, audio_stream, str(out_path),
                vcodec=codec, acodec=audio_codec,
                crf=crf, preset=preset,
                movflags="faststart",
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        raise RuntimeError(f"FFmpeg failed exporting {filename}: {stderr[-800:]}")

    clip.filename = filename
    log.info(f"Exported {filename} ({clip.duration:.1f}s, score={clip.score})")
    return out_path


def export_all_clips(
    video_path: Path,
    clips: List[FinalClip],
    output_dir: Path,
    max_workers: int = 2,
    **export_kwargs,
) -> List[FinalClip]:
    """Export all clips, optionally in parallel (ffmpeg itself is multi-threaded per-call,
    so keep max_workers modest to avoid CPU contention)."""
    logger = export_kwargs.get("logger")
    successful: List[FinalClip] = []

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(export_clip, video_path, clip, output_dir, **export_kwargs): clip
            for clip in clips
        }
        for future in as_completed(futures):
            clip = futures[future]
            try:
                future.result()
                successful.append(clip)
            except Exception as e:
                if logger:
                    logger.error(f"Failed to export clip {clip.index} ('{clip.title}'): {e}")

    successful.sort(key=lambda c: c.index)
    return successful


# --------------------------------------------------------------------------- #
# Bonus: subtitles (SRT) and thumbnails
# --------------------------------------------------------------------------- #

def generate_srt_for_clip(
    clip: FinalClip,
    transcript_segments: List[TranscriptSegment],
    output_dir: Path,
) -> Path:
    """
    Build a standalone .srt file for a clip, re-timed so subtitle timestamps
    are relative to the clip's own start (0:00) rather than the source video.
    """
    srt_path = output_dir / f"clip_{clip.index:03d}.srt"
    relevant = [
        s for s in transcript_segments
        if s.end > clip.start and s.start < clip.end
    ]

    lines = []
    for i, seg in enumerate(relevant, start=1):
        rel_start = max(0.0, seg.start - clip.start)
        rel_end = max(0.0, min(seg.end, clip.end) - clip.start)
        if rel_end <= rel_start:
            continue
        lines.append(str(i))
        lines.append(f"{seconds_to_srt_timestamp(rel_start)} --> {seconds_to_srt_timestamp(rel_end)}")
        lines.append(seg.text.strip())
        lines.append("")

    srt_path.write_text("\n".join(lines), encoding="utf-8")
    return srt_path


def burn_in_subtitles(clip_video_path: Path, srt_path: Path, output_path: Path,
                       logger: Optional[logging.Logger] = None) -> Path:
    """Re-encode a clip with hardcoded (burned-in) captions using the subtitles filter."""
    log = logger or logging.getLogger(__name__)
    escaped_srt = str(srt_path).replace("\\", "/").replace(":", "\\:")
    try:
        (
            ffmpeg
            .input(str(clip_video_path))
            .output(str(output_path), vf=f"subtitles='{escaped_srt}'", acodec="copy")
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        log.error(f"Burning in subtitles failed: {stderr[-500:]}")
        return clip_video_path
    return output_path


def generate_thumbnail(clip_video_path: Path, output_dir: Path, clip_index: int,
                        at_seconds: float = 0.5) -> Path:
    """Grab a representative frame near the start of the clip as a JPEG thumbnail."""
    thumb_path = output_dir / f"clip_{clip_index:03d}_thumb.jpg"
    try:
        (
            ffmpeg
            .input(str(clip_video_path), ss=at_seconds)
            .output(str(thumb_path), vframes=1, format="image2", vcodec="mjpeg")
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error:
        pass  # thumbnail generation is best-effort, non-fatal
    return thumb_path
