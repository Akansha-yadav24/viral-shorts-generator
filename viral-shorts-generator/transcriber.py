"""
transcriber.py
==============
Handles audio extraction and speech-to-text transcription with word/segment
level timestamps.

Primary engine: faster-whisper (CTranslate2 backend - much faster & lower RAM
than the original openai-whisper). Falls back to openai-whisper automatically
if faster-whisper isn't installed/available.

Produces a list of `TranscriptSegment` objects and can render them into
chunked plain-text blocks (with inline timestamps) suitable for feeding to
Gemini without blowing its context window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import ffmpeg

from utils import seconds_to_timestamp


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class TranscriptionError(Exception):
    pass


def extract_audio(video_path: Path, temp_dir: Path, logger: Optional[logging.Logger] = None) -> Path:
    """
    Extract mono 16kHz WAV audio from the input video using ffmpeg.
    16kHz mono is what Whisper models expect internally, and extracting at
    that rate up-front keeps the temp file small and transcription fast.
    """
    audio_path = temp_dir / f"{video_path.stem}_audio.wav"
    if logger:
        logger.info(f"Extracting audio -> {audio_path.name}")
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path), ac=1, ar=16000, format="wav", vn=None)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else str(e)
        raise TranscriptionError(f"FFmpeg audio extraction failed: {stderr[-800:]}")

    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise TranscriptionError("Audio extraction produced an empty file.")
    return audio_path


def transcribe_audio(
    audio_path: Path,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "auto",
    logger: Optional[logging.Logger] = None,
) -> List[TranscriptSegment]:
    """
    Transcribe audio into timestamped segments.
    Tries faster-whisper first; falls back to openai-whisper if unavailable.
    """
    segments = _transcribe_with_faster_whisper(audio_path, model_size, device, compute_type, logger)
    if segments is not None:
        return segments

    if logger:
        logger.warning("faster-whisper unavailable or failed; falling back to openai-whisper.")
    segments = _transcribe_with_openai_whisper(audio_path, model_size, logger)
    if segments is not None:
        return segments

    raise TranscriptionError(
        "Both faster-whisper and openai-whisper are unavailable. "
        "Install one of them: pip install faster-whisper  (or)  pip install openai-whisper"
    )


def _transcribe_with_faster_whisper(
    audio_path: Path, model_size: str, device: str, compute_type: str,
    logger: Optional[logging.Logger],
) -> Optional[List[TranscriptSegment]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    try:
        if logger:
            logger.info(f"Loading faster-whisper model '{model_size}' (device={device})...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        if logger:
            logger.info("Transcribing (this can take a while for long videos)...")
        raw_segments, info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            vad_filter=True,          # skip long silences automatically
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        segments: List[TranscriptSegment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if text:
                segments.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))

        if logger:
            duration = getattr(info, "duration", None)
            logger.info(f"Transcription complete: {len(segments)} segments"
                        + (f", audio duration {duration:.1f}s" if duration else ""))
        return segments
    except Exception as e:
        if logger:
            logger.error(f"faster-whisper transcription failed: {e}")
        return None


def _transcribe_with_openai_whisper(
    audio_path: Path, model_size: str, logger: Optional[logging.Logger],
) -> Optional[List[TranscriptSegment]]:
    try:
        import whisper
    except ImportError:
        return None

    try:
        # openai-whisper model sizes don't include 'large-v3'-style suffixes as cleanly;
        # normalize common faster-whisper names down to something openai-whisper accepts.
        normalized_size = model_size.replace("-v3", "").replace("-v2", "")
        if logger:
            logger.info(f"Loading openai-whisper model '{normalized_size}'...")
        model = whisper.load_model(normalized_size)

        if logger:
            logger.info("Transcribing with openai-whisper (fallback engine)...")
        result = model.transcribe(str(audio_path), verbose=False)

        segments = [
            TranscriptSegment(start=seg["start"], end=seg["end"], text=seg["text"].strip())
            for seg in result.get("segments", [])
            if seg["text"].strip()
        ]
        if logger:
            logger.info(f"Transcription complete: {len(segments)} segments (openai-whisper).")
        return segments
    except Exception as e:
        if logger:
            logger.error(f"openai-whisper transcription failed: {e}")
        return None


# --------------------------------------------------------------------------- #
# Chunking for LLM context limits
# --------------------------------------------------------------------------- #

def segments_to_timestamped_text(segments: List[TranscriptSegment]) -> str:
    """Render segments as '[HH:MM:SS.mmm -> HH:MM:SS.mmm] text' lines."""
    lines = []
    for seg in segments:
        start_ts = seconds_to_timestamp(seg.start)
        end_ts = seconds_to_timestamp(seg.end)
        lines.append(f"[{start_ts} -> {end_ts}] {seg.text}")
    return "\n".join(lines)


def chunk_transcript(segments: List[TranscriptSegment], max_chars: int = 12000) -> List[str]:
    """
    Split the timestamped transcript into text chunks under `max_chars`,
    always breaking on segment boundaries (never mid-sentence) so each
    chunk stays self-contained for Gemini.
    """
    if not segments:
        return []

    chunks: List[str] = []
    current_lines: List[str] = []
    current_len = 0

    for seg in segments:
        line = f"[{seconds_to_timestamp(seg.start)} -> {seconds_to_timestamp(seg.end)}] {seg.text}"
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = line_len
        else:
            current_lines.append(line)
            current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks
