"""
gemini_client.py
=================
Wraps the FREE Google Gemini API (google-generativeai) to analyze a video
transcript and return the best short-clip candidates as structured JSON.

Design notes:
  - Uses response_mime_type="application/json" where supported so Gemini
    returns clean JSON without needing to fight it via prompting alone.
  - Still runs output through utils.extract_json_array as a safety net.
  - Supports multi-chunk transcripts (long videos): each chunk is analyzed
    independently, then results are merged & re-scored for global ranking.
  - Retries with exponential backoff on transient errors/timeouts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from utils import extract_json_array, timestamp_to_seconds


SYSTEM_PROMPT = """You are an expert viral short-form video editor and content strategist \
(YouTube Shorts, Instagram Reels, TikTok). You analyze timestamped video transcripts and \
identify the strongest possible short clips.

Return ONLY a JSON array. No markdown, no code fences, no explanations, no preamble.

Each element must have exactly these fields:
- "title": short catchy title (max 60 chars)
- "start": timestamp string "HH:MM:SS" (use the transcript's own timestamps)
- "end": timestamp string "HH:MM:SS"
- "score": integer 0-100, how viral/engaging this clip is
- "reason": one sentence explaining why this clip works

Rules for clip selection:
- Every clip must be between {min_sec} and {max_sec} seconds long.
- Each clip must start with a hook and contain exactly one complete idea or thought.
- Never cut a clip in the middle of a sentence or a spoken thought.
- Do not include intros, outros, sponsor segments, filler, or long silences.
- Do not produce duplicate or near-duplicate clips covering the same moment.
- Prioritize: strong hooks, mind-blowing facts, problem->solution moments, actionable tips, \
mistakes to avoid, coding tricks or technical explanations, funny moments, storytelling, \
emotional moments, surprising facts, and high-retention explanations.
- Only select timestamps that actually exist in the given transcript excerpt.
- If nothing in this excerpt is strong enough, return an empty array [].
"""

USER_PROMPT_TEMPLATE = """Analyze the following timestamped video transcript excerpt and return \
the best short-clip candidates as a JSON array following the rules above.

TRANSCRIPT EXCERPT:
---
{transcript}
---

Return ONLY the JSON array now.
"""


@dataclass
class ClipCandidate:
    title: str
    start_seconds: float
    end_seconds: float
    score: int
    reason: str

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


class GeminiClientError(Exception):
    pass


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        max_retries: int = 3,
        timeout_seconds: int = 120,
        logger: Optional[logging.Logger] = None,
    ):
        if not api_key:
            raise GeminiClientError("Missing Gemini API key.")

        try:
            import google.generativeai as genai
        except ImportError as e:
            raise GeminiClientError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            ) from e

        self._genai = genai
        self._genai.configure(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.logger = logger or logging.getLogger(__name__)

        self.model = self._genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT.format(min_sec=20, max_sec=90),
        )

    def analyze_chunk(self, transcript_chunk: str, min_sec: int = 20, max_sec: int = 90) -> List[ClipCandidate]:
        """Send a single transcript chunk to Gemini and parse the clip candidates."""
        prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript_chunk)

        generation_config = {
            "temperature": 0.4,
            "response_mime_type": "application/json",
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.debug(f"Gemini request attempt {attempt}/{self.max_retries}...")
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    request_options={"timeout": self.timeout_seconds},
                )
                raw_text = self._extract_text(response)
                data = extract_json_array(raw_text)
                return self._parse_candidates(data, min_sec, max_sec)

            except Exception as e:
                last_error = e
                wait = min(2 ** attempt, 20)
                self.logger.warning(f"Gemini call failed (attempt {attempt}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        raise GeminiClientError(f"Gemini analysis failed after {self.max_retries} attempts: {last_error}")

    def analyze_transcript_chunks(
        self, chunks: List[str], min_sec: int = 20, max_sec: int = 90
    ) -> List[ClipCandidate]:
        """Analyze every chunk of a (possibly multi-part) transcript and merge results."""
        all_candidates: List[ClipCandidate] = []
        for i, chunk in enumerate(chunks, start=1):
            self.logger.info(f"Analyzing transcript chunk {i}/{len(chunks)} with Gemini...")
            try:
                candidates = self.analyze_chunk(chunk, min_sec, max_sec)
                self.logger.info(f"  -> {len(candidates)} candidate clip(s) found in chunk {i}.")
                all_candidates.extend(candidates)
            except GeminiClientError as e:
                self.logger.error(f"Chunk {i} analysis failed, skipping: {e}")
        return all_candidates

    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(response) -> str:
        text = getattr(response, "text", None)
        if text:
            return text
        # Fallback: manually walk candidate parts if .text accessor is unavailable
        try:
            return response.candidates[0].content.parts[0].text
        except (AttributeError, IndexError) as e:
            raise GeminiClientError(f"Could not extract text from Gemini response: {e}")

    @staticmethod
    def _parse_candidates(data, min_sec: int, max_sec: int) -> List[ClipCandidate]:
        candidates: List[ClipCandidate] = []
        for item in data:
            try:
                start_s = timestamp_to_seconds(str(item["start"]))
                end_s = timestamp_to_seconds(str(item["end"]))
                if end_s <= start_s:
                    continue
                candidates.append(
                    ClipCandidate(
                        title=str(item.get("title", "Untitled Clip"))[:80],
                        start_seconds=start_s,
                        end_seconds=end_s,
                        score=int(item.get("score", 50)),
                        reason=str(item.get("reason", "")),
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue  # skip malformed entries rather than failing the whole batch
        return candidates
