"""Output formatters for different response formats."""

from .backends.base import TranscriptionResult
from .models import (
    SegmentResponse,
    TranscriptionResponse,
    VerboseTranscriptionResponse,
    WordResponse,
)


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int((secs % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    milliseconds = int((secs % 1) * 1000)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}.{milliseconds:03d}"


def to_json(result: TranscriptionResult) -> TranscriptionResponse:
    """Format as simple JSON response."""
    return TranscriptionResponse(text=result.text)


def to_text(result: TranscriptionResult) -> str:
    """Format as plain text."""
    return result.text


def to_verbose_json(
    result: TranscriptionResult,
    include_words: bool = False,
    include_segments: bool = True,
) -> VerboseTranscriptionResponse:
    """Format as verbose JSON with timestamps."""
    words = None
    if include_words and result.words:
        words = [
            WordResponse(word=w.word, start=w.start, end=w.end) for w in result.words
        ]

    segments = None
    if include_segments and result.segments:
        segments = [
            SegmentResponse(id=s.id, start=s.start, end=s.end, text=s.text)
            for s in result.segments
        ]

    return VerboseTranscriptionResponse(
        language=result.language or "unknown",
        duration=result.duration,
        text=result.text,
        words=words,
        segments=segments,
    )


def to_srt(result: TranscriptionResult) -> str:
    """Format as SRT subtitle format."""
    if not result.segments:
        return ""

    lines = []
    for seg in result.segments:
        lines.append(str(seg.id + 1))
        lines.append(
            f"{format_timestamp_srt(seg.start)} --> {format_timestamp_srt(seg.end)}"
        )
        lines.append(seg.text.strip())
        lines.append("")

    return "\n".join(lines)


def to_vtt(result: TranscriptionResult) -> str:
    """Format as WebVTT subtitle format."""
    if not result.segments:
        return "WEBVTT\n\n"

    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(
            f"{format_timestamp_vtt(seg.start)} --> {format_timestamp_vtt(seg.end)}"
        )
        lines.append(seg.text.strip())
        lines.append("")

    return "\n".join(lines)
