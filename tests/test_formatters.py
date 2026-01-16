import unittest

try:
    import pydantic  # noqa: F401
except Exception:
    raise unittest.SkipTest("pydantic not installed")

from open_asr_server.backends.base import Segment, TranscriptionResult, WordSegment
from open_asr_server.formatters import (
    to_json,
    to_srt,
    to_text,
    to_verbose_json,
    to_vtt,
)


class FormatterTests(unittest.TestCase):
    def test_basic_formats(self):
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=1.0,
            words=[
                WordSegment(word="Hello", start=0.0, end=0.5),
                WordSegment(word="world", start=0.5, end=1.0),
            ],
            segments=[
                Segment(id=0, start=0.0, end=1.0, text="Hello world"),
            ],
        )

        self.assertEqual(to_text(result), "Hello world")
        self.assertEqual(to_json(result).text, "Hello world")

        srt = to_srt(result)
        self.assertIn("00:00:00,000 --> 00:00:01,000", srt)

        vtt = to_vtt(result)
        self.assertTrue(vtt.startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:01.000", vtt)

        verbose = to_verbose_json(result, include_words=True, include_segments=True)
        self.assertEqual(verbose.language, "en")
        self.assertEqual(len(verbose.words), 2)
        self.assertEqual(len(verbose.segments), 1)

    def test_verbose_unknown_language(self):
        result = TranscriptionResult(text="Hi", language=None, duration=0.0)
        verbose = to_verbose_json(result)
        self.assertEqual(verbose.language, "unknown")
