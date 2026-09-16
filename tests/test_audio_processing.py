import sys
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import ffmpeg


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "douyin-video" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from douyin_downloader import (  # noqa: E402
    DouyinProcessor,
    _audio_mime_type,
    _ffmpeg_user_message,
)


def ffmpeg_error(stderr: str) -> ffmpeg.Error:
    return ffmpeg.Error("ffmpeg", b"", stderr.encode())


class AudioProcessingTests(unittest.TestCase):
    def setUp(self):
        self.processor = DouyinProcessor()
        self.addCleanup(shutil.rmtree, self.processor.temp_dir, True)

    def test_extract_audio_returns_mp3_when_primary_conversion_succeeds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"video")

            def convert(input_path, output_path, **options):
                output_path.write_bytes(b"mp3")

            with patch.object(self.processor, "_convert_audio", side_effect=convert) as mocked:
                audio_path = self.processor.extract_audio(video_path, show_progress=False)

            self.assertEqual(audio_path.suffix, ".mp3")
            self.assertEqual(audio_path.read_bytes(), b"mp3")
            mocked.assert_called_once()

    def test_extract_audio_falls_back_to_wav_when_mp3_encoder_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"video")

            def convert(input_path, output_path, **options):
                if output_path.suffix == ".mp3":
                    raise ffmpeg_error("Unknown encoder 'libmp3lame'")
                output_path.write_bytes(b"wav")

            with patch.object(self.processor, "_convert_audio", side_effect=convert) as mocked:
                audio_path = self.processor.extract_audio(video_path, show_progress=False)

            self.assertEqual(audio_path.suffix, ".wav")
            self.assertEqual(audio_path.read_bytes(), b"wav")
            self.assertEqual(mocked.call_count, 2)

    def test_extract_audio_reports_missing_audio_stream(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            error = ffmpeg_error("Output file #0 does not contain any stream")

            with patch.object(self.processor, "_convert_audio", side_effect=error):
                with self.assertRaisesRegex(ValueError, "没有可用的音轨"):
                    self.processor.extract_audio(video_path, show_progress=False)

    def test_ffmpeg_error_reports_corrupt_video(self):
        error = ffmpeg_error("moov atom not found")

        self.assertIn("视频文件损坏", _ffmpeg_user_message(error))

    def test_audio_mime_type_matches_output_format(self):
        self.assertEqual(_audio_mime_type(Path("audio.mp3")), "audio/mpeg")
        self.assertEqual(_audio_mime_type(Path("audio.wav")), "audio/wav")

    def test_split_audio_preserves_wav_format(self):
        audio_path = self.processor.temp_dir / "audio.wav"
        audio_path.write_bytes(b"wav")

        def convert(input_path, output_path, **options):
            output_path.write_bytes(b"segment")

        with patch.object(
            self.processor,
            "get_audio_info",
            return_value={"duration": 2, "size": 3},
        ), patch.object(self.processor, "_convert_audio", side_effect=convert) as mocked:
            segments = self.processor.split_audio(
                audio_path,
                segment_duration=1,
                show_progress=False,
            )

        self.assertEqual([path.suffix for path in segments], [".wav", ".wav"])
        self.assertEqual(mocked.call_count, 2)

    def test_transcribe_uses_wav_mime_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"wav")
            response = Mock(status_code=200)
            response.json.return_value = {"text": "ok"}
            response.raise_for_status.return_value = None

            with patch("douyin_downloader.requests.post", return_value=response) as post:
                text = self.processor.transcribe_single_audio(audio_path)

            self.assertEqual(text, "ok")
            self.assertEqual(post.call_args.kwargs["files"]["file"][2], "audio/wav")


if __name__ == "__main__":
    unittest.main()
