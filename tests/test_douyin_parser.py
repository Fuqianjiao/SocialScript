import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "douyin-video" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from douyin_downloader import DouyinProcessor  # noqa: E402


class _FakeDetail:
    def _to_raw(self):
        return {
            "aweme_detail": {
                "aweme_id": "1234567890123456789",
                "desc": "测试/标题",
                "video": {
                    "bit_rate": [
                        {"play_addr": {"url_list": ["https://video.example/test.mp4"]}}
                    ]
                },
            }
        }


class _FakeHandler:
    last_kwargs = None

    def __init__(self, kwargs):
        self.kwargs = kwargs
        type(self).last_kwargs = kwargs

    async def fetch_one_video(self, aweme_id):
        return _FakeDetail()


class _FakeVideoResponse:
    headers = {"content-length": "4"}

    @staticmethod
    def raise_for_status():
        return None

    @staticmethod
    def iter_content(chunk_size):
        yield b"test"


class DouyinParserTests(unittest.TestCase):
    def test_rejects_non_douyin_url(self):
        processor = DouyinProcessor()
        with self.assertRaisesRegex(ValueError, "仅支持"):
            processor._resolve_video_id("https://example.com/video/123")

    def test_signed_parser_maps_video_fields(self):
        processor = DouyinProcessor(douyin_cookie="sessionid=test")
        with patch(
            "douyin_downloader._load_douyin_handler",
            return_value=_FakeHandler,
        ):
            result = processor._parse_with_signature("1234567890123456789")

        self.assertEqual(result["video_id"], "1234567890123456789")
        self.assertEqual(result["title"], "测试_标题")
        self.assertEqual(result["url"], "https://video.example/test.mp4")
        self.assertEqual(
            _FakeHandler.last_kwargs["headers"]["x-tt-argus"],
            "1",
        )
        self.assertEqual(_FakeHandler.last_kwargs["cookie"], "sessionid=test")

    def test_missing_cookie_has_clear_error(self):
        processor = DouyinProcessor()
        with patch.object(processor, "_parse_legacy_page", side_effect=KeyError("videoInfoRes")):
            with self.assertRaisesRegex(ValueError, "配置抖音 Cookie"):
                processor.parse_share_url(
                    "https://www.douyin.com/video/1234567890123456789"
                )

    def test_download_uses_signed_desktop_headers(self):
        processor = DouyinProcessor()
        video_info = {
            "video_id": "1234567890123456789",
            "title": "测试视频",
            "url": "https://video.example/test.mp4",
        }

        with tempfile.TemporaryDirectory() as output_dir:
            with patch(
                "douyin_downloader.requests.get",
                return_value=_FakeVideoResponse(),
            ) as mock_get:
                output_path = processor.download_video(
                    video_info,
                    Path(output_dir),
                    show_progress=False,
                )

            self.assertEqual(output_path.read_bytes(), b"test")

        request_kwargs = mock_get.call_args.kwargs
        self.assertEqual(request_kwargs["headers"]["x-tt-argus"], "1")
        self.assertIn("Chrome/130", request_kwargs["headers"]["User-Agent"])
        self.assertEqual(request_kwargs["timeout"], 60)


if __name__ == "__main__":
    unittest.main()
