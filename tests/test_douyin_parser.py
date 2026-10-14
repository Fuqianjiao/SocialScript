import sys
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
    def __init__(self, kwargs):
        self.kwargs = kwargs

    async def fetch_one_video(self, aweme_id):
        return _FakeDetail()


class DouyinParserTests(unittest.TestCase):
    def test_rejects_non_douyin_url(self):
        processor = DouyinProcessor()
        with self.assertRaisesRegex(ValueError, "仅支持"):
            processor._resolve_video_id("https://example.com/video/123")

    def test_signed_parser_maps_video_fields(self):
        processor = DouyinProcessor(douyin_cookie="sessionid=test")
        with patch("f2.apps.douyin.handler.DouyinHandler", _FakeHandler):
            result = processor._parse_with_signature("1234567890123456789")

        self.assertEqual(result["video_id"], "1234567890123456789")
        self.assertEqual(result["title"], "测试_标题")
        self.assertEqual(result["url"], "https://video.example/test.mp4")

    def test_missing_cookie_has_clear_error(self):
        processor = DouyinProcessor()
        with patch.object(processor, "_parse_legacy_page", side_effect=KeyError("videoInfoRes")):
            with self.assertRaisesRegex(ValueError, "配置抖音 Cookie"):
                processor.parse_share_url(
                    "https://www.douyin.com/video/1234567890123456789"
                )


if __name__ == "__main__":
    unittest.main()
