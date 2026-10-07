import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "douyin-video" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from douyin_downloader import DouyinProcessor, _normalize_cookie_header  # noqa: E402


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
    def test_normalizes_multiline_cookie(self):
        value = _normalize_cookie_header(
            "Cookie: sessionid=abc\n"
            "ttwid=xyz; passport_csrf_token=token==;\n"
            "sessionid=new-value"
        )

        self.assertEqual(
            value,
            "sessionid=new-value; ttwid=xyz; passport_csrf_token=token==",
        )

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
            with patch.object(processor, "_parse_with_yt_dlp", side_effect=ValueError("fresh cookies")):
                with self.assertRaisesRegex(ValueError, "有效的 douyin.com Cookie"):
                    processor.parse_share_url(
                        "https://www.douyin.com/video/1234567890123456789"
                    )

    def test_falls_back_to_yt_dlp_after_signature_failure(self):
        processor = DouyinProcessor(douyin_cookie="sessionid=test")
        expected = {
            "video_id": "1234567890123456789",
            "title": "备用解析",
            "url": "https://video.example/fallback.mp4",
        }
        with patch.object(
            processor,
            "_parse_with_signature",
            side_effect=ValueError("Status Code: 403"),
        ):
            with patch.object(processor, "_parse_with_yt_dlp", return_value=expected):
                result = processor.parse_share_url(
                    "https://www.douyin.com/video/1234567890123456789"
                )

        self.assertEqual(result, expected)

    def test_reports_platform_rejection_instead_of_cookie_failure(self):
        processor = DouyinProcessor(douyin_cookie="sessionid=test")
        with patch.object(
            processor,
            "_parse_with_signature",
            side_effect=ValueError("Status Code: 403 Forbidden"),
        ):
            with patch.object(
                processor,
                "_parse_with_yt_dlp",
                side_effect=ValueError("Fresh cookies are needed"),
            ):
                with patch.object(
                    processor,
                    "_parse_legacy_page",
                    side_effect=KeyError("videoInfoRes"),
                ):
                    with self.assertRaisesRegex(ValueError, "HTTP 403"):
                        processor.parse_share_url(
                            "https://www.douyin.com/video/1234567890123456789"
                        )

    def test_writes_cookie_file_without_logging_values(self):
        processor = DouyinProcessor(
            douyin_cookie="sessionid=abc; passport_csrf_token=token=="
        )
        cookie_file = processor._write_yt_dlp_cookie_file()
        content = cookie_file.read_text(encoding="utf-8")

        self.assertIn("\tsessionid\tabc", content)
        self.assertIn("\tpassport_csrf_token\ttoken==", content)
        self.assertEqual(cookie_file.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
