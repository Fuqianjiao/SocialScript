import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "douyin-video" / "scripts"
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(WEB_DIR))

from xiaohongshu_downloader import XiaohongshuProcessor  # noqa: E402


class XiaohongshuParserTests(unittest.TestCase):
    def test_initial_state_accepts_xiaohongshu_javascript_collections(self):
        page = (
            '<script>window.__INITIAL_STATE__={"text":"&quot;",'
            '"map":new Map([]),"set":new Set([]),"missing":undefined};</script>'
        )

        state = XiaohongshuProcessor._initial_state(page)

        self.assertEqual(state["text"], "&quot;")
        self.assertEqual(state["map"], {})
        self.assertEqual(state["set"], [])
        self.assertIsNone(state["missing"])

    def test_pick_video_url_accepts_ef_stream_groups(self):
        note = {
            "type": "video",
            "video": {
                "media": {
                    "stream": {
                        "EF4": [
                            {"masterUrl": "https://example.com/low.mp4"},
                            {
                                "defaultStream": True,
                                "masterUrl": "https://example.com/default.mp4",
                            },
                        ]
                    }
                }
            },
        }

        url = XiaohongshuProcessor._pick_video_url(note)

        self.assertEqual(url, "https://example.com/default.mp4")

    def test_download_video_resumes_after_connection_break(self):
        first = Mock()
        first.status_code = 200
        first.headers = {"content-type": "video/mp4", "content-length": "6"}
        first.__enter__ = Mock(return_value=first)
        first.__exit__ = Mock(return_value=False)
        first.raise_for_status.return_value = None

        def interrupted_chunks(chunk_size):
            yield b"abc"
            raise requests.exceptions.ChunkedEncodingError("connection broken")

        first.iter_content.side_effect = interrupted_chunks

        second = Mock()
        second.status_code = 206
        second.headers = {
            "content-type": "video/mp4",
            "content-length": "3",
            "content-range": "bytes 3-5/6",
        }
        second.__enter__ = Mock(return_value=second)
        second.__exit__ = Mock(return_value=False)
        second.raise_for_status.return_value = None
        second.iter_content.return_value = [b"def"]

        processor = XiaohongshuProcessor(xiaohongshu_cookie="a=b")
        with tempfile.TemporaryDirectory() as temp_dir, \
                patch("xiaohongshu_downloader.requests.get", side_effect=[first, second]) as get, \
                patch("xiaohongshu_downloader.time.sleep"):
            path = processor.download_video(
                {"url": "https://example.com/video.mp4", "video_id": "note"},
                Path(temp_dir),
                show_progress=False,
            )

            self.assertEqual(path.read_bytes(), b"abcdef")
            self.assertEqual(get.call_count, 2)
            self.assertEqual(get.call_args_list[1].kwargs["headers"]["Range"], "bytes=3-")


if __name__ == "__main__":
    unittest.main()
