import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
