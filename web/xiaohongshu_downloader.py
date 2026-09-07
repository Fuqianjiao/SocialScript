"""小红书视频解析与文案提取。"""

import json
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from douyin_downloader import DouyinProcessor


XHS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.xiaohongshu.com/",
}


def _normalize_cookie(cookie: str) -> str:
    value = (cookie or os.getenv("XIAOHONGSHU_COOKIE", "")).strip()
    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()
    return " ".join(value.splitlines()).strip()


class XiaohongshuProcessor(DouyinProcessor):
    """复用下载后处理和 ASR，只替换小红书解析及下载请求头。"""

    def __init__(self, api_key: str = "", model: str | None = None,
                 xiaohongshu_cookie: str = ""):
        super().__init__(api_key=api_key, model=model)
        self.xiaohongshu_cookie = _normalize_cookie(xiaohongshu_cookie)

    def _resolve_note(self, share_text: str) -> tuple[str, str]:
        urls = re.findall(r"https?://[^\s]+", share_text)
        if not urls:
            raise ValueError("未找到有效的小红书分享链接")
        share_url = urls[0].rstrip("，。；、)）]】")
        host = urlparse(share_url).hostname or ""
        if not (host == "xiaohongshu.com" or host.endswith(".xiaohongshu.com")
                or host in {"xhslink.com", "xhslink.cn"}):
            raise ValueError("仅支持 xiaohongshu.com、xhslink.com 或 xhslink.cn 链接")

        headers = dict(XHS_HEADERS)
        if self.xiaohongshu_cookie:
            headers["Cookie"] = self.xiaohongshu_cookie
        response = requests.get(share_url, headers=headers, allow_redirects=True, timeout=20)
        response.raise_for_status()

        parsed = urlparse(response.url)
        query = parse_qs(parsed.query)
        path_match = re.search(r"/(?:explore|discovery/item)/([0-9a-f]+)", parsed.path)
        note_id = path_match.group(1) if path_match else (query.get("target_note_id") or [""])[0]
        if not note_id:
            raise ValueError("没有从小红书分享链接中解析到笔记 ID")

        params = {
            key: query[key][0]
            for key in ("xsec_token", "xsec_source")
            if query.get(key)
        }
        detail_url = f"https://www.xiaohongshu.com/explore/{note_id}"
        if params:
            detail_url += "?" + urlencode(params)
        return note_id, detail_url

    @staticmethod
    def _initial_state(page_html: str) -> dict:
        match = re.search(
            r"<script>window\.__INITIAL_STATE__=(.*?)</script>",
            page_html,
            flags=re.DOTALL,
        )
        if not match:
            raise ValueError("小红书页面没有返回笔记数据，可能需要更新 Cookie")
        # script 元素属于 raw-text，里面的 HTML 实体不能做 html.unescape。
        # 笔记正文可能包含 &quot; 等文本，反解码后会生成未转义引号，
        # 导致原本合法的 JSON 在正文位置解析失败。
        payload = match.group(1).strip().removesuffix(";").rstrip()
        # 小红书部分 store 会把空集合直接序列化成 JavaScript 表达式。
        # 这些字段与笔记解析无关，但会令标准 JSON 解析器失败。
        payload = re.sub(r"new\s+Map\s*\(\s*\[\s*\]\s*\)", "{}", payload)
        payload = re.sub(r"new\s+Set\s*\(\s*\[\s*\]\s*\)", "[]", payload)
        payload = re.sub(r'(?<!["\w])undefined(?!["\w])', "null", payload)
        return json.loads(payload)

    @staticmethod
    def _pick_video_url(note: dict) -> str:
        stream = (
            note.get("video", {})
            .get("media", {})
            .get("stream", {})
        )
        preferred_groups = ["h264", "h265", "av1"]
        stream_groups = preferred_groups + [
            key for key in stream if key not in preferred_groups
        ]
        for codec in stream_groups:
            items = stream.get(codec) or []
            if not isinstance(items, list):
                continue
            items = sorted(
                items,
                key=lambda item: bool((item or {}).get("defaultStream")),
                reverse=True,
            )
            for item in items:
                if item.get("masterUrl"):
                    return item["masterUrl"]
                backups = item.get("backupUrls") or []
                if backups:
                    return backups[0]
        return ""

    def parse_share_url(self, share_text: str) -> dict:
        if not self.xiaohongshu_cookie:
            raise ValueError("请先在解析配置中填写小红书 Cookie")

        note_id, detail_url = self._resolve_note(share_text)
        headers = {**XHS_HEADERS, "Cookie": self.xiaohongshu_cookie}
        response = requests.get(detail_url, headers=headers, timeout=20)
        response.raise_for_status()
        state = self._initial_state(response.text)

        note_map = state.get("note", {}).get("noteDetailMap", {})
        note = (note_map.get(note_id) or {}).get("note") or {}
        if not note:
            server_info = state.get("note", {}).get("serverRequestInfo", {})
            message = server_info.get("errMsg") or "登录态无效或该笔记暂时无法查看"
            raise ValueError(f"小红书笔记解析失败：{message}")

        video_url = self._pick_video_url(note)
        if not video_url:
            if note.get("type") != "video" and note.get("imageList"):
                raise ValueError("该内容是图文笔记，暂时没有可转写的视频")
            raise ValueError("笔记中没有找到可用的视频地址")

        title = (note.get("title") or note.get("desc") or f"xiaohongshu_{note_id}").strip()
        title = re.sub(r'[\\/:*?"<>|]', "_", title)
        return {
            "url": video_url,
            "title": title,
            "video_id": note_id,
            "platform": "xiaohongshu",
        }

    def download_video(self, video_info: dict, output_dir: Path | None = None,
                       show_progress: bool = True) -> Path:
        output_dir = Path(output_dir) if output_dir else self.temp_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{video_info['video_id']}.mp4"
        headers = {**XHS_HEADERS, "Cookie": self.xiaohongshu_cookie}
        with requests.get(
            video_info["url"], headers=headers, stream=True,
            allow_redirects=True, timeout=120
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                raise ValueError("小红书视频地址已过期，请重新解析后再试")
            with open(filepath, "wb") as output:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        output.write(chunk)
        if not filepath.exists() or filepath.stat().st_size == 0:
            raise ValueError("小红书视频下载结果为空")
        return filepath


def get_xiaohongshu_video_info(share_link: str, xiaohongshu_cookie: str) -> dict:
    return XiaohongshuProcessor(xiaohongshu_cookie=xiaohongshu_cookie).parse_share_url(share_link)


def extract_xiaohongshu_text(share_link: str, api_key: str, model: str,
                             xiaohongshu_cookie: str) -> dict:
    processor = XiaohongshuProcessor(
        api_key=api_key,
        model=model,
        xiaohongshu_cookie=xiaohongshu_cookie,
    )
    video_info = processor.parse_share_url(share_link)
    video_path = processor.download_video(video_info, show_progress=False)
    audio_path = processor.extract_audio(video_path, show_progress=False)
    text = processor.extract_text_from_audio(audio_path, show_progress=False)
    return {"video_info": video_info, "text": text, "output_path": None}
