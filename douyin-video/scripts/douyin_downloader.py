#!/usr/bin/env python3
"""
抖音无水印视频下载和文案提取工具

功能:
1. 从抖音分享链接获取无水印视频下载链接
2. 下载视频并提取音频
3. 使用硅基流动 API 从音频中提取文本
4. 自动保存文案到文件 (一个视频一个文件夹)

环境变量:
- API_KEY: 硅基流动 API 密钥 (用于文案提取功能)

使用示例:
  # 获取下载链接 (无需 API 密钥)
  python douyin_downloader.py --link "抖音分享链接" --action info

  # 下载视频
  python douyin_downloader.py --link "抖音分享链接" --action download --output ./videos

  # 提取文案并保存到文件 (需要 API_KEY 环境变量)
  python douyin_downloader.py --link "抖音分享链接" --action extract --output ./output
"""

import os
import re
import sys
import json
import argparse
import tempfile
import shutil
import asyncio
import time
from pathlib import Path
from typing import Optional
from datetime import datetime


def check_dependencies():
    """检查必要的依赖是否已安装"""
    missing = []
    try:
        import requests
    except ImportError:
        missing.append("requests")
    try:
        import ffmpeg
    except ImportError:
        missing.append("ffmpeg-python")

    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        sys.exit(1)


check_dependencies()

import requests
import ffmpeg

# 请求头，模拟移动端访问
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/121.0.2277.107 Version/17.0 Mobile/15E148 Safari/604.1'
}

SIGNED_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Referer': 'https://www.douyin.com/',
}


def _normalize_cookie_header(cookie: str) -> str:
    """兼容浏览器复制的单行或逐行 Cookie，并输出标准请求头格式。"""
    value = (cookie or "").strip()
    if value.lower().startswith("cookie:"):
        value = value.split(":", 1)[1].strip()
    if not value:
        return ""

    pairs = {}
    for line in value.splitlines():
        line = line.strip().rstrip(";")
        if not line:
            continue
        if line.lower().startswith("cookie:"):
            line = line.split(":", 1)[1].strip()
        for part in line.split(";"):
            part = part.strip()
            if "=" not in part:
                continue
            name, item_value = part.split("=", 1)
            name = name.strip()
            if name:
                pairs[name] = item_value.strip()
    return "; ".join(f"{name}={item_value}" for name, item_value in pairs.items())


def _exception_chain_text(exc: Exception) -> str:
    """收集异常链文本，用于区分平台拒绝、登录失效和网络错误。"""
    messages = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(messages)


def _douyin_parse_error(primary_error: Exception, fallback_error: Exception) -> str:
    """把解析器底层异常转换为不误导用户的中文提示。"""
    details = _exception_chain_text(primary_error) + " | " + _exception_chain_text(fallback_error)
    lowered = details.lower()
    if "403" in lowered or "forbidden" in lowered:
        return (
            "抖音接口拒绝了当前解析请求（HTTP 403）。Cookie 可能仍然有效，"
            "常见原因是云端出口 IP、请求签名或账号环境触发风控；建议稍后重试或改用本地服务。"
        )
    if "401" in lowered or "unauthorized" in lowered:
        return "抖音登录状态已失效（HTTP 401），请重新复制 douyin.com 的完整 Cookie 后重试"
    if "fresh cookies" in lowered:
        return "抖音要求更新访问凭证，请先在浏览器打开该作品，再重新复制 douyin.com 的完整 Cookie"
    if "timeout" in lowered or "timed out" in lowered:
        return "访问抖音接口超时，请检查网络后重试"
    if any(marker in lowered for marker in ("private", "not found", "unavailable", "deleted")):
        return "该抖音作品可能已删除、设为私密或仅部分用户可见"
    return "抖音解析失败，签名解析和备用解析均未取得可用的视频地址"


def _ffmpeg_executable() -> str:
    """优先使用系统 ffmpeg，Vercel 等环境回退到 wheel 内置版本。"""
    configured = os.getenv("FFMPEG_BINARY", "").strip()
    if configured:
        return configured
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("未找到 ffmpeg，请安装 ffmpeg 或 imageio-ffmpeg") from exc

# 硅基流动 API 配置
DEFAULT_API_BASE_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
DEFAULT_MODEL = "TeleAI/TeleSpeechASR"

AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
}


def _ffmpeg_stderr(exc: Exception) -> str:
    """读取 ffmpeg-python 隐藏在异常对象中的 stderr。"""
    stderr = getattr(exc, "stderr", b"") or b""
    if isinstance(stderr, bytes):
        return stderr.decode("utf-8", errors="replace").strip()
    return str(stderr).strip()


def _ffmpeg_user_message(exc: Exception) -> str:
    """将常见 FFmpeg 失败转换为适合前端展示的提示。"""
    stderr = _ffmpeg_stderr(exc)
    lowered = stderr.lower()
    if any(marker in lowered for marker in (
        "does not contain any stream",
        "matches no streams",
        "output file #0 does not contain any stream",
    )):
        return "视频中没有可用的音轨，无法提取口播"
    if any(marker in lowered for marker in (
        "invalid data found when processing input",
        "moov atom not found",
        "error opening input",
    )):
        return "下载的视频文件损坏、未下载完整或不是有效的视频文件"
    if any(marker in lowered for marker in (
        "unknown encoder 'libmp3lame'",
        "encoder (codec mp3) not found",
    )):
        return "当前运行环境缺少 MP3 编码器"
    if "permission denied" in lowered:
        return "FFmpeg 无法执行或没有文件读写权限"

    details = [line.strip() for line in stderr.splitlines() if line.strip()]
    if details:
        return f"FFmpeg 处理失败：{details[-1][:500]}"
    return f"FFmpeg 处理失败：{exc}"


def _audio_mime_type(audio_path: Path) -> str:
    return AUDIO_MIME_TYPES.get(audio_path.suffix.lower(), "application/octet-stream")


class DouyinProcessor:
    """抖音视频处理器"""

    def __init__(self, api_key: str = "", api_base_url: Optional[str] = None,
                 model: Optional[str] = None, douyin_cookie: str = ""):
        self.api_key = api_key
        self.api_base_url = api_base_url or DEFAULT_API_BASE_URL
        self.model = model or DEFAULT_MODEL
        cookie = douyin_cookie or os.getenv("DOUYIN_COOKIE", "")
        self.douyin_cookie = _normalize_cookie_header(cookie)
        self.temp_dir = Path(tempfile.mkdtemp())

    def __del__(self):
        """清理临时目录"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _resolve_video_id(self, share_text: str) -> tuple[str, str]:
        """从分享文本或跳转地址中提取视频 ID。"""
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', share_text)
        if not urls:
            raise ValueError("未找到有效的分享链接")

        share_url = urls[0]
        if not re.match(
            r'^https?://(?:[a-z0-9-]+\.)?(?:douyin\.com|iesdouyin\.com)(?:[/:?#]|$)',
            share_url,
            flags=re.IGNORECASE,
        ):
            raise ValueError("仅支持 douyin.com、v.douyin.com 或 iesdouyin.com 的分享链接")
        direct_match = re.search(r'/(?:video|note)/(\d+)', share_url)
        if direct_match:
            return direct_match.group(1), share_url

        headers = dict(HEADERS)
        if self.douyin_cookie:
            headers['Cookie'] = self.douyin_cookie
        try:
            share_response = requests.get(
                share_url, headers=headers, allow_redirects=True, timeout=20
            )
            share_response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"抖音分享链接访问失败：{exc}") from exc

        match = re.search(r'/(?:video|note)/(\d+)', share_response.url)
        if not match:
            match = re.search(r'/(?:video|note)/(\d+)', share_response.text)
        if not match:
            raise ValueError("分享链接已失效或被抖音拦截，请在抖音中重新复制分享链接后再试")
        return match.group(1), share_response.url

    def _parse_with_signature(self, video_id: str) -> dict:
        """使用 f2 生成 msToken/X-Bogus，并携带用户 Cookie 请求作品数据。"""
        if not self.douyin_cookie:
            raise ValueError("抖音已要求登录验证，请先在设置中配置抖音 Cookie")

        try:
            from f2.apps.douyin.handler import DouyinHandler
        except ImportError as exc:
            raise RuntimeError("缺少 f2 解析依赖，请重新执行 uv sync") from exc

        async def fetch():
            kwargs = {
                'headers': {
                    **SIGNED_HEADERS,
                    'Referer': f'https://www.douyin.com/video/{video_id}',
                },
                'proxies': {'http://': None, 'https://': None},
                'cookie': self.douyin_cookie,
                'timeout': 20,
                'max_retries': 1,
            }
            return await DouyinHandler(kwargs).fetch_one_video(aweme_id=video_id)

        try:
            detail = asyncio.run(fetch())
            raw = detail._to_raw() or {}
        except Exception as exc:
            raise ValueError(
                "抖音登录验证失败，请更新 Cookie 后重试；若刚更新，请确认 Cookie 来自 douyin.com"
            ) from exc

        aweme = raw.get('aweme_detail') or {}
        video = aweme.get('video') or {}
        bit_rates = video.get('bit_rate') or []
        url_list = []
        if bit_rates:
            url_list = (bit_rates[0].get('play_addr') or {}).get('url_list') or []
        if not url_list:
            url_list = (video.get('play_addr') or {}).get('url_list') or []
        if not url_list:
            raise ValueError("作品数据中没有可用的视频地址，可能是图文作品、私密作品或已删除作品")

        title = (aweme.get('desc') or f'douyin_{video_id}').strip()
        title = re.sub(r'[\\/:*?"<>|]', '_', title)
        return {'url': url_list[0], 'title': title, 'video_id': video_id}

    def _write_yt_dlp_cookie_file(self) -> Path:
        """将请求头 Cookie 转成 yt-dlp 使用的 Netscape 临时文件。"""
        cookie_file = self.temp_dir / "douyin-cookies.txt"
        lines = ["# Netscape HTTP Cookie File"]
        for pair in self.douyin_cookie.split("; "):
            if "=" not in pair:
                continue
            name, value = pair.split("=", 1)
            lines.append(f".douyin.com\tTRUE\t/\tTRUE\t0\t{name}\t{value}")
        cookie_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cookie_file.chmod(0o600)
        return cookie_file

    def _parse_with_yt_dlp(self, video_id: str) -> dict:
        """使用 yt-dlp 作为 f2 被风控时的备用解析器。"""
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise RuntimeError("缺少 yt-dlp 备用解析依赖，请重新执行 uv sync") from exc

        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "http_headers": SIGNED_HEADERS,
        }
        if self.douyin_cookie:
            options["cookiefile"] = str(self._write_yt_dlp_cookie_file())

        page_url = f"https://www.douyin.com/video/{video_id}"
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(page_url, download=False)
        if not info:
            raise ValueError("yt-dlp 没有返回作品数据")

        video_url = info.get("url")
        if not video_url:
            candidates = [
                item for item in info.get("formats") or []
                if item.get("url") and item.get("vcodec") != "none"
            ]
            if candidates:
                candidates.sort(
                    key=lambda item: (
                        item.get("height") or 0,
                        item.get("tbr") or 0,
                    ),
                    reverse=True,
                )
                video_url = candidates[0]["url"]
        if not video_url:
            raise ValueError("yt-dlp 没有返回可用的视频地址")

        title = (info.get("title") or info.get("description") or f"douyin_{video_id}").strip()
        title = re.sub(r'[\\/:*?"<>|]', '_', title)
        return {"url": video_url, "title": title, "video_id": video_id}

    def _parse_legacy_page(self, video_id: str) -> dict:
        """兼容仍包含 videoInfoRes 的旧版抖音分享页。"""
        share_url = f'https://www.iesdouyin.com/share/video/{video_id}'

        # 获取视频页面内容
        response = requests.get(share_url, headers=HEADERS, timeout=20)
        response.raise_for_status()

        pattern = re.compile(
            pattern=r"window\._ROUTER_DATA\s*=\s*(.*?)</script>",
            flags=re.DOTALL,
        )
        find_res = pattern.search(response.text)

        if not find_res or not find_res.group(1):
            raise ValueError("从HTML中解析视频信息失败")

        # 解析JSON数据
        json_data = json.loads(find_res.group(1).strip())
        VIDEO_ID_PAGE_KEY = "video_(id)/page"
        NOTE_ID_PAGE_KEY = "note_(id)/page"

        loader_data = json_data.get("loaderData") or {}
        if VIDEO_ID_PAGE_KEY in loader_data:
            original_video_info = loader_data[VIDEO_ID_PAGE_KEY].get("videoInfoRes")
        elif NOTE_ID_PAGE_KEY in loader_data:
            original_video_info = loader_data[NOTE_ID_PAGE_KEY].get("videoInfoRes")
        else:
            original_video_info = None

        if not original_video_info:
            raise ValueError("抖音页面已不再返回 videoInfoRes")

        item_list = original_video_info.get("item_list") or []
        if not item_list:
            raise ValueError("抖音页面没有返回作品数据")
        data = item_list[0]

        # 获取视频信息
        video_url = data["video"]["play_addr"]["url_list"][0].replace("playwm", "play")
        desc = data.get("desc", "").strip() or f"douyin_{video_id}"

        # 替换文件名中的非法字符
        desc = re.sub(r'[\\/:*?"<>|]', '_', desc)

        return {
            "url": video_url,
            "title": desc,
            "video_id": video_id
        }

    def parse_share_url(self, share_text: str) -> dict:
        """从分享文本中提取无水印视频链接。"""
        video_id, _ = self._resolve_video_id(share_text)

        if self.douyin_cookie:
            try:
                return self._parse_with_signature(video_id)
            except Exception as signature_error:
                try:
                    return self._parse_with_yt_dlp(video_id)
                except Exception as fallback_error:
                    try:
                        return self._parse_legacy_page(video_id)
                    except Exception:
                        raise ValueError(
                            _douyin_parse_error(signature_error, fallback_error)
                        ) from fallback_error

        try:
            return self._parse_legacy_page(video_id)
        except Exception as legacy_error:
            try:
                return self._parse_with_yt_dlp(video_id)
            except Exception as fallback_error:
                raise ValueError(
                    "抖音页面结构已更新，需要有效的 douyin.com Cookie；"
                    "请先在浏览器打开该作品，再复制完整 Cookie 后重试"
                ) from fallback_error

    def download_video(self, video_info: dict, output_dir: Optional[Path] = None, show_progress: bool = True) -> Path:
        """下载视频"""
        if output_dir is None:
            output_dir = self.temp_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{video_info['video_id']}.mp4"
        filepath = output_dir / filename

        if show_progress:
            print(f"正在下载视频: {video_info['title']}")

        response = requests.get(video_info['url'], headers=HEADERS, stream=True)
        response.raise_for_status()

        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))

        # 下载文件
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total_size > 0:
                        progress = downloaded / total_size * 100
                        print(f"\r下载进度: {progress:.1f}%", end="", flush=True)

        if show_progress:
            print(f"\n视频下载完成: {filepath}")
        return filepath

    @staticmethod
    def _audio_output_options(output_path: Path) -> dict:
        if output_path.suffix.lower() == ".wav":
            return {"acodec": "pcm_s16le", "ac": 1, "ar": 16000}
        return {"acodec": "libmp3lame", "q": 0}

    def _convert_audio(self, input_path: Path, output_path: Path, **input_options) -> None:
        stream = ffmpeg.input(str(input_path), **input_options)
        (
            stream
            .output(str(output_path), **self._audio_output_options(output_path))
            .run(
                cmd=_ffmpeg_executable(),
                capture_stdout=True,
                capture_stderr=True,
                overwrite_output=True,
            )
        )

    @staticmethod
    def _validate_audio_output(audio_path: Path) -> None:
        if not audio_path.exists() or audio_path.stat().st_size == 0:
            raise RuntimeError("FFmpeg 未生成有效的音频文件")

    def extract_audio(self, video_path: Path, show_progress: bool = True) -> Path:
        """从视频文件中提取音频"""
        if not video_path.exists() or video_path.stat().st_size == 0:
            raise ValueError("下载的视频文件为空，无法提取音频")

        mp3_path = video_path.with_suffix('.mp3')

        if show_progress:
            print("正在提取音频...")
        try:
            self._convert_audio(video_path, mp3_path)
            self._validate_audio_output(mp3_path)
            if show_progress:
                print(f"音频提取完成: {mp3_path}")
            return mp3_path
        except ffmpeg.Error as mp3_error:
            message = _ffmpeg_user_message(mp3_error)
            if message.startswith(("视频中没有", "下载的视频文件损坏")):
                raise ValueError(f"提取音频失败：{message}") from mp3_error

            mp3_path.unlink(missing_ok=True)
            wav_path = video_path.with_suffix('.wav')
            try:
                self._convert_audio(video_path, wav_path)
                self._validate_audio_output(wav_path)
                if show_progress:
                    print(f"MP3 编码不可用，已回退为 WAV: {wav_path}")
                return wav_path
            except Exception as wav_error:
                wav_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"提取音频失败：{_ffmpeg_user_message(wav_error)}；"
                    f"MP3 首次失败原因：{message}"
                ) from wav_error
        except Exception as exc:
            raise RuntimeError(f"提取音频失败：{_ffmpeg_user_message(exc)}") from exc

    def get_audio_info(self, audio_path: Path) -> dict:
        """获取音频文件信息（时长和大小）"""
        try:
            probe = ffmpeg.probe(str(audio_path))
            duration = float(probe['format'].get('duration', 0))
            size = audio_path.stat().st_size
            return {'duration': duration, 'size': size}
        except Exception:
            return {'duration': 0, 'size': audio_path.stat().st_size}

    def split_audio(self, audio_path: Path, segment_duration: int = 600, show_progress: bool = True) -> list:
        """
        将音频分割成多个片段

        参数:
            audio_path: 音频文件路径
            segment_duration: 每段时长（秒），默认 10 分钟
            show_progress: 是否显示进度

        返回:
            分割后的音频文件路径列表
        """
        audio_info = self.get_audio_info(audio_path)
        duration = audio_info['duration']

        if duration <= segment_duration:
            return [audio_path]

        segments = []
        segment_index = 0
        current_time = 0

        if show_progress:
            total_segments = int(duration / segment_duration) + 1
            print(f"音频时长 {duration:.0f} 秒，将分割为 {total_segments} 段...")

        while current_time < duration:
            suffix = audio_path.suffix.lower() if audio_path.suffix.lower() in AUDIO_MIME_TYPES else ".mp3"
            segment_path = self.temp_dir / f"segment_{segment_index}{suffix}"

            try:
                self._convert_audio(
                    audio_path,
                    segment_path,
                    ss=current_time,
                    t=segment_duration,
                )
                self._validate_audio_output(segment_path)
                segments.append(segment_path)

                if show_progress:
                    print(f"  分割片段 {segment_index + 1}: {current_time:.0f}s - {min(current_time + segment_duration, duration):.0f}s")

            except Exception as exc:
                segment_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"分割音频片段 {segment_index} 失败：{_ffmpeg_user_message(exc)}"
                ) from exc

            current_time += segment_duration
            segment_index += 1

        return segments

    def transcribe_single_audio(self, audio_path: Path) -> str:
        """转录单个音频文件"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        retryable_statuses = {429, 502, 503, 504}
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            try:
                with open(audio_path, 'rb') as audio_file:
                    files = {
                        'file': (audio_path.name, audio_file, _audio_mime_type(audio_path)),
                        'model': (None, self.model)
                    }
                    response = requests.post(
                        self.api_base_url,
                        files=files,
                        headers=headers,
                        timeout=120,
                    )

                if response.status_code in retryable_statuses and attempt < max_attempts:
                    time.sleep(2 ** (attempt - 1))
                    continue

                response.raise_for_status()
                result = response.json()
                return result.get('text', response.text)
            except requests.RequestException as exc:
                status_code = getattr(exc.response, 'status_code', None)
                if status_code in retryable_statuses and attempt < max_attempts:
                    time.sleep(2 ** (attempt - 1))
                    continue
                if status_code == 503:
                    raise Exception(
                        f"模型 {self.model} 暂不可用。请切换为 TeleAI/TeleSpeechASR 后重试"
                    ) from exc
                raise Exception(f"提取文字时出错: {str(exc)}") from exc

        raise Exception(f"模型 {self.model} 多次请求失败，请稍后重试")

    def extract_text_from_audio(self, audio_path: Path, show_progress: bool = True) -> str:
        """从音频文件中提取文字（支持大文件自动分段）"""
        if not self.api_key:
            raise ValueError("未设置 API 密钥，请设置环境变量 API_KEY")

        # 检查文件大小和时长
        audio_info = self.get_audio_info(audio_path)
        max_duration = 3600  # 1 小时
        max_size = 50 * 1024 * 1024  # 50MB

        # 判断是否需要分段
        need_split = audio_info['duration'] > max_duration or audio_info['size'] > max_size

        if not need_split:
            # 文件在限制范围内，直接处理
            if show_progress:
                print("正在识别语音...")
            return self.transcribe_single_audio(audio_path)

        # 需要分段处理
        if show_progress:
            print(f"音频文件较大（时长: {audio_info['duration']:.0f}秒, 大小: {audio_info['size'] / 1024 / 1024:.1f}MB）")
            print("将自动分段处理...")

        # 分割音频
        segments = self.split_audio(audio_path, segment_duration=540, show_progress=show_progress)  # 9分钟一段，留余量

        # 逐段转录
        all_texts = []
        for i, segment_path in enumerate(segments):
            if show_progress:
                print(f"正在识别第 {i + 1}/{len(segments)} 段...")

            text = self.transcribe_single_audio(segment_path)
            all_texts.append(text)

            # 清理分段文件
            if segment_path != audio_path:
                self.cleanup_files(segment_path)

        # 合并文本
        merged_text = ''.join(all_texts)

        if show_progress:
            print(f"语音识别完成，共处理 {len(segments)} 个片段")

        return merged_text

    def cleanup_files(self, *file_paths: Path):
        """清理指定的文件"""
        for file_path in file_paths:
            if file_path.exists():
                file_path.unlink()


def get_video_info(share_link: str, douyin_cookie: str = "") -> dict:
    """获取视频信息和下载链接"""
    processor = DouyinProcessor(douyin_cookie=douyin_cookie)
    return processor.parse_share_url(share_link)


def download_video(share_link: str, output_dir: str = ".") -> Path:
    """下载视频到指定目录"""
    processor = DouyinProcessor()
    video_info = processor.parse_share_url(share_link)
    return processor.download_video(video_info, Path(output_dir))


def extract_text(share_link: str, api_key: Optional[str] = None, output_dir: Optional[str] = None,
                 save_video: bool = False, show_progress: bool = True,
                 douyin_cookie: str = "", model: Optional[str] = None) -> dict:
    """
    从视频中提取文案并保存到文件

    返回:
        dict: 包含 video_info, text, output_path 的字典
    """
    api_key = api_key or os.getenv('API_KEY') or os.getenv('DOUYIN_API_KEY')
    if not api_key:
        raise ValueError("未设置环境变量 API_KEY，请先获取硅基流动 API 密钥")

    processor = DouyinProcessor(api_key, model=model, douyin_cookie=douyin_cookie)

    if show_progress:
        print("正在解析抖音分享链接...")
    video_info = processor.parse_share_url(share_link)

    if show_progress:
        print("正在下载视频...")
    video_path = processor.download_video(video_info, show_progress=show_progress)

    if show_progress:
        print("正在提取音频...")
    audio_path = processor.extract_audio(video_path, show_progress=show_progress)

    if show_progress:
        print("正在从音频中提取文本...")
    text_content = processor.extract_text_from_audio(audio_path, show_progress=show_progress)

    result = {
        "video_info": video_info,
        "text": text_content,
        "output_path": None
    }

    # 保存到文件
    if output_dir:
        output_base = Path(output_dir)
        video_folder = output_base / video_info['video_id']
        video_folder.mkdir(parents=True, exist_ok=True)

        # 保存文案为 Markdown 格式
        transcript_path = video_folder / "transcript.md"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(f"# {video_info['title']}\n\n")
            f.write(f"| 属性 | 值 |\n")
            f.write(f"|------|----|\n")
            f.write(f"| 视频ID | `{video_info['video_id']}` |\n")
            f.write(f"| 提取时间 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n")
            f.write(f"| 下载链接 | [点击下载]({video_info['url']}) |\n\n")
            f.write(f"---\n\n")
            f.write(f"## 文案内容\n\n")
            f.write(text_content)

        result["output_path"] = str(video_folder)

        if show_progress:
            print(f"文案已保存到: {transcript_path}")

        # 保存视频 (可选)
        if save_video:
            saved_video_path = video_folder / f"{video_info['video_id']}.mp4"
            shutil.copy2(video_path, saved_video_path)
            if show_progress:
                print(f"视频已保存到: {saved_video_path}")

    # 清理临时文件
    if show_progress:
        print("正在清理临时文件...")
    processor.cleanup_files(video_path, audio_path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="抖音无水印视频下载和文案提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 获取视频信息和下载链接
  python douyin_downloader.py --link "抖音分享链接" --action info

  # 下载视频
  python douyin_downloader.py --link "抖音分享链接" --action download --output ./videos

  # 提取文案并保存到文件 (需要设置 API_KEY 环境变量)
  python douyin_downloader.py --link "抖音分享链接" --action extract --output ./output

  # 提取文案并同时保存视频
  python douyin_downloader.py --link "抖音分享链接" --action extract --output ./output --save-video
        """
    )

    parser.add_argument("--link", "-l", required=True, help="抖音分享链接或包含链接的文本")
    parser.add_argument("--action", "-a", choices=["info", "download", "extract"],
                        default="info", help="操作类型: info(获取信息), download(下载视频), extract(提取文案)")
    parser.add_argument("--output", "-o", default="./output", help="输出目录 (默认 ./output)")
    parser.add_argument("--api-key", "-k", help="硅基流动 API 密钥 (也可通过 API_KEY 环境变量设置)")
    parser.add_argument("--save-video", "-v", action="store_true", help="提取文案时同时保存视频")
    parser.add_argument("--quiet", "-q", action="store_true", help="安静模式，减少输出")

    args = parser.parse_args()

    try:
        if args.action == "info":
            info = get_video_info(args.link)
            print("\n" + "=" * 50)
            print("视频信息:")
            print("=" * 50)
            print(f"视频ID: {info['video_id']}")
            print(f"标题: {info['title']}")
            print(f"下载链接: {info['url']}")
            print("=" * 50)

        elif args.action == "download":
            video_path = download_video(args.link, args.output)
            print(f"\n视频已保存到: {video_path}")

        elif args.action == "extract":
            result = extract_text(
                args.link,
                args.api_key,
                output_dir=args.output,
                save_video=args.save_video,
                show_progress=not args.quiet
            )

            if not args.quiet:
                print("\n" + "=" * 50)
                print("提取完成!")
                print("=" * 50)
                print(f"视频ID: {result['video_info']['video_id']}")
                print(f"标题: {result['video_info']['title']}")
                if result['output_path']:
                    print(f"保存位置: {result['output_path']}")
                print("=" * 50)
                print("\n文案内容:\n")
                print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
                print("\n" + "=" * 50)

    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
