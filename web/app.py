#!/usr/bin/env python3
"""
抖音视频文案提取器 WebUI

启动方式:
    cd douyin-mcp-server
    export API_KEY="sk-xxx"
    python web/app.py
    # 访问 http://localhost:8080
"""

import os
import re
import sys
import asyncio
from pathlib import Path
from urllib.parse import quote

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "douyin-video" / "scripts"))

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import requests

# 导入抖音处理模块
from douyin_downloader import get_video_info, extract_text, HEADERS
from xiaohongshu_downloader import get_xiaohongshu_video_info, extract_xiaohongshu_text

app = FastAPI(title="SocialScript 社媒运营口播提取工具", version="1.0.0")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")
VIDEO_INFO_CACHE: dict[str, dict] = {}


def is_xiaohongshu_url(value: str) -> bool:
    return bool(re.search(r"https?://(?:[^/]+\.)?(?:xiaohongshu\.com|xhslink\.com|xhslink\.cn)(?:[/:?#]|$)", value, re.I))


class VideoRequest(BaseModel):
    """视频请求模型"""
    url: str
    api_key: str = ""  # 可选，从前端传入
    model: str = "TeleAI/TeleSpeechASR"  # 硅基流动语音识别模型
    douyin_cookie: str = ""  # 可选，仅用于本次本地解析请求
    xiaohongshu_cookie: str = ""  # 可选，仅用于本次小红书解析请求


class VideoInfoResponse(BaseModel):
    """视频信息响应"""
    success: bool
    video_id: str = ""
    title: str = ""
    download_url: str = ""
    error: str = ""


class ExtractResponse(BaseModel):
    """文案提取响应"""
    success: bool
    video_id: str = ""
    title: str = ""
    text: str = ""
    download_url: str = ""
    error: str = ""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页面"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    """健康检查"""
    api_key = os.getenv("API_KEY", "")
    return {
        "status": "ok",
        "api_key_configured": bool(api_key)
    }


@app.post("/api/video/info", response_model=VideoInfoResponse)
async def get_info(req: VideoRequest):
    """获取视频信息（无需 API_KEY）"""
    try:
        if is_xiaohongshu_url(req.url):
            info = await asyncio.to_thread(
                get_xiaohongshu_video_info,
                req.url,
                req.xiaohongshu_cookie or os.getenv("XIAOHONGSHU_COOKIE", ""),
            )
        else:
            douyin_cookie = req.douyin_cookie or os.getenv("DOUYIN_COOKIE", "")
            info = await asyncio.to_thread(get_video_info, req.url, douyin_cookie)
        VIDEO_INFO_CACHE[info["video_id"]] = info
        return VideoInfoResponse(
            success=True,
            video_id=info["video_id"],
            title=info["title"],
            download_url=info["url"]
        )
    except Exception as e:
        return VideoInfoResponse(success=False, error=str(e))


@app.post("/api/video/extract", response_model=ExtractResponse)
async def extract_transcript(req: VideoRequest):
    """提取视频文案（需要 API_KEY）"""
    # 优先使用请求中的 API Key，其次使用环境变量
    api_key = req.api_key or os.getenv("API_KEY", "")
    if not api_key:
        return ExtractResponse(
            success=False,
            error="请先配置 API Key"
        )

    try:
        model = req.model.strip() or "TeleAI/TeleSpeechASR"
        if is_xiaohongshu_url(req.url):
            result = await asyncio.to_thread(
                extract_xiaohongshu_text,
                req.url,
                api_key,
                model,
                req.xiaohongshu_cookie or os.getenv("XIAOHONGSHU_COOKIE", ""),
            )
        else:
            result = await asyncio.to_thread(
                extract_text,
                req.url,
                api_key=api_key,
                model=model,
                show_progress=False,
                douyin_cookie=req.douyin_cookie or os.getenv("DOUYIN_COOKIE", "")
            )
        VIDEO_INFO_CACHE[result["video_info"]["video_id"]] = result["video_info"]
        return ExtractResponse(
            success=True,
            video_id=result["video_info"]["video_id"],
            title=result["video_info"]["title"],
            text=result["text"],
            download_url=result["video_info"]["url"]
        )
    except Exception as e:
        return ExtractResponse(success=False, error=str(e))


def _content_disposition(filename: str) -> str:
    """构造安全的 Content-Disposition 头（ASCII 回退 + RFC 5987 编码）"""
    ascii_name = re.sub(r'[^A-Za-z0-9._-]', '_', filename) or "video.mp4"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@app.get("/api/video/download")
async def download_video(video_id: str, filename: str = "video.mp4"):
    """代理下载视频（解决跨域和请求头问题）

    只接受抖音视频 ID，由服务端自行解析出 CDN 链接，避免代理任意 URL。
    """
    if not re.fullmatch(r'\d+', video_id):
        raise HTTPException(status_code=400, detail="无效的视频 ID")

    try:
        info = VIDEO_INFO_CACHE.get(video_id)
        if not info:
            share_url = f"https://www.iesdouyin.com/share/video/{video_id}"
            info = await asyncio.to_thread(
                get_video_info, share_url, os.getenv("DOUYIN_COOKIE", "")
            )

        # 完整的请求头，模拟浏览器访问
        download_headers = {
            'User-Agent': HEADERS['User-Agent'],
            'Referer': 'https://www.douyin.com/',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'identity',
            'Connection': 'keep-alive',
        }

        response = await asyncio.to_thread(
            requests.get, info["url"], headers=download_headers, stream=True, allow_redirects=True
        )
        response.raise_for_status()

        content_length = response.headers.get("content-length", "")

        def iter_content():
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        headers = {
            "Content-Disposition": _content_disposition(filename),
        }
        if content_length:
            headers["Content-Length"] = content_length

        return StreamingResponse(
            iter_content(),
            media_type="video/mp4",
            headers=headers
        )
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"下载失败: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """启动服务"""
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    print(f"🚀 启动文案提取器 WebUI: http://localhost:{port}")
    print(f"📝 API_KEY 配置状态: {'已配置' if os.getenv('API_KEY') else '未配置'}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
