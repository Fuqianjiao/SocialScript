#!/usr/bin/env python3
"""
抖音无水印视频下载并提取文本的 MCP 服务器

该服务器提供以下功能：
1. 解析抖音分享链接获取无水印视频链接
2. 下载视频并提取音频
3. 从音频中提取文本内容
4. 自动清理中间文件

语音识别支持两个后端：
- API_KEY: 硅基流动 (https://cloud.siliconflow.cn)，与 README/WebUI/Skill 一致
- DASHSCOPE_API_KEY: 阿里云百炼，兼容 1.2.x 及更早版本的配置
"""

import os
import re
import json
import shutil
import requests
import tempfile
import asyncio
from pathlib import Path
from typing import Optional
import ffmpeg

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp import Context

from .asr_module import create_asr_instance


# 创建 MCP 服务器实例
mcp = FastMCP("Douyin MCP Server",
              dependencies=["requests", "ffmpeg-python", "tqdm", "dashscope"])

# 请求头，模拟移动端访问
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) EdgiOS/121.0.2277.107 Version/17.0 Mobile/15E148 Safari/604.1'
}

# 默认 API 配置
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
DEFAULT_SILICONFLOW_MODEL = "TeleAI/TeleSpeechASR"
DEFAULT_DASHSCOPE_MODEL = "qwen3-asr-flash"


def resolve_asr_config(model: Optional[str] = None) -> tuple:
    """
    根据环境变量解析语音识别后端配置

    返回: (provider, api_key, model)
    """
    api_key = os.getenv('API_KEY')
    if api_key:
        return 'siliconflow', api_key, model or DEFAULT_SILICONFLOW_MODEL

    dashscope_key = os.getenv('DASHSCOPE_API_KEY')
    if dashscope_key:
        return 'dashscope', dashscope_key, model or DEFAULT_DASHSCOPE_MODEL

    raise ValueError(
        "未设置 API 密钥：请设置环境变量 API_KEY（硅基流动，https://cloud.siliconflow.cn）"
        "或 DASHSCOPE_API_KEY（阿里云百炼）"
    )


class DouyinProcessor:
    """抖音视频处理器"""

    def __init__(self, api_key: str = "", provider: str = "siliconflow", model: Optional[str] = None):
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self.temp_dir = Path(tempfile.mkdtemp())

    def __del__(self):
        """清理临时目录"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def parse_share_url(self, share_text: str) -> dict:
        """从分享文本中提取无水印视频链接"""
        # 提取分享链接
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', share_text)
        if not urls:
            raise ValueError("未找到有效的分享链接")

        share_url = urls[0]
        share_response = requests.get(share_url, headers=HEADERS)
        video_id = share_response.url.split("?")[0].strip("/").split("/")[-1]
        share_url = f'https://www.iesdouyin.com/share/video/{video_id}'

        # 获取视频页面内容
        response = requests.get(share_url, headers=HEADERS)
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

        if VIDEO_ID_PAGE_KEY in json_data["loaderData"]:
            original_video_info = json_data["loaderData"][VIDEO_ID_PAGE_KEY]["videoInfoRes"]
        elif NOTE_ID_PAGE_KEY in json_data["loaderData"]:
            original_video_info = json_data["loaderData"][NOTE_ID_PAGE_KEY]["videoInfoRes"]
        else:
            raise Exception("无法从JSON中解析视频或图集信息")

        data = original_video_info["item_list"][0]

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

    def download_video(self, video_info: dict) -> Path:
        """下载视频到临时目录"""
        filename = f"{video_info['video_id']}.mp4"
        filepath = self.temp_dir / filename

        response = requests.get(video_info['url'], headers=HEADERS, stream=True)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        return filepath

    def extract_audio(self, video_path: Path) -> Path:
        """从视频文件中提取音频"""
        audio_path = video_path.with_suffix('.mp3')

        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(str(audio_path), acodec='libmp3lame', q=0)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            return audio_path
        except Exception as e:
            raise Exception(f"提取音频时出错: {str(e)}")

    def transcribe_audio(self, audio_path: Path, context: Optional[str] = None) -> str:
        """从音频文件中提取文字"""
        if self.provider == 'dashscope':
            return self._transcribe_dashscope(audio_path, context)
        return self._transcribe_siliconflow(audio_path)

    def _transcribe_siliconflow(self, audio_path: Path) -> str:
        """使用硅基流动 API 转录音频"""
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': (audio_path.name, audio_file, 'audio/mpeg'),
                'model': (None, self.model)
            }
            response = requests.post(SILICONFLOW_API_URL, files=files, headers=headers)

        if response.status_code != 200:
            raise Exception(f"语音识别请求失败 (HTTP {response.status_code}): {response.text[:200]}")

        result = response.json()
        if 'text' not in result:
            raise Exception(f"语音识别返回异常: {response.text[:200]}")
        return result['text'] or "未识别到文本内容"

    def _transcribe_dashscope(self, audio_path: Path, context: Optional[str] = None) -> str:
        """使用阿里云百炼 qwen3-asr 转录音频"""
        asr = create_asr_instance(self.api_key, self.model)
        result = asr.recognize_file(
            file_path=audio_path,
            context=context,
            enable_lid=True,
            enable_itn=False
        )
        if not result["success"]:
            raise Exception(f"语音识别失败: {result['error']}")
        return result["text"] or "未识别到文本内容"

    def cleanup_files(self, *file_paths: Path):
        """清理指定的文件"""
        for file_path in file_paths:
            if file_path.exists():
                file_path.unlink()


@mcp.tool()
def get_douyin_download_link(share_link: str) -> str:
    """
    获取抖音视频的无水印下载链接

    参数:
    - share_link: 抖音分享链接或包含链接的文本

    返回:
    - 包含下载链接和视频信息的JSON字符串
    """
    try:
        processor = DouyinProcessor()  # 获取下载链接不需要API密钥
        video_info = processor.parse_share_url(share_link)

        return json.dumps({
            "status": "success",
            "video_id": video_info["video_id"],
            "title": video_info["title"],
            "download_url": video_info["url"],
            "description": f"视频标题: {video_info['title']}",
            "usage_tip": "可以直接使用此链接下载无水印视频"
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"获取下载链接失败: {str(e)}"
        }, ensure_ascii=False, indent=2)


@mcp.tool()
async def extract_douyin_text(
    share_link: str,
    model: Optional[str] = None,
    context: Optional[str] = None,
    ctx: Context = None
) -> str:
    """
    从抖音分享链接提取视频中的文本内容

    参数:
    - share_link: 抖音分享链接或包含链接的文本
    - model: 语音识别模型（可选，硅基流动默认 FunAudioLLM/SenseVoiceSmall，百炼默认 qwen3-asr-flash）
    - context: 上下文文本，用于提高识别准确率（可选，仅百炼后端支持）

    返回:
    - 提取的文本内容

    注意: 需要设置环境变量 API_KEY（硅基流动）或 DASHSCOPE_API_KEY（阿里云百炼）
    """
    video_path = None
    audio_path = None
    try:
        provider, api_key, model_name = resolve_asr_config(model)
        processor = DouyinProcessor(api_key, provider, model_name)

        # 解析视频链接
        if ctx:
            await ctx.info("正在解析抖音分享链接...")
        video_info = await asyncio.to_thread(processor.parse_share_url, share_link)

        # 下载视频并提取音频
        if ctx:
            await ctx.info(f"正在下载视频: {video_info['title']}")
        video_path = await asyncio.to_thread(processor.download_video, video_info)

        if ctx:
            await ctx.info("正在提取音频...")
        audio_path = await asyncio.to_thread(processor.extract_audio, video_path)

        # 语音识别
        if ctx:
            await ctx.info("正在从音频中提取文本...")
        full_context = f"视频标题: {video_info['title']}"
        if context:
            full_context = f"{context}\n{full_context}"
        text_content = await asyncio.to_thread(processor.transcribe_audio, audio_path, full_context)

        if ctx:
            await ctx.info("文本提取完成!")
        return text_content

    except Exception as e:
        raise Exception(f"提取抖音视频文本失败: {str(e)}")
    finally:
        # 清理临时文件
        for path in (video_path, audio_path):
            if path is not None and path.exists():
                path.unlink(missing_ok=True)


@mcp.tool()
def recognize_audio_file(
    file_path: str,
    context: Optional[str] = None,
    language: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    识别本地音频文件中的文本

    参数:
    - file_path: 本地音频文件路径
    - context: 上下文文本，用于提高识别准确率（可选）
    - language: 指定语言代码（如 'zh', 'en'），可选，默认自动检测
    - model: 语音识别模型（可选，默认使用qwen3-asr-flash）

    返回:
    - 识别的文本内容

    注意: 需要设置环境变量 DASHSCOPE_API_KEY
    """
    try:
        # 从环境变量获取API密钥
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("未设置环境变量 DASHSCOPE_API_KEY，请在配置中添加阿里云百炼API密钥")

        # 创建ASR实例
        asr = create_asr_instance(api_key, model or DEFAULT_DASHSCOPE_MODEL)

        # 识别音频文件
        result = asr.recognize_file(
            file_path=file_path,
            context=context,
            language=language,
            enable_lid=True,
            enable_itn=False
        )

        if result["success"]:
            return json.dumps({
                "status": "success",
                "text": result["text"],
                "language": result.get("language"),
                "usage": result.get("usage"),
                "request_id": result.get("request_id")
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "error": result["error"]
            }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"识别音频文件失败: {str(e)}"
        }, ensure_ascii=False, indent=2)


@mcp.tool()
def recognize_audio_url(
    audio_url: str,
    context: Optional[str] = None,
    language: Optional[str] = None,
    model: Optional[str] = None
) -> str:
    """
    识别在线音频URL中的文本

    参数:
    - audio_url: 音频URL链接
    - context: 上下文文本，用于提高识别准确率（可选）
    - language: 指定语言代码（如 'zh', 'en'），可选，默认自动检测
    - model: 语音识别模型（可选，默认使用qwen3-asr-flash）

    返回:
    - 识别的文本内容

    注意: 需要设置环境变量 DASHSCOPE_API_KEY
    """
    try:
        # 从环境变量获取API密钥
        api_key = os.getenv('DASHSCOPE_API_KEY')
        if not api_key:
            raise ValueError("未设置环境变量 DASHSCOPE_API_KEY，请在配置中添加阿里云百炼API密钥")

        # 创建ASR实例
        asr = create_asr_instance(api_key, model or DEFAULT_DASHSCOPE_MODEL)

        # 识别音频URL
        result = asr.recognize_url(
            audio_url=audio_url,
            context=context,
            language=language,
            enable_lid=True,
            enable_itn=False
        )

        if result["success"]:
            return json.dumps({
                "status": "success",
                "text": result["text"],
                "language": result.get("language"),
                "usage": result.get("usage"),
                "request_id": result.get("request_id")
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "status": "error",
                "error": result["error"]
            }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": f"识别音频URL失败: {str(e)}"
        }, ensure_ascii=False, indent=2)


@mcp.tool()
def parse_douyin_video_info(share_link: str) -> str:
    """
    解析抖音分享链接，获取视频基本信息

    参数:
    - share_link: 抖音分享链接或包含链接的文本

    返回:
    - 视频信息（JSON格式字符串）
    """
    try:
        processor = DouyinProcessor()  # 不需要API密钥来解析链接
        video_info = processor.parse_share_url(share_link)

        return json.dumps({
            "video_id": video_info["video_id"],
            "title": video_info["title"],
            "download_url": video_info["url"],
            "status": "success"
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, ensure_ascii=False, indent=2)


@mcp.resource("douyin://video/{video_id}")
def get_video_info(video_id: str) -> str:
    """
    获取指定视频ID的详细信息

    参数:
    - video_id: 抖音视频ID

    返回:
    - 视频详细信息
    """
    share_url = f"https://www.iesdouyin.com/share/video/{video_id}"
    try:
        processor = DouyinProcessor()
        video_info = processor.parse_share_url(share_url)
        return json.dumps(video_info, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取视频信息失败: {str(e)}"


@mcp.prompt()
def douyin_text_extraction_guide() -> str:
    """抖音视频文本提取使用指南"""
    return """
# 抖音视频文本提取使用指南

## 功能说明
这个MCP服务器可以从抖音分享链接中提取视频的文本内容，以及获取无水印下载链接。

## 环境变量配置
语音识别支持两个后端，设置其中一个即可：
- `API_KEY`: 硅基流动 API 密钥（推荐，获取地址 https://cloud.siliconflow.cn）
- `DASHSCOPE_API_KEY`: 阿里云百炼 API 密钥（兼容旧版本配置）

## 使用步骤
1. 复制抖音视频的分享链接
2. 在Claude Desktop配置中设置环境变量 API_KEY（或 DASHSCOPE_API_KEY）
3. 使用相应的工具进行操作

## 工具说明
- `extract_douyin_text`: 完整的文本提取流程（需要API密钥）
- `get_douyin_download_link`: 获取无水印视频下载链接（无需API密钥）
- `parse_douyin_video_info`: 仅解析视频基本信息
- `recognize_audio_file`: 识别本地音频文件（需要 DASHSCOPE_API_KEY）
- `recognize_audio_url`: 识别在线音频URL（需要 DASHSCOPE_API_KEY）
- `douyin://video/{video_id}`: 获取指定视频的详细信息

## Claude Desktop 配置示例
```json
{
  "mcpServers": {
    "douyin-mcp": {
      "command": "uvx",
      "args": ["douyin-mcp-server"],
      "env": {
        "API_KEY": "your-siliconflow-api-key"
      }
    }
  }
}
```

## 注意事项
- 需要提供有效的 API 密钥（通过环境变量）
- 支持大部分抖音视频格式
- 获取下载链接无需API密钥
"""


def main():
    """启动MCP服务器"""
    mcp.run()


if __name__ == "__main__":
    main()
