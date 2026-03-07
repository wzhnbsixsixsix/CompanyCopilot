import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Union

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent_pipeline import CompanyResearchPipeline
from .agent_service import DueDiligenceAgentService
from .agents import GuidanceAgent
from .config import get_settings


class DueDiligenceRequest(BaseModel):
    company_name: str = Field(min_length=1)
    prompt: str | None = None


class CompanyResearchRequest(BaseModel):
    domain: str = Field(min_length=1, description="公司域名，如 apple.com")


class ChatMessage(BaseModel):
    role: str = Field(description="消息角色：user、assistant、system")
    content: Union[str, list] = Field(
        description="消息内容，可以是字符串或OpenAI格式的内容块数组"
    )


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen-plus", description="模型名称")
    messages: list[ChatMessage] = Field(description="对话历史")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    stream: bool = Field(default=True, description="是否流式返回")
    max_tokens: int | None = Field(default=None, description="最大Token数")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="top_p参数")


app = FastAPI(title="CompanyCopilot Agent API", version="0.1.0")


def extract_text_content(content: Union[str, list]) -> str:
    """从OpenAI格式的content中提取纯文本内容

    Args:
        content: 字符串或OpenAI内容块数组

    Returns:
        提取的纯文本字符串
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # 从OpenAI content blocks中提取text部分
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return " ".join(text_parts)
    else:
        return str(content)  # fallback


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/due-diligence")
async def due_diligence(payload: DueDiligenceRequest) -> dict:
    """原有的简单背调功能（保留兼容性）"""
    try:
        settings = get_settings()
        service = DueDiligenceAgentService(settings)
        result = await service.run_due_diligence(
            company_name=payload.company_name,
            user_prompt=payload.prompt,
            structured=True,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/api/company-research")
async def company_research(payload: CompanyResearchRequest) -> dict:
    """新的企业背调功能（3-Agent流水线）"""
    try:
        settings = get_settings()
        pipeline = CompanyResearchPipeline(settings)
        report = await pipeline.run(payload.domain)
        return {"report": report}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/api/due-diligence-research")
async def due_diligence_research(payload: CompanyResearchRequest) -> dict:
    """快速尽职调查功能（产品维度调研）"""
    try:
        settings = get_settings()
        pipeline = CompanyResearchPipeline(settings)
        report = await pipeline.run(payload.domain, mode="quick")
        return {"report": report}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


async def generate_streaming_research_response(
    pipeline: CompanyResearchPipeline, domain: str, mode: str
) -> AsyncIterator[str]:
    """生成流式研究报告响应"""
    try:
        async for chunk in pipeline.run_streaming(domain, mode):
            if chunk.strip():  # 只输出非空内容块
                # 使用纯文本格式，不使用SSE格式
                yield chunk
    except Exception as e:
        # 错误处理
        error_msg = (
            f"\n\n**生成报告时遇到错误**: {str(e)}\n\n请检查网络连接和API配置后重试。"
        )
        yield error_msg


@app.post("/api/company-research/stream")
async def company_research_stream(payload: CompanyResearchRequest):
    """全面企业背调功能（流式输出）"""
    try:
        settings = get_settings()
        pipeline = CompanyResearchPipeline(settings)

        return StreamingResponse(
            generate_streaming_research_response(pipeline, payload.domain, "full"),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用nginx缓冲，确保真实流式
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/api/due-diligence-research/stream")
async def due_diligence_research_stream(payload: CompanyResearchRequest):
    """快速尽职调查功能（流式输出）"""
    try:
        settings = get_settings()
        pipeline = CompanyResearchPipeline(settings)

        return StreamingResponse(
            generate_streaming_research_response(pipeline, payload.domain, "quick"),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用nginx缓冲，确保真实流式
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.post("/v1/chat/completions")
async def chat_completions(payload: ChatCompletionRequest):
    """OpenAI兼容的对话接口，使用DashScope直接流式返回"""
    try:
        settings = get_settings()

        # 提取最后一条用户消息
        if not payload.messages:
            raise ValueError("Messages cannot be empty")

        # 构建对话历史字符串
        conversation_history = []
        for msg in payload.messages:
            text_content = extract_text_content(msg.content)
            if msg.role == "user":
                conversation_history.append(f"用户：{text_content}")
            elif msg.role == "assistant":
                conversation_history.append(f"助手：{text_content}")

        # 当前用户输入
        latest_message = (
            extract_text_content(payload.messages[-1].content)
            if payload.messages
            else ""
        )

        # 构建完整的提示词
        if len(conversation_history) > 1:
            context = "\n".join(conversation_history[:-1])
            prompt = f"对话历史：\n{context}\n\n当前用户问题：{latest_message}"
        else:
            prompt = latest_message

        # 使用系统prompt引导用户意图
        guidance_system_prompt = (
            "你是 CompanyCopilot 智能助手，专门帮助用户进行企业背景调查和尽职调查。\n\n"
            "**核心功能**：\n"
            "- 全面背景调查：输入 `/research 企业域名`（如 `/research apple.com`）启动全面的8维度企业调研\n"
            "- 快速尽调：输入 `/due-diligence 企业域名`（如 `/due-diligence apple.com`）进行快速的产品维度调研\n\n"
            "**功能对比**：\n"
            "- `/research`：全面调研8个维度（公司概况、产品服务、市场表现、团队信息、融资状况、技术栈、近期动态、竞争对手）\n"
            "- `/due-diligence`：快速调研产品维度（产品线、定价策略、商业模式等），适合快速评估和测试\n\n"
            "**工作原则**：\n"
            "1. **意图识别**：当用户询问某个公司/企业的信息时，主动引导使用相应命令\n"
            "2. **命令推荐**：根据用户需求推荐合适的调研模式（全面 vs 快速）\n"
            "3. **格式规范**：强调域名格式的重要性（如 apple.com，不是 Apple 或 apple）\n"
            "4. **简洁高效**：作为引导员，回答要简洁明了，不进行实际的企业调研工作\n\n"
            "**典型对话示例**：\n"
            '用户："帮我了解一下苹果公司的情况"\n'
            '你："我来帮您调研苹果公司！您可以选择：\n\n'
            "**全面调研**（8个维度）：`/research apple.com`\n"
            "**快速调研**（产品重点）：`/due-diligence apple.com`\n\n"
            '建议测试时使用快速调研，正式分析时使用全面调研。"\n\n'
            "记住：你不直接提供企业信息，而是引导用户使用正确的命令来获取专业的调研报告。"
        )

        # 构建完整的消息历史用于OpenAI API
        messages = [
            {"role": "system", "content": guidance_system_prompt},
            {"role": "user", "content": prompt},
        ]

        # 创建OpenAI客户端指向DashScope
        client = openai.AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        if payload.stream:
            # 真正的流式响应
            return StreamingResponse(
                generate_real_openai_stream(client, messages, payload.model, settings),
                media_type="text/event-stream",
            )
        else:
            # 非流式响应
            completion = await client.chat.completions.create(
                model=settings.dashscope_model,
                messages=messages,
                temperature=payload.temperature,
                max_tokens=payload.max_tokens,
                top_p=payload.top_p,
                extra_body={"enable_thinking": False},
            )

            return {
                "id": completion.id,
                "object": completion.object,
                "created": completion.created,
                "model": completion.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in completion.choices
                ],
            }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


async def generate_real_openai_stream(
    client: openai.AsyncOpenAI, messages: list[dict], model_name: str, settings
) -> AsyncIterator[str]:
    """使用真实的DashScope流式API生成SSE响应"""
    try:
        # 调用DashScope的流式API
        stream = await client.chat.completions.create(
            model=settings.dashscope_model,
            messages=messages,
            stream=True,
            temperature=0.7,
            extra_body={"enable_thinking": False},
        )

        # 流式传输每个token
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_data = {
                    "id": chunk.id,
                    "object": chunk.object,
                    "created": chunk.created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.choices[0].delta.content},
                            "finish_reason": chunk.choices[0].finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # 发送结束标记
        final_chunk_data = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_running_loop().time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        # 错误处理
        error_data = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_running_loop().time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n抱歉，生成回复时遇到错误：{str(e)}"},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


# 保留旧的假流式函数以防需要回滚
async def generate_openai_stream(content: str) -> AsyncIterator[str]:
    """生成OpenAI兼容的SSE流式响应（假流式，已弃用）"""
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(asyncio.get_running_loop().time())

    # 将内容分割成小块进行流式传输
    words = content.split()
    chunk_size = 3  # 每次发送3个词

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        chunk_content = " " + " ".join(chunk_words) if i > 0 else " ".join(chunk_words)

        chunk_data = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": timestamp,
            "model": "guidance-agent",
            "choices": [
                {"index": 0, "delta": {"content": chunk_content}, "finish_reason": None}
            ],
        }

        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.05)  # 50ms 延迟模拟真实流式体验

    # 发送结束标记
    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": "guidance-agent",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }

    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
