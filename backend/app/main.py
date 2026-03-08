import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Optional, Union

import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .agent_pipeline import CompanyResearchPipeline
from .agent_service import DueDiligenceAgentService
from .agents import GuidanceAgent
from .config import get_settings
from .rag import (
    DocumentMetadata,
    KnowledgeService,
    ImageProcessor,
    FileStorage,
    RAGQueryRequest,
)
from .rag.schemas import (
    DocumentListResponse,
    DeleteDocumentResponse,
    DocumentType,
    KnowledgeBase,
    CreateKnowledgeBaseRequest,
    UpdateKnowledgeBaseRequest,
    KnowledgeBaseListResponse,
    KnowledgeBaseDetailResponse,
    DeleteKnowledgeBaseResponse,
    DocumentChunksResponse,
    ChunkInfo,
    RetrievedContext,
)
from .agents.rag_agent import RAGAgentFactory


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

# ============ RAG Services Initialization ============
# 延迟初始化，在首次使用时创建
_knowledge_service: Optional[KnowledgeService] = None
_image_processor: Optional[ImageProcessor] = None
_file_storage: Optional[FileStorage] = None
_rag_agent_factory: Optional[RAGAgentFactory] = None


def get_knowledge_service() -> KnowledgeService:
    """获取知识库服务实例（单例）"""
    global _knowledge_service
    if _knowledge_service is None:
        settings = get_settings()
        _knowledge_service = KnowledgeService(settings)
    return _knowledge_service


def get_image_processor() -> ImageProcessor:
    """获取图片处理器实例（单例）"""
    global _image_processor
    if _image_processor is None:
        settings = get_settings()
        _image_processor = ImageProcessor(settings, get_knowledge_service())
    return _image_processor


def get_file_storage() -> FileStorage:
    """获取文件存储实例（单例）"""
    global _file_storage
    if _file_storage is None:
        _file_storage = FileStorage()
    return _file_storage


def get_rag_agent_factory() -> RAGAgentFactory:
    """获取 RAG Agent 工厂实例（单例）"""
    global _rag_agent_factory
    if _rag_agent_factory is None:
        settings = get_settings()
        _rag_agent_factory = RAGAgentFactory(settings)
        _rag_agent_factory.set_knowledge_service(get_knowledge_service())
    return _rag_agent_factory


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
            "你是 CompanyCopilot 智能助手，专门帮助用户进行企业背景调查、尽职调查和知识库问答。\n\n"
            "**核心功能**：\n"
            "- 全面背景调查：输入 `/research 企业域名`（如 `/research apple.com`）启动全面的8维度企业调研\n"
            "- 快速尽调：输入 `/due-diligence 企业域名`（如 `/due-diligence apple.com`）进行快速的产品维度调研\n"
            "- 知识库问答：输入 `/kg 问题`（如 `/kg 公司的年度营收是多少？`）从已上传的文档中检索信息并回答\n\n"
            "**功能对比**：\n"
            "- `/research`：全面调研8个维度（公司概况、产品服务、市场表现、团队信息、融资状况、技术栈、近期动态、竞争对手）\n"
            "- `/due-diligence`：快速调研产品维度（产品线、定价策略、商业模式等），适合快速评估和测试\n"
            "- `/kg`：基于用户上传的文档进行智能问答，支持 PDF、Word、Excel、PPT、图片等格式\n\n"
            "**工作原则**：\n"
            "1. **意图识别**：当用户询问某个公司/企业的信息时，主动引导使用相应命令\n"
            "2. **命令推荐**：根据用户需求推荐合适的调研模式（全面 vs 快速 vs 知识库）\n"
            "3. **格式规范**：强调域名格式的重要性（如 apple.com，不是 Apple 或 apple）\n"
            "4. **简洁高效**：作为引导员，回答要简洁明了，不进行实际的企业调研工作\n\n"
            "**典型对话示例**：\n"
            '用户："帮我了解一下苹果公司的情况"\n'
            '你："我来帮您调研苹果公司！您可以选择：\n\n'
            "**全面调研**（8个维度）：`/research apple.com`\n"
            "**快速调研**（产品重点）：`/due-diligence apple.com`\n\n"
            '建议测试时使用快速调研，正式分析时使用全面调研。"\n\n'
            '用户："我上传的文档里有什么内容？"\n'
            '你："您可以使用 `/kg 您的问题` 来查询已上传文档的内容。例如：\n'
            "`/kg 文档的主要内容是什么？`\n"
            '系统会自动检索相关文档并为您生成答案。"\n\n'
            "记住：你不直接提供企业信息或文档内容，而是引导用户使用正确的命令来获取专业的调研报告或知识库回答。"
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


# ============ RAG API Endpoints ============


# ---- 知识库管理 API ----


@app.post("/api/rag/knowledge-bases", response_model=KnowledgeBase)
async def create_knowledge_base(request: CreateKnowledgeBaseRequest):
    """创建知识库"""
    try:
        knowledge_service = get_knowledge_service()
        kb = knowledge_service.create_knowledge_base(
            name=request.name,
            description=request.description,
            user_id=request.user_id,
            workspace_id=request.workspace_id,
        )
        return kb
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to create knowledge base: {exc}"
        ) from exc


@app.get("/api/rag/knowledge-bases", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
):
    """列出知识库"""
    try:
        knowledge_service = get_knowledge_service()
        kbs = knowledge_service.list_knowledge_bases(user_id, workspace_id)
        return KnowledgeBaseListResponse(knowledge_bases=kbs, total=len(kbs))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list knowledge bases: {exc}"
        ) from exc


@app.get("/api/rag/knowledge-bases/{kb_id}", response_model=KnowledgeBaseDetailResponse)
async def get_knowledge_base(kb_id: str):
    """获取知识库详情（包含文档列表）"""
    try:
        knowledge_service = get_knowledge_service()
        kb = knowledge_service.get_knowledge_base(kb_id)
        if kb is None:
            raise HTTPException(
                status_code=404, detail=f"Knowledge base not found: {kb_id}"
            )
        documents = knowledge_service.list_documents(knowledge_base_id=kb_id)
        return KnowledgeBaseDetailResponse(knowledge_base=kb, documents=documents)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to get knowledge base: {exc}"
        ) from exc


@app.put("/api/rag/knowledge-bases/{kb_id}", response_model=KnowledgeBase)
async def update_knowledge_base(kb_id: str, request: UpdateKnowledgeBaseRequest):
    """更新知识库"""
    try:
        knowledge_service = get_knowledge_service()
        kb = knowledge_service.update_knowledge_base(
            kb_id=kb_id,
            name=request.name,
            description=request.description,
        )
        if kb is None:
            raise HTTPException(
                status_code=404, detail=f"Knowledge base not found: {kb_id}"
            )
        return kb
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to update knowledge base: {exc}"
        ) from exc


@app.delete(
    "/api/rag/knowledge-bases/{kb_id}", response_model=DeleteKnowledgeBaseResponse
)
async def delete_knowledge_base(kb_id: str):
    """删除知识库"""
    try:
        knowledge_service = get_knowledge_service()

        # 不允许删除默认知识库
        if kb_id == "default":
            raise HTTPException(
                status_code=400, detail="Cannot delete default knowledge base"
            )

        # 检查知识库是否存在
        kb = knowledge_service.get_knowledge_base(kb_id)
        if kb is None:
            raise HTTPException(
                status_code=404, detail=f"Knowledge base not found: {kb_id}"
            )

        docs = knowledge_service.list_documents(knowledge_base_id=kb_id)

        # 删除知识库
        success = await knowledge_service.delete_knowledge_base(kb_id)

        if success:
            file_storage = get_file_storage()
            for doc in docs:
                ext = (
                    "." + doc.filename.rsplit(".", 1)[-1] if "." in doc.filename else ""
                )
                is_image = doc.file_type == DocumentType.IMAGE
                file_storage.delete_file(doc.id, ext, is_image)

        return DeleteKnowledgeBaseResponse(
            success=success,
            message="Knowledge base deleted successfully"
            if success
            else "Failed to delete knowledge base",
            kb_id=kb_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete knowledge base: {exc}"
        ) from exc


@app.post("/api/rag/knowledge-bases/{kb_id}/upload", response_model=DocumentMetadata)
async def upload_to_knowledge_base(
    kb_id: str,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    workspace_id: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """上传文档到指定知识库"""
    try:
        knowledge_service = get_knowledge_service()

        # 检查知识库是否存在
        kb = knowledge_service.get_knowledge_base(kb_id)
        if kb is None:
            raise HTTPException(
                status_code=404, detail=f"Knowledge base not found: {kb_id}"
            )

        # 读取文件内容
        content = await file.read()
        filename = file.filename or "unknown"

        # 检测文件类型
        doc_type = KnowledgeService.get_document_type(filename)
        if doc_type is None:
            raise ValueError(f"Unsupported file type: {filename}")

        # 保存文件到本地存储
        file_storage = get_file_storage()
        is_image = doc_type == DocumentType.IMAGE

        # 根据文件类型选择处理方式
        if is_image:
            # 图片使用 ImageProcessor 处理
            image_processor = get_image_processor()
            metadata = await image_processor.process_image(
                content=content,
                filename=filename,
                knowledge_base_id=kb_id,
                user_id=user_id,
                workspace_id=workspace_id,
                description=description,
            )
        else:
            # 其他文档使用 KnowledgeService 处理
            metadata = await knowledge_service.process_document(
                content=content,
                filename=filename,
                knowledge_base_id=kb_id,
                user_id=user_id,
                workspace_id=workspace_id,
                description=description,
            )

        # 保存原始文件
        file_storage.save_file(content, filename, metadata.id, is_image)

        return metadata

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


# ---- 文档上传 API（保留旧接口，使用默认知识库）----


@app.post("/api/rag/upload", response_model=DocumentMetadata)
async def rag_upload_document(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    workspace_id: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
):
    """上传文档到知识库

    支持的文档类型：PDF, TXT, MD, DOCX, XLSX, PPTX, CSV, 图片 (JPG, PNG, JPEG, WEBP)
    """
    try:
        # 读取文件内容
        content = await file.read()
        filename = file.filename or "unknown"

        # 检测文件类型
        doc_type = KnowledgeService.get_document_type(filename)
        if doc_type is None:
            raise ValueError(f"Unsupported file type: {filename}")

        # 保存文件到本地存储
        file_storage = get_file_storage()
        is_image = doc_type == DocumentType.IMAGE

        # 根据文件类型选择处理方式
        if is_image:
            # 图片使用 ImageProcessor 处理
            image_processor = get_image_processor()
            metadata = await image_processor.process_image(
                content=content,
                filename=filename,
                user_id=user_id,
                workspace_id=workspace_id,
                description=description,
            )
        else:
            # 其他文档使用 KnowledgeService 处理
            knowledge_service = get_knowledge_service()
            metadata = await knowledge_service.process_document(
                content=content,
                filename=filename,
                user_id=user_id,
                workspace_id=workspace_id,
                description=description,
            )

        # 保存原始文件
        file_storage.save_file(content, filename, metadata.id, is_image)

        return metadata

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@app.post("/api/rag/query/stream")
async def rag_query_stream(request: RAGQueryRequest):
    """流式 RAG 问答

    从知识库检索相关文档并使用 RAG Agent 生成回答
    """
    try:
        knowledge_service = get_knowledge_service()
        settings = get_settings()

        # 检索相关文档
        context = await knowledge_service.retrieve(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
        )

        # 如果没有检索到相关内容
        if not context.chunks:
            return StreamingResponse(
                generate_no_context_response(request.query),
                media_type="text/event-stream",
            )

        # 使用 RAG Agent 生成回答（包含 sources 信息）
        return StreamingResponse(
            generate_rag_response_with_sources(
                request.query, context, knowledge_service, settings
            ),
            media_type="text/event-stream",
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


async def generate_rag_response_with_sources(
    query: str,
    context: RetrievedContext,
    knowledge_service: KnowledgeService,
    settings,
) -> AsyncIterator[str]:
    """包装 RAG 响应，先发送 sources 再发送回答"""
    # 1. 构建 sources 列表
    sources_list = []
    for chunk in context.chunks:
        kb_id = chunk.metadata.get("knowledge_base_id", "")
        kb = knowledge_service.get_knowledge_base(kb_id)
        kb_name = kb.name if kb else "未知知识库"

        sources_list.append(
            {
                "content": chunk.content[:500],  # 限制长度避免过长
                "score": round(chunk.score, 3) if chunk.score else 0,
                "source_file": chunk.metadata.get("source_file", "unknown"),
                "knowledge_base_id": kb_id,
                "knowledge_base_name": kb_name,
                "chunk_index": chunk.chunk_index,
                "doc_id": chunk.doc_id,
            }
        )

    # 发送 sources 信息
    sources_data = {"type": "sources", "sources": sources_list}
    yield f"data: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

    # 2. 流式输出 LLM 回答
    async for chunk in generate_rag_response(
        query, context.formatted_context, settings
    ):
        yield chunk


async def generate_no_context_response(query: str) -> AsyncIterator[str]:
    """生成无上下文时的响应"""
    chat_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(asyncio.get_running_loop().time())

    message = (
        f'抱歉，在知识库中没有找到与 "{query}" 相关的信息。\n\n'
        "可能的原因：\n"
        "1. 知识库中尚未上传相关文档\n"
        "2. 查询关键词与文档内容匹配度较低\n\n"
        "建议：\n"
        "- 尝试使用不同的关键词重新查询\n"
        "- 上传包含相关信息的文档到知识库"
    )

    chunk_data = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": "rag-agent",
        "choices": [{"index": 0, "delta": {"content": message}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": "rag-agent",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


async def generate_rag_response(
    query: str, context: str, settings
) -> AsyncIterator[str]:
    """使用 LLM 基于检索内容生成回答"""
    try:
        # 构建 RAG 提示词
        system_prompt = (
            "你是一位专业的知识库问答助手。请根据提供的参考资料回答用户的问题。\n\n"
            "回答要求：\n"
            "1. 基于参考资料内容回答，不要编造信息\n"
            "2. 如果参考资料不足以完全回答问题，请如实说明\n"
            "3. 适当引用来源信息，方便用户核实\n"
            "4. 回答要清晰、有条理"
        )

        user_prompt = f"参考资料：\n{context}\n\n用户问题：{query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 创建 OpenAI 客户端
        client = openai.AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 流式调用
        stream = await client.chat.completions.create(
            model=settings.dashscope_model,
            messages=messages,  # type: ignore
            stream=True,
            temperature=0.7,
            extra_body={"enable_thinking": False},
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunk_data = {
                    "id": chunk.id,
                    "object": chunk.object,
                    "created": chunk.created,
                    "model": "rag-agent",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.choices[0].delta.content},
                            "finish_reason": chunk.choices[0].finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # 结束标记
        final_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_running_loop().time()),
            "model": "rag-agent",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(asyncio.get_running_loop().time()),
            "model": "rag-agent",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n抱歉，生成回答时遇到错误：{str(e)}"},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/api/rag/documents", response_model=DocumentListResponse)
async def rag_list_documents(
    user_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
):
    """列出知识库中的文档"""
    try:
        knowledge_service = get_knowledge_service()
        documents = knowledge_service.list_documents(user_id, workspace_id)
        return DocumentListResponse(documents=documents, total=len(documents))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to list documents: {exc}"
        ) from exc


@app.delete("/api/rag/documents/{doc_id}", response_model=DeleteDocumentResponse)
async def rag_delete_document(doc_id: str):
    """删除知识库中的文档"""
    try:
        knowledge_service = get_knowledge_service()

        # 检查文档是否存在
        doc = knowledge_service.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        # 删除文档
        success = await knowledge_service.delete_document(doc_id)

        # 删除本地文件
        if success and doc:
            file_storage = get_file_storage()
            ext = "." + doc.filename.rsplit(".", 1)[-1] if "." in doc.filename else ""
            is_image = doc.file_type == DocumentType.IMAGE
            file_storage.delete_file(doc_id, ext, is_image)

        return DeleteDocumentResponse(
            success=success,
            message="Document deleted successfully"
            if success
            else "Failed to delete document",
            doc_id=doc_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete document: {exc}"
        ) from exc


@app.get("/api/rag/documents/{doc_id}", response_model=DocumentMetadata)
async def rag_get_document(doc_id: str):
    """获取文档详情"""
    try:
        knowledge_service = get_knowledge_service()
        doc = knowledge_service.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return doc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to get document: {exc}"
        ) from exc


@app.get("/api/rag/documents/{doc_id}/chunks", response_model=DocumentChunksResponse)
async def rag_get_document_chunks(doc_id: str):
    """获取文档的 chunks（分块内容）"""
    try:
        knowledge_service = get_knowledge_service()

        # 检查文档是否存在
        doc = knowledge_service.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        # 获取 chunks
        chunks_data = knowledge_service.get_document_chunks(doc_id)
        if chunks_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Chunks not found for document: {doc_id}. "
                "The document may have been uploaded before chunks storage was enabled.",
            )

        # 转换为响应格式
        chunks = [
            ChunkInfo(chunk_id=c["chunk_id"], content=c["content"])
            for c in chunks_data.get("chunks", [])
        ]

        return DocumentChunksResponse(
            doc_id=chunks_data["doc_id"],
            filename=chunks_data["filename"],
            total_chunks=chunks_data["total_chunks"],
            created_at=chunks_data["created_at"],
            chunks=chunks,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to get document chunks: {exc}"
        ) from exc
