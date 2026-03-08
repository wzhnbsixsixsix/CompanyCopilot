"""Image Processor - 图片理解服务

使用 qwen VL 模型对图片进行 OCR 和内容理解，
将图片内容转换为文本描述后存入知识库。
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx
from agentscope.rag import Document

from .schemas import DocumentMetadata, DocumentStatus, DocumentType

if TYPE_CHECKING:
    from ..config import Settings
    from .knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)

# 支持的图片 MIME 类型
IMAGE_MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class ImageProcessor:
    """图片理解处理器

    使用 DashScope qwen VL 模型提取图片中的文字和内容描述，
    然后将描述文本向量化存入知识库。
    """

    def __init__(
        self,
        settings: "Settings",
        knowledge_service: "KnowledgeService",
    ):
        """初始化图片处理器

        Args:
            settings: 应用配置
            knowledge_service: 知识库服务实例
        """
        self.settings = settings
        self.knowledge_service = knowledge_service

        # VL 模型配置
        self.vl_model = "qwen3.5-flash-2026-02-23"
        self.api_url = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )

    async def process_image(
        self,
        content: bytes,
        filename: str,
        knowledge_base_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> DocumentMetadata:
        """处理图片文档

        1. 使用 VL 模型提取图片内容
        2. 将提取的文本向量化
        3. 存入知识库

        Args:
            content: 图片字节内容
            filename: 原始文件名
            knowledge_base_id: 目标知识库ID
            user_id: 用户ID
            workspace_id: 工作空间ID
            description: 用户提供的描述

        Returns:
            文档元数据
        """
        # 检测图片类型
        ext = Path(filename).suffix.lower()
        if ext not in IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported image type: {filename}")

        mime_type = IMAGE_MIME_TYPES[ext]

        # 确定目标知识库
        if knowledge_base_id is None:
            knowledge_base_id = "default"

        # 生成文档ID
        doc_id = self.knowledge_service.generate_doc_id(content, filename)

        # 检查是否已存在
        existing = self.knowledge_service.get_document(doc_id)
        if existing and existing.status == DocumentStatus.COMPLETED:
            logger.info(f"Image already processed: {doc_id}")
            return existing

        # 创建元数据
        metadata = DocumentMetadata(
            id=doc_id,
            filename=filename,
            file_type=DocumentType.IMAGE,
            file_size=len(content),
            status=DocumentStatus.PROCESSING,
            user_id=user_id,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            description=description,
        )
        self.knowledge_service._documents[doc_id] = metadata

        try:
            # 使用 VL 模型提取图片内容
            image_description = await self._extract_image_content(
                content, mime_type, filename
            )

            # 更新元数据
            metadata.image_description = image_description

            # 创建 Document 对象
            documents = self._create_documents_from_description(
                image_description, doc_id, filename
            )

            # 添加知识库 ID 到每个文档的元数据
            for doc in documents:
                doc.metadata["knowledge_base_id"] = knowledge_base_id

            # 添加到对应知识库
            if documents:
                knowledge = self.knowledge_service._get_knowledge_instance(
                    knowledge_base_id
                )
                await self.knowledge_service._add_documents_in_batches(
                    knowledge=knowledge,
                    documents=documents,
                    kb_id=knowledge_base_id,
                    operation=f"image upload {filename}",
                )
                metadata.chunk_count = len(documents)

            metadata.status = DocumentStatus.COMPLETED

            # 更新知识库文档计数
            kb = self.knowledge_service._knowledge_bases.get(knowledge_base_id)
            if kb:
                from .schemas import KnowledgeBase

                update_data = kb.model_dump()
                update_data["document_count"] = kb.document_count + 1
                from datetime import datetime

                update_data["updated_at"] = datetime.now()
                self.knowledge_service._knowledge_bases[knowledge_base_id] = (
                    KnowledgeBase(**update_data)
                )

            self.knowledge_service._save_metadata()
            self.knowledge_service._ready_knowledge_bases.discard(knowledge_base_id)

            logger.info(
                f"Image processed successfully: {filename}, "
                f"kb: {knowledge_base_id}, "
                f"description length: {len(image_description)}"
            )

        except Exception as e:
            metadata.status = DocumentStatus.FAILED
            metadata.error_message = str(e)
            logger.error(f"Failed to process image {filename}: {e}")
            raise

        return metadata

    async def _extract_image_content(
        self,
        content: bytes,
        mime_type: str,
        filename: str,
    ) -> str:
        """使用 VL 模型提取图片内容

        Args:
            content: 图片字节内容
            mime_type: MIME 类型
            filename: 文件名

        Returns:
            图片内容描述文本
        """
        # 将图片转为 base64
        base64_image = base64.b64encode(content).decode("utf-8")
        image_url = f"data:{mime_type};base64,{base64_image}"

        # 构建请求
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的文档分析助手。请仔细分析图片内容，提取所有可见的文字信息，"
                    "并对图片中的图表、表格、图形等视觉元素进行详细描述。"
                    "输出应该包含：\n"
                    "1. 图片中的所有文字内容（如有）\n"
                    "2. 图表/表格的数据描述（如有）\n"
                    "3. 图片的整体内容概述"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": f"请分析这张图片（文件名：{filename}）的内容，提取所有文字信息并描述图片内容。",
                    },
                ],
            },
        ]

        # 调用 DashScope API
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.settings.dashscope_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.vl_model,
                    "messages": messages,
                    "max_tokens": 4096,
                },
            )

            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(
                    f"VL model API error: {response.status_code} - {error_text}"
                )

            result = response.json()
            response_content: str = result["choices"][0]["message"]["content"]

            # 处理可能的思考模式输出
            if isinstance(response_content, list):
                # 如果是列表，提取文本部分
                text_parts: list[str] = []
                for item in response_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                response_content = "\n".join(text_parts)

            return str(response_content)

    def _create_documents_from_description(
        self,
        description: str,
        doc_id: str,
        filename: str,
    ) -> list[Document]:
        """从图片描述创建 Document 对象

        Args:
            description: 图片内容描述
            doc_id: 文档ID
            filename: 文件名

        Returns:
            Document 对象列表
        """
        # 对于图片，通常描述不会太长，作为单个文档块
        # 如果描述很长，可以考虑分块
        from agentscope.rag import DocMetadata

        doc_metadata = DocMetadata(
            content={"type": "text", "text": description},
            doc_id=doc_id,
            chunk_id=0,
            total_chunks=1,
        )

        # 添加额外元数据
        doc_metadata["source_file"] = filename
        doc_metadata["source_doc_id"] = doc_id
        doc_metadata["content_type"] = "image_description"

        document = Document(metadata=doc_metadata)

        return [document]

    async def get_image_description(self, doc_id: str) -> Optional[str]:
        """获取图片的描述文本

        Args:
            doc_id: 文档ID

        Returns:
            图片描述，如果不存在返回 None
        """
        metadata = self.knowledge_service.get_document(doc_id)
        if metadata and metadata.file_type == DocumentType.IMAGE:
            return metadata.image_description
        return None
