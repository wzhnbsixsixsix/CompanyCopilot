"""RAG Module - 基于 AgentScope 的知识库检索增强生成

提供文档上传、向量化存储和智能问答功能。
使用 Qdrant 作为向量数据库，DashScope 作为嵌入模型。
"""

from .schemas import (
    DocumentMetadata,
    UploadedDocument,
    RAGQueryRequest,
    RAGQueryResponse,
    DocumentChunk,
)
from .knowledge_service import KnowledgeService
from .image_processor import ImageProcessor
from .file_storage import FileStorage

__all__ = [
    # Schemas
    "DocumentMetadata",
    "UploadedDocument",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "DocumentChunk",
    # Services
    "KnowledgeService",
    "ImageProcessor",
    "FileStorage",
]
