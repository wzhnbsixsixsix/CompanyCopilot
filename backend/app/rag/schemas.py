"""RAG Schemas - 数据模型定义

定义 RAG 模块使用的所有 Pydantic 数据模型。
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """支持的文档类型"""

    PDF = "pdf"
    TXT = "txt"
    MD = "md"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    CSV = "csv"
    IMAGE = "image"  # jpg, png, jpeg, webp


class DocumentStatus(str, Enum):
    """文档处理状态"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """文档元数据"""

    id: str = Field(..., description="文档唯一标识符")
    filename: str = Field(..., description="原始文件名")
    file_type: DocumentType = Field(..., description="文档类型")
    file_size: int = Field(..., description="文件大小（字节）")
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING, description="处理状态"
    )
    chunk_count: int = Field(default=0, description="分块数量")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")
    user_id: Optional[str] = Field(default=None, description="上传用户ID")
    workspace_id: Optional[str] = Field(default=None, description="工作空间ID")
    knowledge_base_id: Optional[str] = Field(default=None, description="所属知识库ID")
    description: Optional[str] = Field(default=None, description="文档描述")
    error_message: Optional[str] = Field(default=None, description="错误信息")

    # 图片文档特有字段
    image_description: Optional[str] = Field(
        default=None, description="图片内容描述（VL模型生成）"
    )


class DocumentChunk(BaseModel):
    """文档分块"""

    id: str = Field(..., description="分块唯一标识符")
    doc_id: str = Field(..., description="所属文档ID")
    chunk_index: int = Field(..., description="分块索引")
    content: str = Field(..., description="分块内容")
    metadata: dict = Field(default_factory=dict, description="额外元数据")
    score: Optional[float] = Field(default=None, description="相关性分数（检索时填充）")


class UploadedDocument(BaseModel):
    """上传文档响应"""

    metadata: DocumentMetadata
    message: str = Field(default="Document uploaded successfully")


class RAGQueryRequest(BaseModel):
    """RAG 查询请求"""

    query: str = Field(..., description="用户查询问题")
    limit: int = Field(default=5, ge=1, le=20, description="返回结果数量上限")
    score_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="相关性分数阈值"
    )
    user_id: Optional[str] = Field(default=None, description="用户ID（用于过滤）")
    workspace_id: Optional[str] = Field(
        default=None, description="工作空间ID（用于过滤）"
    )
    knowledge_base_id: Optional[str] = Field(
        default=None, description="知识库ID（不指定则搜索全部）"
    )


class RAGQueryResponse(BaseModel):
    """RAG 查询响应"""

    query: str = Field(..., description="原始查询")
    answer: str = Field(..., description="生成的答案")
    chunks: list[DocumentChunk] = Field(
        default_factory=list, description="检索到的相关文档块"
    )
    sources: list[str] = Field(default_factory=list, description="来源文件名列表")


class RetrievedContext(BaseModel):
    """检索到的上下文（用于内部传递）"""

    chunks: list[DocumentChunk]
    formatted_context: str = Field(..., description="格式化后的上下文文本")


# ============ API 请求/响应模型 ============


class DocumentListResponse(BaseModel):
    """文档列表响应"""

    documents: list[DocumentMetadata]
    total: int


class DeleteDocumentResponse(BaseModel):
    """删除文档响应"""

    success: bool
    message: str
    doc_id: str


# ============ 知识库模型 ============


class KnowledgeBase(BaseModel):
    """知识库"""

    id: str = Field(..., description="知识库唯一标识符")
    name: str = Field(..., description="知识库名称")
    description: Optional[str] = Field(default=None, description="知识库描述")
    user_id: Optional[str] = Field(default=None, description="所属用户ID")
    workspace_id: Optional[str] = Field(default=None, description="所属工作空间ID")
    document_count: int = Field(default=0, description="文档数量")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    updated_at: datetime = Field(default_factory=datetime.now, description="更新时间")


class CreateKnowledgeBaseRequest(BaseModel):
    """创建知识库请求"""

    name: str = Field(..., min_length=1, max_length=100, description="知识库名称")
    description: Optional[str] = Field(
        default=None, max_length=500, description="知识库描述"
    )
    user_id: Optional[str] = Field(default=None, description="用户ID")
    workspace_id: Optional[str] = Field(default=None, description="工作空间ID")


class UpdateKnowledgeBaseRequest(BaseModel):
    """更新知识库请求"""

    name: Optional[str] = Field(
        default=None, min_length=1, max_length=100, description="知识库名称"
    )
    description: Optional[str] = Field(default=None, max_length=500, description="描述")


class KnowledgeBaseListResponse(BaseModel):
    """知识库列表响应"""

    knowledge_bases: list[KnowledgeBase]
    total: int


class KnowledgeBaseDetailResponse(BaseModel):
    """知识库详情响应（包含文档列表）"""

    knowledge_base: KnowledgeBase
    documents: list[DocumentMetadata]


class DeleteKnowledgeBaseResponse(BaseModel):
    """删除知识库响应"""

    success: bool
    message: str
    kb_id: str


class ChunkInfo(BaseModel):
    """单个 Chunk 信息"""

    chunk_id: int = Field(..., description="分块ID")
    content: str = Field(..., description="分块内容")


class DocumentChunksResponse(BaseModel):
    """文档 Chunks 响应"""

    doc_id: str = Field(..., description="文档ID")
    filename: str = Field(..., description="原始文件名")
    total_chunks: int = Field(..., description="总分块数")
    created_at: str = Field(..., description="创建时间")
    chunks: list[ChunkInfo] = Field(..., description="分块列表")
