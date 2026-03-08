"""Knowledge Service - 知识库服务核心

基于 AgentScope RAG 组件实现知识库的创建、文档处理和检索功能。
使用 Qdrant 作为向量存储，DashScope qwen3-vl-embedding 作为嵌入模型（2560维）。
支持多知识库管理。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from agentscope.embedding import DashScopeMultiModalEmbedding
from agentscope.rag import (
    Document,
    DocMetadata,
    PDFReader,
    QdrantStore,
    SimpleKnowledge,
    TextReader,
)
from qdrant_client import AsyncQdrantClient

from .schemas import (
    DocumentChunk,
    DocumentMetadata,
    DocumentStatus,
    DocumentType,
    KnowledgeBase,
    RetrievedContext,
)

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)


# 文件扩展名到文档类型的映射
EXTENSION_TO_TYPE: dict[str, DocumentType] = {
    ".pdf": DocumentType.PDF,
    ".txt": DocumentType.TXT,
    ".md": DocumentType.MD,
    ".docx": DocumentType.DOCX,
    ".xlsx": DocumentType.XLSX,
    ".pptx": DocumentType.PPTX,
    ".csv": DocumentType.CSV,
    ".jpg": DocumentType.IMAGE,
    ".jpeg": DocumentType.IMAGE,
    ".png": DocumentType.IMAGE,
    ".webp": DocumentType.IMAGE,
}

# 默认知识库ID
DEFAULT_KB_ID = "default"
DEFAULT_KB_NAME = "默认知识库"

# Embedding 批处理配置（DashScope 限制：40 QPS）
EMBEDDING_BATCH_SIZE = 10  # 每批处理的文档数量
EMBEDDING_BATCH_DELAY = 0.3  # 批次之间的延迟（秒），确保不超过速率限制


class KnowledgeService:
    """知识库服务

    负责：
    - 多知识库管理（创建、列表、删除）
    - 文档处理和向量化
    - 语义检索
    - 文档元数据管理
    """

    def __init__(
        self,
        settings: "Settings",
        qdrant_path: Optional[str] = None,
    ):
        """初始化知识库服务

        Args:
            settings: 应用配置
            qdrant_path: Qdrant 数据持久化路径，默认为 backend/qdrant_data
        """
        self.settings = settings

        # 设置 Qdrant 持久化路径
        if qdrant_path is None:
            qdrant_path = str(
                Path(__file__).resolve().parent.parent.parent / "qdrant_data"
            )
        self.qdrant_path = qdrant_path

        # 确保目录存在
        os.makedirs(self.qdrant_path, exist_ok=True)

        # 元数据持久化路径
        self._metadata_path = Path(self.qdrant_path) / "metadata"
        self._metadata_path.mkdir(exist_ok=True)

        # Chunks 存储路径
        self._chunks_path = Path(self.qdrant_path) / "chunks"
        self._chunks_path.mkdir(exist_ok=True)

        # Qdrant 本地持久化目录
        self._vector_store_path = Path(self.qdrant_path) / "vector_store"
        self._vector_store_path.mkdir(exist_ok=True)

        # 知识库缓存（知识库ID -> SimpleKnowledge 实例）
        self._knowledge_instances: dict[str, SimpleKnowledge] = {}

        # 已确认可用的知识库索引，避免重复检查和重建
        self._ready_knowledge_bases: set[str] = set()

        # 嵌入模型（共享一个实例）
        self._embedding_model: Optional[DashScopeMultiModalEmbedding] = None

        # Qdrant 客户端（共享一个实例，避免文件锁冲突）
        self._qdrant_client: Optional[AsyncQdrantClient] = None

        # 知识库元数据存储
        self._knowledge_bases: dict[str, KnowledgeBase] = {}

        # 文档元数据存储（doc_id -> metadata）
        self._documents: dict[str, DocumentMetadata] = {}

        # 加载已有的元数据
        self._load_metadata()

    def _get_embedding_model(self) -> DashScopeMultiModalEmbedding:
        """获取嵌入模型（延迟初始化，共享实例）"""
        if self._embedding_model is None:
            logger.info("Initializing embedding model (qwen3-vl-embedding)...")
            self._embedding_model = DashScopeMultiModalEmbedding(
                api_key=self.settings.dashscope_api_key,
                model_name="qwen3-vl-embedding",
                dimensions=2560,
            )
        return self._embedding_model  # type: ignore[return-value]

    def _get_qdrant_client(self) -> AsyncQdrantClient:
        """获取 Qdrant 客户端（延迟初始化，共享实例）

        所有知识库共享同一个客户端实例，避免本地文件存储的锁冲突。
        每个知识库使用不同的 collection 名称来区分数据。
        """
        if self._qdrant_client is None:
            logger.info(
                f"Initializing shared Qdrant client at {self._vector_store_path}"
            )
            self._qdrant_client = AsyncQdrantClient(
                path=str(self._vector_store_path),
                check_compatibility=False,
            )
        return self._qdrant_client

    def _get_collection_name(self, kb_id: str) -> str:
        """获取知识库对应的 Qdrant collection 名称"""
        return f"kb_{kb_id}"

    def _get_knowledge_instance(self, kb_id: str) -> SimpleKnowledge:
        """获取或创建知识库实例"""
        if kb_id not in self._knowledge_instances:
            logger.info(f"Creating knowledge instance for kb_id: {kb_id}")

            # Create QdrantStore with in-memory location (temporary)
            # We'll override _client with our shared singleton client
            embedding_store = QdrantStore(
                location=":memory:",  # Temporary, will be replaced with shared client
                collection_name=self._get_collection_name(kb_id),
                dimensions=2560,
            )
            # Override with shared client for local file persistence
            # This avoids file lock conflicts when multiple KBs access the same storage
            embedding_store._client = self._get_qdrant_client()

            knowledge = SimpleKnowledge(
                embedding_model=self._get_embedding_model(),
                embedding_store=embedding_store,
            )

            self._knowledge_instances[kb_id] = knowledge

        return self._knowledge_instances[kb_id]

    def _build_documents_from_chunk_data(
        self,
        doc_id: str,
        chunks_data: dict,
    ) -> list[Document]:
        """从已保存的 chunk 数据重建 Document 列表。"""
        chunks = chunks_data.get("chunks", [])
        total_chunks = chunks_data.get("total_chunks", len(chunks))
        documents: list[Document] = []

        for chunk in sorted(chunks, key=lambda item: item.get("chunk_id", 0)):
            content = str(chunk.get("content", ""))
            chunk_id = int(chunk.get("chunk_id", 0))
            documents.append(
                Document(
                    metadata=DocMetadata(
                        content={"type": "text", "text": content},
                        doc_id=doc_id,
                        chunk_id=chunk_id,
                        total_chunks=total_chunks,
                    )
                )
            )

        return documents

    def _build_documents_for_reindex(
        self,
        kb_id: str,
    ) -> list[Document]:
        """根据本地元数据和 chunks 文件重建指定知识库的文档索引。"""
        documents: list[Document] = []
        kb_documents = self.list_documents(knowledge_base_id=kb_id)

        for doc in kb_documents:
            if doc.file_type == DocumentType.IMAGE:
                if not doc.image_description:
                    logger.warning(
                        "Skip rebuilding image document without description: %s",
                        doc.id,
                    )
                    continue

                documents.append(
                    Document(
                        metadata=DocMetadata(
                            content={"type": "text", "text": doc.image_description},
                            doc_id=doc.id,
                            chunk_id=0,
                            total_chunks=1,
                        )
                    )
                )
                continue

            chunks_data = self.get_document_chunks(doc.id)
            if chunks_data is None:
                logger.warning(
                    "Skip rebuilding document without chunks file: %s (%s)",
                    doc.id,
                    doc.filename,
                )
                continue

            documents.extend(self._build_documents_from_chunk_data(doc.id, chunks_data))

        return documents

    def _get_expected_point_count(self, kb_id: str) -> int:
        """根据元数据计算知识库应有的向量点数量。"""
        return sum(
            max(doc.chunk_count, 0)
            for doc in self.list_documents(knowledge_base_id=kb_id)
            if doc.status == DocumentStatus.COMPLETED
        )

    async def _ensure_knowledge_instance(self, kb_id: str) -> SimpleKnowledge:
        """确保知识库实例和持久化索引可用。"""
        knowledge = self._get_knowledge_instance(kb_id)

        if kb_id in self._ready_knowledge_bases:
            return knowledge

        client = knowledge.embedding_store.get_client()
        collection_name = self._get_collection_name(kb_id)
        expected_point_count = self._get_expected_point_count(kb_id)
        collection_exists = await client.collection_exists(collection_name)

        if collection_exists:
            count_result = await client.count(
                collection_name=collection_name,
                exact=True,
            )
            actual_point_count = count_result.count

            if actual_point_count != expected_point_count:
                logger.warning(
                    "Qdrant collection count mismatch for kb %s: expected=%s actual=%s. Rebuilding collection.",
                    kb_id,
                    expected_point_count,
                    actual_point_count,
                )
                await client.delete_collection(collection_name=collection_name)
                collection_exists = False

        if not collection_exists:
            rebuild_documents = self._build_documents_for_reindex(kb_id)
            if rebuild_documents:
                logger.info(
                    "Rebuilding persistent Qdrant collection for kb %s with %s chunks",
                    kb_id,
                    len(rebuild_documents),
                )
                await self._add_documents_in_batches(
                    knowledge=knowledge,
                    documents=rebuild_documents,
                    kb_id=kb_id,
                    operation="collection rebuild",
                )
            else:
                logger.info("No existing chunks to rebuild for kb %s", kb_id)

        self._ready_knowledge_bases.add(kb_id)
        return knowledge

    @staticmethod
    def _is_retryable_embedding_error(exc: Exception) -> bool:
        """判断是否为可重试的嵌入服务错误。"""
        message = str(exc)
        retryable_markers = [
            "502",
            "503",
            "504",
            "Bad Gateway",
            "Gateway Timeout",
            "Service Unavailable",
            "ConnectionError",
            "ReadTimeout",
        ]
        return any(marker in message for marker in retryable_markers)

    async def _add_documents_with_retry(
        self,
        knowledge: SimpleKnowledge,
        documents: list[Document],
        kb_id: str,
        operation: str,
        max_attempts: int = 3,
    ) -> None:
        """向知识库写入文档，并对 embedding 上游短暂故障做重试。"""
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                await knowledge.add_documents(documents)
                return
            except Exception as exc:
                last_exc = exc
                if attempt >= max_attempts or not self._is_retryable_embedding_error(
                    exc
                ):
                    raise

                backoff_seconds = attempt * 2
                logger.warning(
                    "Embedding write failed for kb %s during %s (attempt %s/%s): %s. Retrying in %ss.",
                    kb_id,
                    operation,
                    attempt,
                    max_attempts,
                    exc,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)

        if last_exc is not None:
            raise last_exc

    async def _add_documents_in_batches(
        self,
        knowledge: SimpleKnowledge,
        documents: list[Document],
        kb_id: str,
        operation: str,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        batch_delay: float = EMBEDDING_BATCH_DELAY,
    ) -> None:
        """分批向知识库写入文档，控制请求速率避免触发限流。

        Args:
            knowledge: 知识库实例
            documents: 待写入的文档列表
            kb_id: 知识库 ID
            operation: 操作描述（用于日志）
            batch_size: 每批文档数量
            batch_delay: 批次之间的延迟（秒）
        """
        total = len(documents)
        if total == 0:
            return

        total_batches = (total + batch_size - 1) // batch_size

        # 分批处理
        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(
                "Processing embedding batch %d/%d (%d documents) for kb %s",
                batch_num,
                total_batches,
                len(batch),
                kb_id,
            )

            await self._add_documents_with_retry(
                knowledge,
                batch,
                kb_id,
                f"{operation} batch {batch_num}/{total_batches}",
            )

            # 非最后一批时添加延迟，避免触发速率限制
            if i + batch_size < total:
                await asyncio.sleep(batch_delay)

        logger.info(
            "Completed all %d batches for kb %s during %s",
            total_batches,
            kb_id,
            operation,
        )

    def _load_metadata(self) -> None:
        """从磁盘加载元数据"""
        # 加载知识库元数据
        kb_file = self._metadata_path / "knowledge_bases.json"
        if kb_file.exists():
            try:
                with open(kb_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for kb_data in data:
                        # 解析日期
                        kb_data["created_at"] = datetime.fromisoformat(
                            kb_data["created_at"]
                        )
                        kb_data["updated_at"] = datetime.fromisoformat(
                            kb_data["updated_at"]
                        )
                        kb = KnowledgeBase(**kb_data)
                        self._knowledge_bases[kb.id] = kb
                logger.info(f"Loaded {len(self._knowledge_bases)} knowledge bases")
            except Exception as e:
                logger.error(f"Failed to load knowledge bases metadata: {e}")

        # 加载文档元数据
        docs_file = self._metadata_path / "documents.json"
        if docs_file.exists():
            try:
                with open(docs_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for doc_data in data:
                        doc_data["created_at"] = datetime.fromisoformat(
                            doc_data["created_at"]
                        )
                        doc_data["updated_at"] = datetime.fromisoformat(
                            doc_data["updated_at"]
                        )
                        doc_data["file_type"] = DocumentType(doc_data["file_type"])
                        doc_data["status"] = DocumentStatus(doc_data["status"])
                        doc = DocumentMetadata(**doc_data)
                        self._documents[doc.id] = doc
                logger.info(f"Loaded {len(self._documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load documents metadata: {e}")

        # 确保默认知识库存在
        if DEFAULT_KB_ID not in self._knowledge_bases:
            self._knowledge_bases[DEFAULT_KB_ID] = KnowledgeBase(
                id=DEFAULT_KB_ID,
                name=DEFAULT_KB_NAME,
                description="系统默认知识库，用于存储未指定知识库的文档",
            )
            self._save_metadata()

    def _save_metadata(self) -> None:
        """保存元数据到磁盘"""
        # 保存知识库元数据
        kb_file = self._metadata_path / "knowledge_bases.json"
        kb_data = [
            {
                **kb.model_dump(),
                "created_at": kb.created_at.isoformat(),
                "updated_at": kb.updated_at.isoformat(),
            }
            for kb in self._knowledge_bases.values()
        ]
        with open(kb_file, "w", encoding="utf-8") as f:
            json.dump(kb_data, f, ensure_ascii=False, indent=2)

        # 保存文档元数据
        docs_file = self._metadata_path / "documents.json"
        docs_data = [
            {
                **doc.model_dump(),
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
                "file_type": doc.file_type.value,
                "status": doc.status.value,
            }
            for doc in self._documents.values()
        ]
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)

    def _save_chunks(
        self,
        doc_id: str,
        filename: str,
        chunks: list[dict],
    ) -> None:
        """保存文档的 chunks 到 JSON 文件

        Args:
            doc_id: 文档ID
            filename: 原始文件名
            chunks: chunk 列表，每个 chunk 包含 chunk_id 和 content
        """
        chunks_file = self._chunks_path / f"{doc_id}.json"
        chunks_data = {
            "doc_id": doc_id,
            "filename": filename,
            "total_chunks": len(chunks),
            "created_at": datetime.now().isoformat(),
            "chunks": chunks,
        }
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(chunks)} chunks for document {doc_id}")

    def get_document_chunks(self, doc_id: str) -> Optional[dict]:
        """获取文档的 chunks

        Args:
            doc_id: 文档ID

        Returns:
            包含 chunks 信息的字典，如果文档不存在则返回 None
        """
        chunks_file = self._chunks_path / f"{doc_id}.json"
        if not chunks_file.exists():
            return None

        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chunks for {doc_id}: {e}")
            return None

    def _delete_chunks(self, doc_id: str) -> bool:
        """删除文档的 chunks 文件

        Args:
            doc_id: 文档ID

        Returns:
            是否成功删除
        """
        chunks_file = self._chunks_path / f"{doc_id}.json"
        if chunks_file.exists():
            try:
                chunks_file.unlink()
                logger.info(f"Deleted chunks file for document {doc_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete chunks file for {doc_id}: {e}")
                return False
        return True  # 文件不存在也算成功

    # ============ 知识库管理 ============

    def create_knowledge_base(
        self,
        name: str,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> KnowledgeBase:
        """创建知识库"""
        kb_id = uuid.uuid4().hex[:16]
        now = datetime.now()

        kb = KnowledgeBase(
            id=kb_id,
            name=name,
            description=description,
            user_id=user_id,
            workspace_id=workspace_id,
            document_count=0,
            created_at=now,
            updated_at=now,
        )

        self._knowledge_bases[kb_id] = kb
        self._save_metadata()

        logger.info(f"Created knowledge base: {kb_id} ({name})")
        return kb

    def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
        """获取知识库"""
        return self._knowledge_bases.get(kb_id)

    def list_knowledge_bases(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> list[KnowledgeBase]:
        """列出知识库"""
        kbs = list(self._knowledge_bases.values())

        if user_id:
            kbs = [kb for kb in kbs if kb.user_id == user_id or kb.id == DEFAULT_KB_ID]
        if workspace_id:
            kbs = [
                kb
                for kb in kbs
                if kb.workspace_id == workspace_id or kb.id == DEFAULT_KB_ID
            ]

        # 按创建时间倒序，但默认知识库置顶
        kbs.sort(
            key=lambda x: (x.id != DEFAULT_KB_ID, -x.created_at.timestamp()),
        )
        return kbs

    def update_knowledge_base(
        self,
        kb_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[KnowledgeBase]:
        """更新知识库"""
        kb = self._knowledge_bases.get(kb_id)
        if kb is None:
            return None

        # 创建新的 KnowledgeBase 对象（因为 Pydantic 模型是不可变的）
        update_data = kb.model_dump()
        if name is not None:
            update_data["name"] = name
        if description is not None:
            update_data["description"] = description
        update_data["updated_at"] = datetime.now()

        updated_kb = KnowledgeBase(**update_data)
        self._knowledge_bases[kb_id] = updated_kb
        self._save_metadata()

        logger.info(f"Updated knowledge base: {kb_id}")
        return updated_kb

    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库"""
        if kb_id == DEFAULT_KB_ID:
            logger.warning("Cannot delete default knowledge base")
            return False

        if kb_id not in self._knowledge_bases:
            return False

        # 删除该知识库的所有文档
        docs_to_delete = [
            doc_id
            for doc_id, doc in self._documents.items()
            if doc.knowledge_base_id == kb_id
        ]
        for doc_id in docs_to_delete:
            await self.delete_document(doc_id)

        # 删除知识库实例
        if kb_id in self._knowledge_instances:
            knowledge = self._knowledge_instances[kb_id]
            try:
                client = knowledge.embedding_store.get_client()
                await client.delete_collection(
                    collection_name=self._get_collection_name(kb_id)
                )
            except Exception as e:
                logger.warning(f"Failed to delete Qdrant collection for {kb_id}: {e}")
            del self._knowledge_instances[kb_id]
        self._ready_knowledge_bases.discard(kb_id)

        # 删除元数据
        del self._knowledge_bases[kb_id]
        self._save_metadata()

        logger.info(f"Deleted knowledge base: {kb_id}")
        return True

    # ============ 文档管理 ============

    @staticmethod
    def get_document_type(filename: str) -> Optional[DocumentType]:
        """根据文件名获取文档类型"""
        ext = Path(filename).suffix.lower()
        return EXTENSION_TO_TYPE.get(ext)

    @staticmethod
    def generate_doc_id(content: bytes, filename: str) -> str:
        """生成文档唯一ID"""
        hash_input = f"{filename}:{len(content)}:{hashlib.md5(content).hexdigest()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    async def process_document(
        self,
        content: bytes,
        filename: str,
        knowledge_base_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> DocumentMetadata:
        """处理上传的文档

        Args:
            content: 文件内容（字节）
            filename: 原始文件名
            knowledge_base_id: 目标知识库ID（不指定则使用默认知识库）
            user_id: 用户ID
            workspace_id: 工作空间ID
            description: 文档描述

        Returns:
            文档元数据

        Raises:
            ValueError: 不支持的文件类型或知识库不存在
        """
        # 检测文件类型
        doc_type = self.get_document_type(filename)
        if doc_type is None:
            raise ValueError(f"Unsupported file type: {filename}")

        # 确定目标知识库
        if knowledge_base_id is None:
            knowledge_base_id = DEFAULT_KB_ID
        elif knowledge_base_id not in self._knowledge_bases:
            raise ValueError(f"Knowledge base not found: {knowledge_base_id}")

        # 生成文档ID
        doc_id = self.generate_doc_id(content, filename)

        # 检查是否已存在
        if doc_id in self._documents:
            existing = self._documents[doc_id]
            if existing.status == DocumentStatus.COMPLETED:
                logger.info(f"Document already processed: {doc_id}")
                return existing

        # 创建元数据
        metadata = DocumentMetadata(
            id=doc_id,
            filename=filename,
            file_type=doc_type,
            file_size=len(content),
            status=DocumentStatus.PROCESSING,
            user_id=user_id,
            workspace_id=workspace_id,
            knowledge_base_id=knowledge_base_id,
            description=description,
        )
        self._documents[doc_id] = metadata

        try:
            # 根据类型处理文档
            if doc_type == DocumentType.IMAGE:
                # 图片类型需要特殊处理（使用 VL 模型）
                documents = await self._process_image(content, filename, doc_id)
            else:
                documents = await self._process_text_document(
                    content, filename, doc_id, doc_type
                )

            # 注意：不在 doc.metadata 中添加自定义字段（如 knowledge_base_id），
            # 因为 AgentScope 的 QdrantStore 检索时会用 DocMetadata(**payload) 重建对象，
            # 而 DocMetadata 不接受额外参数。知识库 ID 通过我们搜索的 collection 隐式关联。

            # 添加到对应知识库
            if documents:
                knowledge = self._get_knowledge_instance(knowledge_base_id)
                await self._add_documents_in_batches(
                    knowledge=knowledge,
                    documents=documents,
                    kb_id=knowledge_base_id,
                    operation=f"upload {filename}",
                )
                metadata.chunk_count = len(documents)

            metadata.status = DocumentStatus.COMPLETED
            metadata.updated_at = datetime.now()

            # 更新知识库文档计数
            kb = self._knowledge_bases.get(knowledge_base_id)
            if kb:
                update_data = kb.model_dump()
                update_data["document_count"] = kb.document_count + 1
                update_data["updated_at"] = datetime.now()
                self._knowledge_bases[knowledge_base_id] = KnowledgeBase(**update_data)

            self._save_metadata()
            self._ready_knowledge_bases.discard(knowledge_base_id)

            logger.info(
                f"Document processed successfully: {filename}, "
                f"kb: {knowledge_base_id}, chunks: {metadata.chunk_count}"
            )

        except Exception as e:
            metadata.status = DocumentStatus.FAILED
            metadata.error_message = str(e)
            metadata.updated_at = datetime.now()
            self._save_metadata()
            logger.error(f"Failed to process document {filename}: {e}")
            raise

        return metadata

    async def _process_text_document(
        self,
        content: bytes,
        filename: str,
        doc_id: str,
        doc_type: DocumentType,
    ) -> list[Document]:
        """处理文本类文档"""
        # 保存临时文件
        temp_dir = Path(self.qdrant_path) / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / f"{doc_id}_{filename}"

        try:
            with open(temp_path, "wb") as f:
                f.write(content)

            if doc_type == DocumentType.PDF:
                reader = PDFReader(chunk_size=512, split_by="paragraph")
                documents = await reader(pdf_path=str(temp_path))
            elif doc_type in (DocumentType.TXT, DocumentType.MD, DocumentType.CSV):
                # 使用 TextReader 处理纯文本
                text_content = content.decode("utf-8", errors="ignore")
                reader = TextReader(chunk_size=512, split_by="paragraph")
                documents = await reader(text=text_content)
            elif doc_type == DocumentType.DOCX:
                from agentscope.rag import WordReader

                reader = WordReader(chunk_size=512, split_by="paragraph")
                documents = await reader(word_path=str(temp_path))
            elif doc_type == DocumentType.XLSX:
                from agentscope.rag import ExcelReader

                reader = ExcelReader()
                documents = await reader(excel_path=str(temp_path))
            elif doc_type == DocumentType.PPTX:
                from agentscope.rag import PowerPointReader

                reader = PowerPointReader(chunk_size=512)
                documents = await reader(ppt_path=str(temp_path))
            else:
                raise ValueError(f"Unsupported document type: {doc_type}")

            # 更新每个文档的 doc_id 为我们生成的 ID（Reader 会生成自己的 doc_id）
            # 这样检索时可以通过 doc_id 关联到我们的 _documents 字典中的元数据
            for doc in documents:
                doc.metadata.doc_id = doc_id

            # 保存 chunks 到 JSON 文件，以便后续查看
            chunks_data = []
            for doc in documents:
                content_data = doc.metadata.content
                if isinstance(content_data, dict):
                    text = content_data.get("text", str(content_data))
                else:
                    text = str(content_data)
                chunks_data.append(
                    {
                        "chunk_id": doc.metadata.chunk_id,
                        "content": text,
                    }
                )
            self._save_chunks(doc_id, filename, chunks_data)

            # 注意：不要在 doc.metadata 中添加自定义字段，因为 AgentScope 的
            # QdrantStore 在检索时会用 DocMetadata(**payload) 重建对象，
            # 而 DocMetadata 是 dataclass，不接受额外的关键字参数。
            # 自定义元数据应该存储在我们自己的 _documents 字典中。

            return documents

        finally:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()

    async def _process_image(
        self,
        content: bytes,
        filename: str,
        doc_id: str,
    ) -> list[Document]:
        """处理图片文档（占位，实际由 ImageProcessor 处理）"""
        raise NotImplementedError(
            "Image processing requires ImageProcessor. "
            "Use ImageProcessor.process_image() instead."
        )

    # ============ 检索功能 ============

    async def retrieve(
        self,
        query: str,
        knowledge_base_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.3,
    ) -> RetrievedContext:
        """检索相关文档

        Args:
            query: 查询文本
            knowledge_base_id: 知识库ID（不指定则搜索所有知识库）
            limit: 返回结果数量上限
            score_threshold: 相关性分数阈值

        Returns:
            检索到的上下文
        """
        all_chunks: list[DocumentChunk] = []

        # 确定要搜索的知识库
        if knowledge_base_id:
            kb_ids = [knowledge_base_id]
        else:
            kb_ids = list(self._knowledge_bases.keys())

        # 从每个知识库检索
        for kb_id in kb_ids:
            try:
                knowledge = await self._ensure_knowledge_instance(kb_id)
                documents = await knowledge.retrieve(
                    query=query,
                    limit=limit,
                    score_threshold=score_threshold,
                )

                # 转换为 DocumentChunk
                for doc in documents:
                    content_data = doc.metadata.content
                    if isinstance(content_data, dict):
                        text = content_data.get("text", str(content_data))
                    else:
                        text = str(content_data)

                    # 从我们自己的 _documents 中查找源文件名
                    source_doc_id = doc.metadata.doc_id
                    source_file = "unknown"
                    if source_doc_id in self._documents:
                        source_file = self._documents[source_doc_id].filename

                    chunk = DocumentChunk(
                        id=(
                            doc.id
                            if hasattr(doc, "id")
                            else f"{doc.metadata.doc_id}_{doc.metadata.chunk_id}"
                        ),
                        doc_id=doc.metadata.doc_id,
                        chunk_index=doc.metadata.chunk_id,
                        content=text,
                        metadata={
                            "source_file": source_file,
                            "total_chunks": doc.metadata.total_chunks,
                            "knowledge_base_id": kb_id,
                        },
                        score=doc.score,
                    )
                    all_chunks.append(chunk)

            except Exception as e:
                logger.warning(f"Failed to retrieve from kb {kb_id}: {e}")
                continue

        # 按分数排序并限制数量
        all_chunks.sort(key=lambda x: x.score or 0, reverse=True)
        all_chunks = all_chunks[:limit]

        # 格式化上下文
        formatted_parts = []
        for i, chunk in enumerate(all_chunks, 1):
            source = chunk.metadata.get("source_file", "unknown")
            kb_id = chunk.metadata.get("knowledge_base_id", "unknown")
            kb_name = self._knowledge_bases.get(
                kb_id, KnowledgeBase(id=kb_id, name=kb_id)
            ).name
            formatted_parts.append(
                f"[{i}] (来源: {source}, 知识库: {kb_name}, 相关度: {chunk.score:.2f})\n{chunk.content}"
            )

        formatted_context = "\n\n---\n\n".join(formatted_parts)

        return RetrievedContext(
            chunks=all_chunks,
            formatted_context=formatted_context,
        )

    # ============ 文档查询 ============

    def get_document(self, doc_id: str) -> Optional[DocumentMetadata]:
        """获取文档元数据"""
        return self._documents.get(doc_id)

    def list_documents(
        self,
        knowledge_base_id: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> list[DocumentMetadata]:
        """列出文档"""
        docs = list(self._documents.values())

        if knowledge_base_id:
            docs = [d for d in docs if d.knowledge_base_id == knowledge_base_id]
        if user_id:
            docs = [d for d in docs if d.user_id == user_id]
        if workspace_id:
            docs = [d for d in docs if d.workspace_id == workspace_id]

        # 按创建时间倒序
        docs.sort(key=lambda x: x.created_at, reverse=True)
        return docs

    async def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        if doc_id not in self._documents:
            return False

        doc = self._documents[doc_id]
        kb_id = doc.knowledge_base_id or DEFAULT_KB_ID

        # 从元数据中移除
        del self._documents[doc_id]

        # 更新知识库文档计数
        kb = self._knowledge_bases.get(kb_id)
        if kb and kb.document_count > 0:
            update_data = kb.model_dump()
            update_data["document_count"] = kb.document_count - 1
            update_data["updated_at"] = datetime.now()
            self._knowledge_bases[kb_id] = KnowledgeBase(**update_data)

        self._save_metadata()
        self._ready_knowledge_bases.discard(kb_id)

        # 删除 chunks 文件
        self._delete_chunks(doc_id)

        # 从 Qdrant 中删除相关向量
        try:
            if kb_id in self._knowledge_instances:
                knowledge = self._knowledge_instances[kb_id]
                client = knowledge.embedding_store.get_client()
                collection_name = self._get_collection_name(kb_id)

                # 使用 Qdrant 的 filter 删除所有 doc_id 匹配的 points
                from qdrant_client.models import Filter, FieldCondition, MatchValue

                await client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchValue(value=doc_id),
                            )
                        ]
                    ),
                )
                logger.info(f"Deleted vectors for document {doc_id} from Qdrant")
        except Exception as e:
            # 向量删除失败不影响整体删除操作
            logger.warning(f"Failed to delete vectors for {doc_id}: {e}")

        logger.info(f"Document deleted: {doc_id}")
        return True

    def get_retrieve_tool(self, kb_id: Optional[str] = None):
        """获取检索工具函数（用于注册到 Agent Toolkit）"""
        if kb_id is None:
            kb_id = DEFAULT_KB_ID

        async def retrieve_knowledge(
            query: str,
            limit: int = 5,
            score_threshold: float | None = None,
            **kwargs,
        ):
            knowledge = await self._ensure_knowledge_instance(kb_id)
            return await knowledge.retrieve_knowledge(
                query=query,
                limit=limit,
                score_threshold=score_threshold,
                **kwargs,
            )

        retrieve_knowledge.__name__ = "retrieve_knowledge"
        return retrieve_knowledge
