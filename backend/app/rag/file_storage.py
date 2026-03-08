"""File Storage - 文件存储管理

管理上传文件的本地存储，包括临时文件和持久化文件。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FileStorage:
    """文件存储管理器

    负责：
    - 管理上传文件的本地存储
    - 提供文件的读取和删除
    - 清理临时文件
    """

    def __init__(self, base_path: Optional[str] = None):
        """初始化文件存储

        Args:
            base_path: 存储基础路径，默认为 backend/uploaded_files
        """
        if base_path is None:
            base_path = str(
                Path(__file__).resolve().parent.parent.parent / "uploaded_files"
            )
        self.base_path = Path(base_path)

        # 创建必要的目录
        self.documents_path = self.base_path / "documents"
        self.images_path = self.base_path / "images"
        self.temp_path = self.base_path / "temp"

        for path in [self.documents_path, self.images_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(f"File storage initialized at: {self.base_path}")

    def save_file(
        self,
        content: bytes,
        filename: str,
        doc_id: str,
        is_image: bool = False,
    ) -> Path:
        """保存文件

        Args:
            content: 文件内容
            filename: 原始文件名
            doc_id: 文档ID
            is_image: 是否为图片

        Returns:
            保存后的文件路径
        """
        # 确定存储目录
        if is_image:
            target_dir = self.images_path
        else:
            target_dir = self.documents_path

        # 构建文件名（使用 doc_id 作为前缀避免冲突）
        ext = Path(filename).suffix
        safe_filename = f"{doc_id}{ext}"
        file_path = target_dir / safe_filename

        # 写入文件
        with open(file_path, "wb") as f:
            f.write(content)

        logger.debug(f"File saved: {file_path}")
        return file_path

    def get_file(self, doc_id: str, ext: str, is_image: bool = False) -> Optional[Path]:
        """获取文件路径

        Args:
            doc_id: 文档ID
            ext: 文件扩展名
            is_image: 是否为图片

        Returns:
            文件路径，如果不存在返回 None
        """
        if is_image:
            target_dir = self.images_path
        else:
            target_dir = self.documents_path

        file_path = target_dir / f"{doc_id}{ext}"
        if file_path.exists():
            return file_path
        return None

    def read_file(
        self, doc_id: str, ext: str, is_image: bool = False
    ) -> Optional[bytes]:
        """读取文件内容

        Args:
            doc_id: 文档ID
            ext: 文件扩展名
            is_image: 是否为图片

        Returns:
            文件内容，如果不存在返回 None
        """
        file_path = self.get_file(doc_id, ext, is_image)
        if file_path:
            with open(file_path, "rb") as f:
                return f.read()
        return None

    def delete_file(self, doc_id: str, ext: str, is_image: bool = False) -> bool:
        """删除文件

        Args:
            doc_id: 文档ID
            ext: 文件扩展名
            is_image: 是否为图片

        Returns:
            是否删除成功
        """
        file_path = self.get_file(doc_id, ext, is_image)
        if file_path and file_path.exists():
            file_path.unlink()
            logger.debug(f"File deleted: {file_path}")
            return True
        return False

    def save_temp_file(self, content: bytes, filename: str) -> Path:
        """保存临时文件

        Args:
            content: 文件内容
            filename: 文件名

        Returns:
            临时文件路径
        """
        file_path = self.temp_path / filename
        with open(file_path, "wb") as f:
            f.write(content)
        return file_path

    def delete_temp_file(self, filename: str) -> bool:
        """删除临时文件

        Args:
            filename: 文件名

        Returns:
            是否删除成功
        """
        file_path = self.temp_path / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def cleanup_temp(self) -> int:
        """清理所有临时文件

        Returns:
            删除的文件数量
        """
        count = 0
        for file_path in self.temp_path.iterdir():
            if file_path.is_file():
                file_path.unlink()
                count += 1
        logger.info(f"Cleaned up {count} temp files")
        return count

    def get_storage_stats(self) -> dict:
        """获取存储统计信息

        Returns:
            统计信息字典
        """

        def count_files(path: Path) -> tuple[int, int]:
            """统计目录中的文件数和总大小"""
            files = list(path.iterdir()) if path.exists() else []
            count = len([f for f in files if f.is_file()])
            size = sum(f.stat().st_size for f in files if f.is_file())
            return count, size

        doc_count, doc_size = count_files(self.documents_path)
        img_count, img_size = count_files(self.images_path)
        temp_count, temp_size = count_files(self.temp_path)

        return {
            "documents": {"count": doc_count, "size_bytes": doc_size},
            "images": {"count": img_count, "size_bytes": img_size},
            "temp": {"count": temp_count, "size_bytes": temp_size},
            "total": {
                "count": doc_count + img_count,
                "size_bytes": doc_size + img_size,
            },
        }
