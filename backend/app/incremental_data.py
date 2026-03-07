"""增量数据传递机制

定义了用于实时流式报告生成的数据结构。
支持增量更新、部分数据处理和实时报告构建。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class DataDimension(Enum):
    """数据维度枚举"""

    BASIC_INFO = "basic_info"  # 公司基本信息
    PRODUCTS = "products"  # 产品与服务
    MARKET = "market"  # 市场表现
    PERSONNEL = "personnel"  # 人员架构
    FINANCIALS = "financials"  # 财务状况
    TECH_STACK = "tech_stack"  # 技术栈
    NEWS = "news"  # 近期动态
    COMPETITORS = "competitors"  # 竞争对手


class DataStatus(Enum):
    """数据状态"""

    PENDING = "pending"  # 待收集
    COLLECTING = "collecting"  # 收集中
    COLLECTED = "collected"  # 已收集
    ANALYZING = "analyzing"  # 分析中
    ANALYZED = "analyzed"  # 已分析
    ERROR = "error"  # 错误


@dataclass
class DimensionData:
    """维度数据"""

    dimension: DataDimension
    status: DataStatus = DataStatus.PENDING
    raw_data: str = ""  # 原始搜索数据
    analyzed_data: str = ""  # 分析后的数据
    sources: List[str] = field(default_factory=list)  # 数据源
    error_message: Optional[str] = None  # 错误信息
    timestamp: Optional[float] = None  # 更新时间


@dataclass
class IncrementalCompanyData:
    """增量公司数据

    用于在研究过程中逐步构建和更新公司信息。
    支持部分数据处理和实时报告生成。
    """

    domain: str
    company_name: Optional[str] = None
    mode: str = "full"  # "full" 或 "quick"

    # 各维度数据
    dimensions: Dict[DataDimension, DimensionData] = field(default_factory=dict)

    # 全局状态
    research_status: DataStatus = DataStatus.PENDING
    analysis_status: DataStatus = DataStatus.PENDING

    # 报告构建状态
    report_sections: Dict[str, str] = field(default_factory=dict)  # 已生成的报告章节
    current_report: str = ""  # 当前完整报告

    def __post_init__(self):
        """初始化维度数据"""
        if self.mode == "full":
            dimensions = list(DataDimension)
        else:  # quick mode
            dimensions = [DataDimension.BASIC_INFO, DataDimension.PRODUCTS]

        for dim in dimensions:
            if dim not in self.dimensions:
                self.dimensions[dim] = DimensionData(dimension=dim)

    def update_dimension(self, dimension: DataDimension, **kwargs) -> bool:
        """更新维度数据

        Args:
            dimension: 要更新的维度
            **kwargs: 要更新的字段

        Returns:
            是否有实质性更新（触发报告重新生成）
        """
        if dimension not in self.dimensions:
            self.dimensions[dimension] = DimensionData(dimension=dimension)

        dim_data = self.dimensions[dimension]
        has_update = False

        for key, value in kwargs.items():
            if hasattr(dim_data, key):
                old_value = getattr(dim_data, key)
                if old_value != value:
                    setattr(dim_data, key, value)
                    has_update = True

        return has_update

    def get_available_data(self) -> Dict[DataDimension, DimensionData]:
        """获取所有已收集的数据"""
        return {
            dim: data
            for dim, data in self.dimensions.items()
            if data.status in [DataStatus.COLLECTED, DataStatus.ANALYZED]
        }

    def get_completion_rate(self) -> float:
        """获取数据收集完成率"""
        if not self.dimensions:
            return 0.0

        completed = sum(
            1
            for data in self.dimensions.values()
            if data.status in [DataStatus.COLLECTED, DataStatus.ANALYZED]
        )

        return completed / len(self.dimensions)

    def is_ready_for_analysis(self) -> bool:
        """检查是否有足够数据进行分析"""
        # 至少需要基本信息和一个其他维度
        basic_ready = (
            DataDimension.BASIC_INFO in self.dimensions
            and self.dimensions[DataDimension.BASIC_INFO].status == DataStatus.COLLECTED
        )

        if not basic_ready:
            return False

        # 检查是否有其他维度数据
        other_ready = any(
            data.status == DataStatus.COLLECTED
            for dim, data in self.dimensions.items()
            if dim != DataDimension.BASIC_INFO
        )

        return other_ready

    def get_analyzed_dimensions(self) -> List[DataDimension]:
        """获取已分析的维度列表"""
        return [
            dim
            for dim, data in self.dimensions.items()
            if data.status == DataStatus.ANALYZED
        ]

    def update_report_section(self, section: str, content: str):
        """更新报告章节"""
        self.report_sections[section] = content
        self._rebuild_current_report()

    def _rebuild_current_report(self):
        """重建当前完整报告"""
        # 构建报告的基本结构
        report_parts = []

        # 标题
        company_display = self.company_name or self.domain
        report_parts.append(f"# {company_display} 企业背景调查报告")

        if self.mode == "quick":
            report_parts.append("\n*快速尽职调查报告 - 专注于产品与服务维度*\n")

        # 按顺序添加已完成的章节
        section_order = [
            "公司概览与基础信息",
            "产品与服务体系",
            "市场表现与受众分析",
            "关键人员与组织架构",
            "融资历史与财务状况",
            "技术栈与数字化水平",
            "近期动态与发展趋势",
            "竞争格局与市场地位",
        ]

        for section in section_order:
            if section in self.report_sections:
                content = self.report_sections[section]
                if content.strip():
                    report_parts.append(f"\n## {section}\n\n{content}")

        # 添加进度信息
        completion_rate = self.get_completion_rate()
        if completion_rate < 1.0:
            report_parts.append(
                f"\n---\n\n**调研进度**: {completion_rate:.1%} 完成 | 报告将随数据收集实时更新"
            )

        self.current_report = "\n".join(report_parts)
