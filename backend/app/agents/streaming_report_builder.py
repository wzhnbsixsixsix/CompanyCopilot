"""StreamingReportBuilder - 流式报告构建器

负责实时处理增量数据，动态构建和更新报告内容。
支持部分数据处理和智能报告章节生成。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, AsyncIterator, Dict, Optional

import openai
from agentscope.message import Msg

from ..incremental_data import DataDimension, DataStatus, IncrementalCompanyData

if TYPE_CHECKING:
    from ..config import Settings


class StreamingReportBuilder:
    """流式报告构建器

    核心功能：
    - 接收增量数据更新
    - 实时生成报告章节
    - 智能合并新内容
    - 流式输出报告变化
    """

    def __init__(self, settings: Settings):
        """初始化流式报告构建器

        Args:
            settings: 应用配置
        """
        self.settings = settings
        self.client = openai.AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        # 维度到章节的映射
        self.dimension_to_section = {
            DataDimension.BASIC_INFO: "公司概览与基础信息",
            DataDimension.PRODUCTS: "产品与服务体系",
            DataDimension.MARKET: "市场表现与受众分析",
            DataDimension.PERSONNEL: "关键人员与组织架构",
            DataDimension.FINANCIALS: "融资历史与财务状况",
            DataDimension.TECH_STACK: "技术栈与数字化水平",
            DataDimension.NEWS: "近期动态与发展趋势",
            DataDimension.COMPETITORS: "竞争格局与市场地位",
        }

    async def stream_report_updates(
        self, company_data: IncrementalCompanyData
    ) -> AsyncIterator[str]:
        """流式处理公司数据更新并输出报告变化

        Args:
            company_data: 增量公司数据对象

        Yields:
            报告内容的增量更新
        """
        try:
            # 首先输出初始报告框架
            yield await self._generate_initial_framework(company_data)

            # 监听数据更新并实时生成内容
            last_update_time = 0
            processed_dimensions = set()

            while True:
                # 检查是否有新的已分析数据
                analyzed_dims = company_data.get_analyzed_dimensions()
                new_dimensions = [
                    dim for dim in analyzed_dims if dim not in processed_dimensions
                ]

                if new_dimensions:
                    # 为新维度生成报告章节
                    for dimension in new_dimensions:
                        try:
                            section_name = self.dimension_to_section.get(dimension)
                            if section_name:
                                # 流式生成章节内容
                                async for chunk in self._generate_section_content(
                                    company_data, dimension, section_name
                                ):
                                    yield chunk

                                processed_dimensions.add(dimension)
                        except Exception as e:
                            # 输出章节生成错误，但继续处理其他章节
                            error_content = f"\n## {section_name}\n\n*章节生成时发生错误: {str(e)}*\n\n"
                            yield error_content

                # 检查是否所有维度都已处理完成
                if len(processed_dimensions) >= len(company_data.dimensions):
                    # 生成最终总结
                    async for chunk in self._generate_final_summary(company_data):
                        yield chunk
                    break

                # 短暂等待，避免过度轮询
                await asyncio.sleep(0.5)

                # 超时保护：如果超过5分钟仍未完成，强制结束
                if time.time() - last_update_time > 300:
                    yield f"\n\n---\n\n**注意**: 部分数据收集超时，报告基于已获取数据生成。\n"
                    break

        except Exception as e:
            # 全局错误处理
            error_report = (
                f"\n\n**报告生成错误**: {str(e)}\n\n请检查网络连接和API配置后重试。"
            )
            yield error_report

    async def _generate_initial_framework(
        self, company_data: IncrementalCompanyData
    ) -> str:
        """生成初始报告框架

        Args:
            company_data: 公司数据

        Returns:
            初始报告框架
        """
        company_display = company_data.company_name or company_data.domain

        if company_data.mode == "quick":
            framework = f"""# {company_display} 快速尽职调查报告

*专注于产品与服务维度的快速评估*

---

## 🔍 调研进度

正在收集企业数据... 报告将随数据获取实时更新

"""
        else:
            framework = f"""# {company_display} 企业背景调查报告

*全面8维度企业调研分析*

---

## 🔍 调研进度

正在收集企业数据... 报告将随数据获取实时更新

"""

        return framework

    async def _generate_section_content(
        self,
        company_data: IncrementalCompanyData,
        dimension: DataDimension,
        section_name: str,
    ) -> AsyncIterator[str]:
        """流式生成报告章节内容

        Args:
            company_data: 公司数据
            dimension: 数据维度
            section_name: 章节名称

        Yields:
            章节内容的文本块
        """
        try:
            # 获取维度数据
            dim_data = company_data.dimensions.get(dimension)
            if not dim_data or dim_data.status != DataStatus.ANALYZED:
                return

            # 构建章节生成提示
            prompt = self._build_section_prompt(
                company_data, dimension, section_name, dim_data.analyzed_data
            )

            # 输出章节标题
            yield f"\n## {section_name}\n\n"

            # 流式生成章节内容
            messages = [
                {
                    "role": "system",
                    "content": "你是专业的商业报告撰写专家。根据提供的分析数据，生成清晰、结构化的报告章节。直接输出章节内容，无需额外说明。",
                },
                {"role": "user", "content": prompt},
            ]

            stream = await self.client.chat.completions.create(
                model=self.settings.dashscope_model,
                messages=messages,
                stream=True,
                temperature=0.7,
                extra_body={"enable_thinking": False},
            )

            section_content = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if self._is_valid_content(content):
                        section_content += content
                        yield content

            # 更新公司数据中的章节内容
            company_data.update_report_section(section_name, section_content.strip())

        except Exception as e:
            error_content = f"\n*章节生成错误: {str(e)}*\n\n"
            yield error_content

    def _build_section_prompt(
        self,
        company_data: IncrementalCompanyData,
        dimension: DataDimension,
        section_name: str,
        analyzed_data: str,
    ) -> str:
        """构建章节生成提示词

        Args:
            company_data: 公司数据
            dimension: 数据维度
            section_name: 章节名称
            analyzed_data: 已分析的数据

        Returns:
            章节生成提示词
        """
        company_name = company_data.company_name or company_data.domain

        # 获取已有的基础信息作为上下文
        context_info = ""
        if DataDimension.BASIC_INFO in company_data.dimensions:
            basic_data = company_data.dimensions[DataDimension.BASIC_INFO]
            if basic_data.analyzed_data:
                context_info = (
                    f"\n\n**企业基础信息参考**:\n{basic_data.analyzed_data[:500]}..."
                )

        prompt = f"""请为 {company_name} 生成 "{section_name}" 章节的报告内容。

**分析数据**:
{analyzed_data}

{context_info}

**要求**:
1. 基于提供的分析数据生成具体、客观的内容
2. 使用Markdown格式，包含适当的子标题和列表
3. 重点突出关键信息和风险点
4. 保持专业的商业报告语调
5. 如果数据不足，明确标注"待进一步核实"
6. 内容要具体详实，避免泛泛而谈

请直接生成章节内容，不要包含章节标题（已单独处理）。
"""

        return prompt

    async def _generate_final_summary(
        self, company_data: IncrementalCompanyData
    ) -> AsyncIterator[str]:
        """生成最终总结

        Args:
            company_data: 公司数据

        Yields:
            总结内容的文本块
        """
        try:
            yield "\n\n---\n\n## 📋 调研总结\n\n"

            # 构建总结提示
            available_sections = list(company_data.report_sections.keys())
            completion_rate = company_data.get_completion_rate()

            summary_prompt = f"""基于以上 {len(available_sections)} 个章节的调研结果，为 {company_data.company_name or company_data.domain} 提供简洁的总结评估。

**已完成章节**: {", ".join(available_sections)}
**完成度**: {completion_rate:.1%}

请提供：
1. **核心发现**（2-3个要点）
2. **主要风险**（如有）
3. **投资/合作建议**（简要）

保持客观、专业，基于已收集数据进行评估。"""

            messages = [
                {
                    "role": "system",
                    "content": "你是专业的商业分析师。基于调研报告内容，提供简洁的总结评估。",
                },
                {"role": "user", "content": summary_prompt},
            ]

            stream = await self.client.chat.completions.create(
                model=self.settings.dashscope_model,
                messages=messages,
                stream=True,
                temperature=0.7,
                extra_body={"enable_thinking": False},
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if self._is_valid_content(content):
                        yield content

            # 添加数据来源说明
            yield f"\n\n---\n\n*本报告基于公开信息整理，数据收集时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*"

        except Exception as e:
            error_content = f"\n\n**总结生成错误**: {str(e)}"
            yield error_content

    def _is_valid_content(self, content: str) -> bool:
        """验证内容是否有效

        Args:
            content: 内容片段

        Returns:
            是否为有效内容
        """
        if not content or not content.strip():
            return False

        # 过滤可能的JSON或工具调用
        content_stripped = content.strip()
        if content_stripped.startswith("{") or content_stripped.startswith("["):
            try:
                json.loads(content_stripped)
                return False
            except:
                pass

        return True
