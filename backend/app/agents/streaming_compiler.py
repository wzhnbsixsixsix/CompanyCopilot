"""StreamingCompilerAgent - 流式商业报告撰写专家

基于直接模型访问的流式报告编写智能体，
绕过 AgentScope 的 Agent 抽象层，直接使用模型的流式能力。
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, AsyncIterator

import openai
from agentscope.message import Msg

if TYPE_CHECKING:
    from ..config import Settings


class StreamingCompilerAgent:
    """流式商业报告撰写专家

    角色定位：
    - 直接使用模型流式能力生成报告内容
    - 过滤工具调用和中间输出，只返回报告正文
    - 支持实时流式输出，提升用户体验
    """

    def __init__(self, settings: Settings):
        """初始化流式编译器代理

        Args:
            settings: 应用配置，包含API密钥等信息
        """
        self.settings = settings
        self.client = openai.AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        self.system_prompt = (
            "你是一位专业的商业报告撰写专家，具有丰富的企业背景调查报告编写经验。"
            "你的职责是将分析师提供的深度洞察整合为结构化、专业的背调报告。\n\n"
            "报告撰写能力：\n"
            "- 结构化表达：按照标准章节架构组织内容\n"
            "- 信息整合：将散乱的分析要点整合为逻辑清晰的叙述\n"
            "- 专业用词：使用准确的商业术语和行业表达\n"
            "- 重点突出：合理安排信息层次，突出关键发现\n"
            "- 风险提示：清晰标注风险信号和注意事项\n\n"
            "格式要求：\n"
            "- 使用Markdown格式，包含清晰的标题层级\n"
            "- 每个章节包含具体的分析内容\n"
            "- 在报告末尾单独列出风险信号和信息来源\n"
            "- 保持客观中性的语调，基于事实进行描述\n"
            "- 对不确定信息明确标注\n\n"
            "重要说明：\n"
            "- 直接输出最终的Markdown报告内容\n"
            "- 不需要工具调用或结构化输出\n"
            "- 不需要解释或meta信息，只输出报告正文\n"
            "- 确保报告完整且格式正确"
        )

    async def stream_report(
        self, analyst_result: Msg, mode: str = "full"
    ) -> AsyncIterator[str]:
        """流式生成报告内容

        Args:
            analyst_result: 分析师的分析结果
            mode: 报告模式，"full" 为全面报告，"quick" 为快速报告

        Yields:
            报告内容的文本块
        """
        try:
            # 构建提示词
            if mode == "full":
                structure_prompt = (
                    "报告结构要求（8个章节）：\n"
                    "1. 公司概览与基础信息\n"
                    "2. 产品与服务体系\n"
                    "3. 市场表现与受众分析\n"
                    "4. 关键人员与组织架构\n"
                    "5. 融资历史与财务状况\n"
                    "6. 技术栈与数字化水平\n"
                    "7. 近期动态与发展趋势\n"
                    "8. 竞争格局与市场地位\n\n"
                )
            else:  # quick mode
                structure_prompt = (
                    "报告结构要求（产品重点）：\n"
                    "1. 公司基本信息\n"
                    "2. 产品与服务体系（详细分析）\n"
                    "3. 商业模式与定价策略\n"
                    "4. 市场定位与竞争优势\n\n"
                )

            user_content = (
                f"请基于以下分析师洞察，撰写专业的企业背调报告：\n\n"
                f"{analyst_result.content}\n\n"
                f"{structure_prompt}"
                f"请直接生成完整的Markdown格式报告，无需额外说明。"
            )

            # 构建消息
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ]

            # 调用流式API
            stream = await self.client.chat.completions.create(
                model=self.settings.dashscope_model,
                messages=messages,
                stream=True,
                temperature=0.7,
                extra_body={"enable_thinking": False},  # 禁用thinking模式
            )

            # 流式输出内容
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # 过滤掉可能的工具调用或meta信息
                    if self._is_report_content(content):
                        yield content

        except Exception as e:
            # 错误处理 - 返回错误信息作为报告内容
            error_report = (
                f"# 报告生成错误\n\n"
                f"生成报告时发生错误：{str(e)}\n\n"
                f"请检查网络连接和API配置后重试。"
            )
            # 将错误信息分块流式输出
            chunk_size = 50
            for i in range(0, len(error_report), chunk_size):
                chunk = error_report[i : i + chunk_size]
                yield chunk

    def _is_report_content(self, content: str) -> bool:
        """判断内容是否为报告正文

        Args:
            content: 内容片段

        Returns:
            是否为报告内容
        """
        # 简单的过滤逻辑，过滤掉明显的非报告内容
        if not content or not content.strip():
            return False

        # 过滤掉可能的JSON或工具调用格式
        content_stripped = content.strip()
        if content_stripped.startswith("{") or content_stripped.startswith("["):
            try:
                json.loads(content_stripped)
                return False  # 是JSON格式，可能是工具调用
            except:
                pass  # 不是有效JSON，继续检查

        # 过滤掉明显的meta信息
        meta_indicators = [
            "```json",
            "tool_call",
            "function_call",
            "thinking:",
            "analysis:",
            "```",
        ]

        content_lower = content.lower()
        for indicator in meta_indicators:
            if indicator in content_lower:
                return False

        return True

    async def generate_complete_report(
        self, analyst_result: Msg, mode: str = "full"
    ) -> str:
        """生成完整报告（非流式）

        这是备用方法，用于非流式场景

        Args:
            analyst_result: 分析师的分析结果
            mode: 报告模式

        Returns:
            完整的报告内容
        """
        report_chunks = []
        async for chunk in self.stream_report(analyst_result, mode):
            report_chunks.append(chunk)
        return "".join(report_chunks)
