"""CompilerAgent - 商业报告撰写专家

基于 AgentScope ReActAgent 构建的专业报告编写智能体，
负责将分析洞察格式化为结构化的Markdown背调报告。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel

if TYPE_CHECKING:
    from ..config import Settings


class CompilerAgent:
    """商业报告撰写专家

    角色定位：
    - 经验丰富的商业报告撰写专家，擅长将复杂分析结果转化为清晰易读的报告
    - 遵循标准8章节结构，确保报告的专业性和可读性
    - 输出结构化的Markdown格式报告
    """

    @staticmethod
    def build(settings: Settings) -> ReActAgent:
        """构建CompilerAgent实例

        Args:
            settings: 应用配置，包含API密钥等信息

        Returns:
            配置完成的ReActAgent实例
        """
        # 创建OpenAI兼容的DashScope模型实例
        model = OpenAIChatModel(
            model_name=settings.dashscope_model,
            api_key=settings.dashscope_api_key,
            client_kwargs={
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            },
            generate_kwargs={
                "extra_body": {"enable_thinking": False},  # 禁用thinking模式
                "parallel_tool_calls": False,  # 纯文本生成，无需工具调用
            },
        )

        # 创建ReActAgent（无工具，专注于报告编写）
        return ReActAgent(
            name="CompilerAgent",
            sys_prompt=(
                "你是一位专业的商业报告撰写专家，具有丰富的企业背景调查报告编写经验。"
                "你的职责是将分析师提供的深度洞察整合为结构化、专业的背调报告。\n\n"
                "报告撰写能力：\n"
                "- 结构化表达：按照标准8章节架构组织内容\n"
                "- 信息整合：将散乱的分析要点整合为逻辑清晰的叙述\n"
                "- 专业用词：使用准确的商业术语和行业表达\n"
                "- 重点突出：合理安排信息层次，突出关键发现\n"
                "- 风险提示：清晰标注风险信号和注意事项\n\n"
                "报告结构要求：\n"
                "1. 公司概览与基础信息\n"
                "2. 产品与服务体系\n"
                "3. 市场表现与受众分析\n"
                "4. 关键人员与组织架构\n"
                "5. 融资历史与财务状况\n"
                "6. 技术栈与数字化水平\n"
                "7. 近期动态与发展趋势\n"
                "8. 竞争格局与市场地位\n\n"
                "格式要求：\n"
                "- 使用Markdown格式，包含清晰的标题层级\n"
                "- 每个章节包含具体的分析内容\n"
                "- 在报告末尾单独列出风险信号和信息来源\n"
                "- 保持客观中性的语调，基于事实进行描述\n"
                "- 对不确定信息明确标注\n\n"
                "你将接收到分析师提供的企业分析洞察，请据此撰写完整的背调报告。"
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            # 不配置toolkit，专注于报告编写任务
        )
