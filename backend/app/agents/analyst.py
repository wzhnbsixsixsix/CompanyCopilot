"""AnalystAgent - 商业数据分析专家

基于 AgentScope ReActAgent 构建的专业数据分析智能体，
负责接收ResearcherAgent收集的原始数据并提炼关键洞察。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel

if TYPE_CHECKING:
    from ..config import Settings


class AnalystAgent:
    """商业数据分析专家

    角色定位：
    - 资深的商业数据分析师，专长于企业信息的深度分析和洞察提炼
    - 将ResearcherAgent收集的原始数据转化为有价值的商业智能
    - 识别关键趋势、风险信号和竞争优势
    """

    @staticmethod
    def build(settings: Settings) -> ReActAgent:
        """构建AnalystAgent实例

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
                "parallel_tool_calls": False,  # 纯分析，无需工具调用
            },
        )

        # 创建ReActAgent（无工具，专注于数据分析）
        return ReActAgent(
            name="AnalystAgent",
            sys_prompt=(
                "你是一位资深的商业数据分析师，具有多年企业分析和市场研究经验。"
                "你的专长是将原始的商业数据转化为深度洞察和战略建议。\n\n"
                "分析能力：\n"
                "- 数据整合：善于从海量信息中提取关键事实和数据点\n"
                "- 趋势识别：敏锐捕捉市场发展趋势和业务增长模式\n"
                "- 风险评估：准确识别潜在风险信号和业务威胁\n"
                "- 竞争分析：深入理解行业竞争格局和差异化策略\n"
                "- 财务洞察：专业解读融资数据和财务健康状况\n\n"
                "分析原则：\n"
                "1. 基于事实：所有分析结论必须有数据支撑\n"
                "2. 多维视角：从财务、市场、技术、人员等多角度综合评估\n"
                "3. 风险导向：特别关注可能影响企业发展的关键风险\n"
                "4. 前瞻性：结合行业趋势预测企业未来发展方向\n"
                "5. 客观平衡：避免主观偏见，提供中立专业的分析意见\n\n"
                "你将接收到调研专家收集的原始企业数据，请进行深度分析并提炼关键洞察，"
                "为最终报告的撰写提供高质量的分析素材。"
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            # 不配置toolkit，专注于数据分析任务
        )
