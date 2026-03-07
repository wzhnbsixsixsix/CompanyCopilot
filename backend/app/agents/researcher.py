"""ResearcherAgent - 企业调研数据收集专家

基于 AgentScope ReActAgent 构建的专业商业调研智能体，
负责通过 firecrawl_search 工具并行收集企业8个维度的原始数据。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, view_text_file

from ..tools import firecrawl_search

if TYPE_CHECKING:
    from ..config import Settings


class ResearcherAgent:
    """企业调研数据收集专家

    角色定位：
    - 专业的商业调研专家，擅长从多个维度收集企业信息
    - 利用并行工具调用能力高效收集8个调研维度的原始数据
    - 遵循SKILL.md中定义的调研标准和数据质量要求
    """

    @staticmethod
    def build(settings: Settings) -> ReActAgent:
        """构建ResearcherAgent实例

        Args:
            settings: 应用配置，包含API密钥等信息

        Returns:
            配置完成的ReActAgent实例
        """
        # 创建工具包
        toolkit = Toolkit()

        # 注册文件读取工具（访问SKILL.md必需）
        toolkit.register_tool_function(view_text_file)

        # 注册firecrawl搜索工具，预设API密钥
        toolkit.register_tool_function(
            firecrawl_search,
            preset_kwargs={
                "api_url": settings.firecrawl_api_url,
                "api_key": settings.firecrawl_api_key,
            },
        )

        # 注册企业调研技能
        company_research_skill_dir = (
            Path(__file__).resolve().parent.parent.parent
            / "skills"
            / "company_research"
        )
        if company_research_skill_dir.exists():
            toolkit.register_agent_skill(str(company_research_skill_dir))

        # 注册快速尽职调查技能
        due_diligence_skill_dir = (
            Path(__file__).resolve().parent.parent.parent / "skills" / "due_diligence"
        )
        if due_diligence_skill_dir.exists():
            toolkit.register_agent_skill(str(due_diligence_skill_dir))

        # 创建OpenAI兼容的DashScope模型实例
        model = OpenAIChatModel(
            model_name=settings.dashscope_model,
            api_key=settings.dashscope_api_key,
            client_kwargs={
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            },
            generate_kwargs={
                "extra_body": {"enable_thinking": False},  # 禁用thinking模式
                "parallel_tool_calls": True,  # 启用并行工具调用
            },
        )

        # 创建ReActAgent
        return ReActAgent(
            name="ResearcherAgent",
            sys_prompt=(
                "你是一位专业的商业调研专家，具有丰富的企业背景调查经验。"
                "你拥有两种调研技能，请根据任务要求选择合适的技能文件：\n"
                "- company_research：全面的8维度企业背景调查\n"
                "- due_diligence：快速的产品维度尽职调查\n\n"
                "工作特点：\n"
                "- 技能选择：根据任务要求选择对应的技能文件进行调研\n"
                "- 系统性思维：按照技能文件中定义的维度有序开展调研工作\n"
                "- 效率优先：充分利用并行工具调用能力，同时执行多个搜索任务\n"
                "- 证据导向：所有信息必须有可靠来源，优先使用官方和权威资料\n"
                "- 客观中立：保持专业态度，如实记录发现的信息，不主观臆测\n\n"
                "请根据用户指定的调研类型，仔细阅读相应的技能文件以了解详细的工作规范和要求。"
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=toolkit,
            parallel_tool_calls=True,  # 在智能体层面启用并行工具调用
        )
