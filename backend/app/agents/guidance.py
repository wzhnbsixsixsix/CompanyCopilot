"""GuidanceAgent - 意图识别与用户引导智能体

基于 AgentScope ReActAgent 构建的用户引导智能体，
专门负责识别用户的企业调研意图并引导使用正确的命令。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel

if TYPE_CHECKING:
    from ..config import Settings


class GuidanceAgent:
    """意图识别与用户引导智能体

    角色定位：
    - CompanyCopilot的前端接待员，负责理解用户意图
    - 当用户询问企业相关信息时，引导使用 /research <域名> 命令
    - 回答关于系统功能的一般性问题
    - 不使用任何工具，纯LLM对话，确保响应速度
    """

    @staticmethod
    def build(settings: Settings) -> ReActAgent:
        """构建GuidanceAgent实例

        Args:
            settings: 应用配置，包含API密钥等信息

        Returns:
            配置完成的ReActAgent实例（无工具）
        """
        # 创建OpenAI兼容的DashScope模型实例（无工具，纯LLM对话）
        model = OpenAIChatModel(
            model_name=settings.dashscope_model,
            api_key=settings.dashscope_api_key,
            client_kwargs={
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            },
            generate_kwargs={
                "extra_body": {"enable_thinking": False},  # 禁用thinking模式
                "parallel_tool_calls": False,  # 不需要工具调用
            },
        )

        # 创建ReActAgent（无工具包）
        return ReActAgent(
            name="GuidanceAgent",
            sys_prompt=(
                "你是 CompanyCopilot 智能助手，专门帮助用户进行企业背景调查和尽职调查。\n\n"
                "**核心功能**：\n"
                "- 企业背景调查：输入 `/research 企业域名`（如 `/research apple.com`）启动全面的8维度企业调研\n"
                "- 快速尽调：输入 `/due-diligence 公司名称` 进行基础背景调查\n\n"
                "**工作原则**：\n"
                "1. **意图识别**：当用户询问某个公司/企业的信息时，主动引导使用 `/research` 命令\n"
                "2. **功能介绍**：向用户介绍系统的企业调研能力，包括公司概况、产品服务、市场地位、人员结构、财务状况、技术栈、最新动态、竞争对手等8个维度\n"
                "3. **格式规范**：强调域名格式的重要性（如 apple.com，不是 Apple 或 apple）\n"
                "4. **简洁高效**：作为引导员，回答要简洁明了，不进行实际的企业调研工作\n\n"
                "**典型对话示例**：\n"
                '用户："帮我了解一下苹果公司的情况"\n'
                '你："我来帮您调研苹果公司！请使用以下命令启动全面的企业背景调查：\n\n'
                "`/research apple.com`\n\n"
                '系统将从8个维度为您提供详细的企业调研报告，包括公司概况、产品服务、市场地位、财务状况等。"\n\n'
                "记住：你不直接提供企业信息，而是引导用户使用正确的命令来获取专业的调研报告。"
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=None,  # 无工具包，纯LLM对话
            parallel_tool_calls=False,  # 不使用工具调用
        )
