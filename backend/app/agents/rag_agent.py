"""RAGAgent - 知识库问答专家

基于 AgentScope ReActAgent 构建的知识库问答智能体，
负责通过 retrieve_knowledge 工具检索知识库并生成回答。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, view_text_file

if TYPE_CHECKING:
    from ..config import Settings
    from ..rag import KnowledgeService


class RAGAgent:
    """知识库问答专家

    角色定位：
    - 专业的知识问答专家，擅长从知识库中检索信息并生成准确回答
    - 利用 retrieve_knowledge 工具进行语义检索
    - 遵循 SKILL.md 中定义的问答规范和回答格式
    """

    @staticmethod
    def build(
        settings: "Settings",
        knowledge_service: "KnowledgeService",
    ) -> ReActAgent:
        """构建 RAGAgent 实例

        Args:
            settings: 应用配置，包含 API 密钥等信息
            knowledge_service: 知识库服务实例

        Returns:
            配置完成的 ReActAgent 实例
        """
        # 创建工具包
        toolkit = Toolkit()

        # 注册文件读取工具（访问 SKILL.md 必需）
        toolkit.register_tool_function(view_text_file)

        # 注册知识库检索工具
        toolkit.register_tool_function(
            knowledge_service.get_retrieve_tool(),
            func_description=(
                "用于从知识库中检索与给定查询相关的文档的工具。"
                "当你需要查找用户上传的文档中的信息时使用此工具。"
                "返回最相关的文档块及其相关性分数。"
            ),
        )

        # 注册知识库问答技能
        knowledge_base_skill_dir = (
            Path(__file__).resolve().parent.parent.parent / "skills" / "knowledge_base"
        )
        if knowledge_base_skill_dir.exists():
            toolkit.register_agent_skill(str(knowledge_base_skill_dir))

        # 创建 OpenAI 兼容的 DashScope 模型实例
        model = OpenAIChatModel(
            model_name=settings.dashscope_model,
            api_key=settings.dashscope_api_key,
            client_kwargs={
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            },
            generate_kwargs={
                "extra_body": {"enable_thinking": False},
            },
        )

        # 创建 ReActAgent
        return ReActAgent(
            name="RAGAgent",
            sys_prompt=(
                "你是一位专业的知识库问答专家，擅长从用户上传的文档中检索信息并提供准确的回答。\n\n"
                "工作特点：\n"
                "- 使用 retrieve_knowledge 工具从知识库中检索相关信息\n"
                "- 基于检索到的文档内容生成回答，不凭空编造\n"
                "- 明确标注信息来源，方便用户核实\n"
                "- 如果检索结果不足以回答问题，如实告知用户\n\n"
                "请仔细阅读 knowledge_base 技能文件以了解详细的工作规范和回答格式要求。"
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=toolkit,
        )


class RAGAgentFactory:
    """RAGAgent 工厂类

    用于延迟创建 RAGAgent，确保在 KnowledgeService 初始化后才创建。
    """

    def __init__(self, settings: "Settings"):
        self.settings = settings
        self._agent: ReActAgent | None = None
        self._knowledge_service: "KnowledgeService" | None = None

    def set_knowledge_service(self, knowledge_service: "KnowledgeService") -> None:
        """设置知识库服务"""
        self._knowledge_service = knowledge_service
        # 重置 agent，下次获取时重新创建
        self._agent = None

    def get_agent(self) -> ReActAgent:
        """获取 RAGAgent 实例"""
        if self._knowledge_service is None:
            raise RuntimeError(
                "KnowledgeService not set. Call set_knowledge_service() first."
            )

        if self._agent is None:
            self._agent = RAGAgent.build(
                self.settings,
                self._knowledge_service,
            )

        return self._agent
