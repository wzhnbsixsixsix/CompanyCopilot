from __future__ import annotations

from pathlib import Path
from typing import Any

import agentscope
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, view_text_file

from .config import Settings
from .schemas import DueDiligenceSummary
from .tools import demo_sleep_tool, firecrawl_search


class DueDiligenceAgentService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.toolkit = self._build_toolkit()
        self.agent = self._build_agent()

    def _build_toolkit(self) -> Toolkit:
        toolkit = Toolkit()

        # Built-in file tool for agent skill file access.
        toolkit.register_tool_function(view_text_file)

        # Register Firecrawl with preset secrets so API keys are hidden in schema.
        toolkit.register_tool_function(
            firecrawl_search,
            preset_kwargs={
                "api_url": self.settings.firecrawl_api_url,
                "api_key": self.settings.firecrawl_api_key,
            },
            group_name="basic",
        )

        # Async tool used to validate parallel tool calls behavior.
        toolkit.register_tool_function(demo_sleep_tool)

        skill_dir = Path(__file__).resolve().parent.parent / "skills" / "due_diligence"
        if skill_dir.exists():
            toolkit.register_agent_skill(str(skill_dir))

        return toolkit

    def _build_agent(self) -> ReActAgent:
        studio_url = self._get_studio_url()
        if studio_url:
            agentscope.init(studio_url=studio_url, project="CompanyCopilot")

        model = OpenAIChatModel(
            model_name=self.settings.dashscope_model,
            api_key=self.settings.dashscope_api_key,
            client_kwargs={
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            },
            generate_kwargs={
                "extra_body": {"enable_thinking": False},  # 禁用thinking模式
                "parallel_tool_calls": True,
            },
        )

        return ReActAgent(
            name="DueDiligenceAgent",
            sys_prompt=(
                "You are CompanyCopilot due diligence assistant. "
                "Use tools to gather verifiable evidence and avoid hallucinations."
            ),
            model=model,
            formatter=OpenAIChatFormatter(),
            memory=InMemoryMemory(),
            toolkit=self.toolkit,
            parallel_tool_calls=True,
        )

    @staticmethod
    def _get_studio_url() -> str | None:
        import os

        studio_url = os.getenv("AGENTSCOPE_STUDIO_URL", "").strip()
        return studio_url or None

    async def run_due_diligence(
        self,
        company_name: str,
        user_prompt: str | None = None,
        structured: bool = True,
    ) -> dict[str, Any]:
        prompt = user_prompt or (
            f"Perform due diligence for {company_name}. "
            "Collect founding year, headquarters, core business, key executives, "
            "risk signals, and include source URLs."
        )

        response = await self.agent(
            Msg("user", prompt, "user"),
            structured_model=DueDiligenceSummary if structured else None,
        )

        return {
            "text": str(response.content),
            "metadata": response.metadata,
        }
