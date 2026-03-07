"""CompanyResearchPipeline - 企业背调智能体流水线

实现三个专职智能体的顺序执行管道：ResearcherAgent → AnalystAgent → CompilerAgent
遵循 AgentScope 官方文档的 Agent 调用和消息传递模式。
同时支持流式和非流式两种执行模式。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from agentscope.message import Msg

from .agents import AnalystAgent, CompilerAgent, ResearcherAgent
from .agents.streaming_compiler import StreamingCompilerAgent
from .schemas import CompanyReport

if TYPE_CHECKING:
    from .config import Settings


class CompanyResearchPipeline:
    """企业背调智能体流水线

    协调三个专职智能体完成企业背景调查：
    1. ResearcherAgent: 并行收集8个维度的原始数据
    2. AnalystAgent: 深度分析数据并提炼关键洞察
    3. CompilerAgent: 格式化生成结构化Markdown报告

    支持流式和非流式两种执行模式。
    """

    def __init__(self, settings: Settings) -> None:
        """初始化流水线

        Args:
            settings: 应用配置，包含API密钥等信息
        """
        self.settings = settings
        self._researcher = None
        self._analyst = None
        self._compiler = None
        self._streaming_compiler = None

    @property
    def researcher(self):
        """懒加载ResearcherAgent"""
        if self._researcher is None:
            self._researcher = ResearcherAgent.build(self.settings)
        return self._researcher

    @property
    def analyst(self):
        """懒加载AnalystAgent"""
        if self._analyst is None:
            self._analyst = AnalystAgent.build(self.settings)
        return self._analyst

    @property
    def compiler(self):
        """懒加载CompilerAgent"""
        if self._compiler is None:
            self._compiler = CompilerAgent.build(self.settings)
        return self._compiler

    @property
    def streaming_compiler(self):
        """懒加载StreamingCompilerAgent"""
        if self._streaming_compiler is None:
            self._streaming_compiler = StreamingCompilerAgent(self.settings)
        return self._streaming_compiler

    async def run(self, domain: str, mode: str = "full") -> str:
        """执行企业背调流水线

        Args:
            domain: 目标公司域名，如 "apple.com"
            mode: 调研模式，"full" 为全面8维度调研，"quick" 为快速产品维度调研

        Returns:
            生成的Markdown格式背调报告

        Raises:
            ValueError: 域名格式无效或模式无效
            Exception: 流水线执行过程中的其他错误
        """
        if not domain or not domain.strip():
            raise ValueError("域名不能为空")

        if mode not in ("full", "quick"):
            raise ValueError("模式必须是 'full' 或 'quick'")

        domain = domain.strip()

        try:
            # 第一阶段：ResearcherAgent 收集原始数据
            # 按照官方文档，使用 Msg(name, content, role) 格式
            if mode == "full":
                # 全面8维度调研
                research_msg = Msg(
                    "user",
                    f"请对以下公司域名进行全面的背景调查：{domain}\\n\\n"
                    f"调查要求：\\n"
                    f"1. 按照你的调研技能文件中定义的8个维度进行系统性调研\\n"
                    f"2. 利用并行工具调用能力，同时执行多个搜索任务提高效率\\n"
                    f"3. 收集的所有信息必须标注可靠来源\\n"
                    f"4. 重点关注风险信号和负面信息\\n"
                    f"5. 整理输出所有原始数据，为下一步分析做准备",
                    "user",
                )
            else:  # mode == "quick"
                # 快速产品维度调研
                research_msg = Msg(
                    "user",
                    f"请对以下公司域名进行快速的尽职调查：{domain}\\n\\n"
                    f"调查要求：\\n"
                    f"1. 使用 due_diligence 技能文件，专注于产品与服务维度\\n"
                    f"2. 执行2次核心搜索：产品概述和定价信息\\n"
                    f"3. 利用并行工具调用提高效率\\n"
                    f"4. 收集的信息必须标注来源\\n"
                    f"5. 输出产品相关的原始数据，为快速评估做准备",
                    "user",
                )

            researcher_result = await self.researcher(research_msg)

            # 第二阶段：AnalystAgent 分析数据
            # 直接传递前一个智能体的输出作为输入
            analyst_result = await self.analyst(researcher_result)

            # 第三阶段：CompilerAgent 生成结构化报告
            # 使用 structured_model 参数强制结构化输出
            compiler_result = await self.compiler(
                analyst_result,
                structured_model=CompanyReport,  # 强制输出为CompanyReport格式
            )

            # 从metadata中提取结构化输出的报告内容
            if hasattr(compiler_result, "metadata") and compiler_result.metadata:
                report_content = compiler_result.metadata.get("report")
                if report_content:
                    return report_content

            # 如果structured_model失败，使用content内容作为备选
            if hasattr(compiler_result, "content"):
                return str(compiler_result.content)

            # 最终备选方案
            return f"# {domain} 背调报告\\n\\n报告生成过程中出现异常，请检查系统配置。"

        except Exception as e:
            error_msg = f"企业背调流水线执行失败: {str(e)}"
            # 返回错误信息而不是抛出异常，确保前端能正常处理
            return f"# {domain} 背调报告\\n\\n## 错误信息\\n\\n{error_msg}\\n\\n请检查网络连接和API配置后重试。"

    async def run_streaming(
        self, domain: str, mode: str = "full"
    ) -> AsyncIterator[str]:
        """执行企业背调流水线（流式输出）

        执行前两个阶段的智能体，然后流式输出最终报告内容。

        Args:
            domain: 目标公司域名，如 "apple.com"
            mode: 调研模式，"full" 为全面8维度调研，"quick" 为快速产品维度调研

        Yields:
            报告内容的文本块

        Raises:
            ValueError: 域名格式无效或模式无效
            Exception: 流水线执行过程中的其他错误
        """
        if not domain or not domain.strip():
            raise ValueError("域名不能为空")

        if mode not in ("full", "quick"):
            raise ValueError("模式必须是 'full' 或 'quick'")

        domain = domain.strip()

        try:
            # 第一阶段：ResearcherAgent 收集原始数据
            # 按照官方文档，使用 Msg(name, content, role) 格式
            if mode == "full":
                # 全面8维度调研
                research_msg = Msg(
                    "user",
                    f"请对以下公司域名进行全面的背景调查：{domain}\\n\\n"
                    f"调查要求：\\n"
                    f"1. 按照你的调研技能文件中定义的8个维度进行系统性调研\\n"
                    f"2. 利用并行工具调用能力，同时执行多个搜索任务提高效率\\n"
                    f"3. 收集的所有信息必须标注可靠来源\\n"
                    f"4. 重点关注风险信号和负面信息\\n"
                    f"5. 整理输出所有原始数据，为下一步分析做准备",
                    "user",
                )
            else:  # mode == "quick"
                # 快速产品维度调研
                research_msg = Msg(
                    "user",
                    f"请对以下公司域名进行快速的尽职调查：{domain}\\n\\n"
                    f"调查要求：\\n"
                    f"1. 使用 due_diligence 技能文件，专注于产品与服务维度\\n"
                    f"2. 执行2次核心搜索：产品概述和定价信息\\n"
                    f"3. 利用并行工具调用提高效率\\n"
                    f"4. 收集的信息必须标注来源\\n"
                    f"5. 输出产品相关的原始数据，为快速评估做准备",
                    "user",
                )

            # 执行前两个阶段（非流式）
            researcher_result = await self.researcher(research_msg)
            analyst_result = await self.analyst(researcher_result)

            # 第三阶段：流式生成报告
            async for chunk in self.streaming_compiler.stream_report(
                analyst_result, mode
            ):
                yield chunk

        except Exception as e:
            error_msg = f"企业背调流水线执行失败: {str(e)}"
            # 流式输出错误信息
            error_report = f"# {domain} 背调报告\\n\\n## 错误信息\\n\\n{error_msg}\\n\\n请检查网络连接和API配置后重试。"

            # 分块输出错误信息以保持流式特性
            chunk_size = 50
            for i in range(0, len(error_report), chunk_size):
                chunk = error_report[i : i + chunk_size]
                yield chunk
