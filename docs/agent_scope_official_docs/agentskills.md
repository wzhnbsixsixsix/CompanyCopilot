Title: 智能体技能 - AgentScope

URL Source: https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html

Published Time: Thu, 12 Feb 2026 10:16:56 GMT

Markdown Content:
智能体技能 - AgentScope
===============
- [x] - [x] 

Hide navigation sidebar

 

Hide table of contents sidebar

 [Skip to content](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#furo-main-content)

Toggle site navigation sidebar

 

[AgentScope](https://doc.agentscope.io/zh_CN/index.html)

Toggle Light / Dark / Auto color theme

Toggle table of contents sidebar

 

[![Image 1: Logo](https://doc.agentscope.io/zh_CN/_static/logo.svg) AgentScope](https://doc.agentscope.io/zh_CN/index.html)

Version

Tutorial

*   [安装](https://doc.agentscope.io/zh_CN/tutorial/quickstart_installation.html)
*   [核心概念](https://doc.agentscope.io/zh_CN/tutorial/quickstart_key_concept.html)
*   [创建消息](https://doc.agentscope.io/zh_CN/tutorial/quickstart_message.html)
*   [创建 ReAct 智能体](https://doc.agentscope.io/zh_CN/tutorial/quickstart_agent.html)

Workflow

*   [Conversation](https://doc.agentscope.io/zh_CN/tutorial/workflow_conversation.html)
*   [Multi-Agent Debate](https://doc.agentscope.io/zh_CN/tutorial/workflow_multiagent_debate.html)
*   [Concurrent Agents](https://doc.agentscope.io/zh_CN/tutorial/workflow_concurrent_agents.html)
*   [Routing](https://doc.agentscope.io/zh_CN/tutorial/workflow_routing.html)
*   [Handoffs](https://doc.agentscope.io/zh_CN/tutorial/workflow_handoffs.html)

FAQ

*   [常见问题](https://doc.agentscope.io/zh_CN/tutorial/faq.html)

Model and Context

*   [模型](https://doc.agentscope.io/zh_CN/tutorial/task_model.html)
*   [提示词格式化](https://doc.agentscope.io/zh_CN/tutorial/task_prompt.html)
*   [Token 计数](https://doc.agentscope.io/zh_CN/tutorial/task_token.html)
*   [记忆](https://doc.agentscope.io/zh_CN/tutorial/task_memory.html)
*   [长期记忆](https://doc.agentscope.io/zh_CN/tutorial/task_long_term_memory.html)

Tool

*   [工具](https://doc.agentscope.io/zh_CN/tutorial/task_tool.html)
*   [MCP](https://doc.agentscope.io/zh_CN/tutorial/task_mcp.html)
*   [智能体技能](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#)

Agent

*   [智能体](https://doc.agentscope.io/zh_CN/tutorial/task_agent.html)
*   [状态/会话管理](https://doc.agentscope.io/zh_CN/tutorial/task_state.html)
*   [智能体钩子函数](https://doc.agentscope.io/zh_CN/tutorial/task_hook.html)
*   [中间件](https://doc.agentscope.io/zh_CN/tutorial/task_middleware.html)
*   [A2A 智能体](https://doc.agentscope.io/zh_CN/tutorial/task_a2a.html)
*   [实时智能体](https://doc.agentscope.io/zh_CN/tutorial/task_realtime.html)

Features

*   [管道 (Pipeline)](https://doc.agentscope.io/zh_CN/tutorial/task_pipeline.html)
*   [计划](https://doc.agentscope.io/zh_CN/tutorial/task_plan.html)
*   [RAG](https://doc.agentscope.io/zh_CN/tutorial/task_rag.html)
*   [AgentScope Studio](https://doc.agentscope.io/zh_CN/tutorial/task_studio.html)
*   [追踪](https://doc.agentscope.io/zh_CN/tutorial/task_tracing.html)
*   [智能体评测](https://doc.agentscope.io/zh_CN/tutorial/task_eval.html)
*   [OpenJudge 评估器](https://doc.agentscope.io/zh_CN/tutorial/task_eval_openjudge.html)
*   [嵌入(Embedding)](https://doc.agentscope.io/zh_CN/tutorial/task_embedding.html)
*   [TTS](https://doc.agentscope.io/zh_CN/tutorial/task_tts.html)
*   [Tuner](https://doc.agentscope.io/zh_CN/tutorial/task_tuner.html)

[Back to top](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#)

[View this page](https://doc.agentscope.io/zh_CN/_sources/tutorial/task_agent_skill.rst.txt "View this page")

Toggle Light / Dark / Auto color theme

Toggle table of contents sidebar

 

备注

[Go to the end](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#sphx-glr-download-tutorial-task-agent-skill-py) to download the full example code.

智能体技能[¶](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#agent-skill "Link to this heading")
===========================================================================================================

[智能体技能（Agent skill）](https://claude.com/blog/skills) 是 Anthropic 提出的一种提升智能体在特定任务上能力的方法。

AgentScope 通过 `Toolkit` 类提供了对智能体技能的内置支持，让开发者可以注册和管理智能体技能。

相关 API 如下：

`Toolkit` 类中的智能体技能 API[¶](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#id4 "Link to this table")| API | 描述 |
| --- | --- |
| `register_agent_skill` | 从指定目录注册智能体技能 |
| `remove_agent_skill` | 根据名称移除已注册的智能体技能 |
| `get_agent_skill_prompt` | 获取所有已注册智能体技能的提示词，可以附加到智能体的系统提示词中 |

本节将演示如何注册智能体技能并在 ReActAgent 类中使用它们。

import os

from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit

注册智能体技能[¶](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#id3 "Link to this heading")
-----------------------------------------------------------------------------------------------------

首先，我们需要准备一个智能体技能目录，该目录需要遵循 [Anthropic blog](https://claude.com/blog/skills) 中指定的要求。

备注

技能目录必须包含一个 `SKILL.md` 文件，其中包含 YAML 前置元数据和指令说明。

这里我们创建一个示例技能目录 `sample_skill`，包含以下文件：

---
name: sample_skill
description: 用于演示的示例智能体技能
---

# 示例技能
...

os.makedirs("sample_skill", exist_ok=True)
with open("sample_skill/SKILL.md", "w", encoding="utf-8") as f:
    f.write(
 """---
name: sample_skill
description: 用于演示的示例智能体技能
---

# 示例技能
...
""",
    )

然后，我们可以使用 `Toolkit` 类的 `register_agent_skill` API 注册技能。

toolkit = Toolkit()

toolkit.register_agent_skill("sample_skill")

之后，我们可以使用 `get_agent_skill_prompt` API 获取所有已注册智能体技能的提示词

agent_skill_prompt = toolkit.get_agent_skill_prompt()
print("智能体技能提示词:")
print(agent_skill_prompt)

智能体技能提示词:
# Agent Skills
The agent skills are a collection of folds of instructions, scripts, and resources that you can load dynamically to improve performance on specialized tasks. Each agent skill has a `SKILL.md` file in its folder that describes how to use the skill. If you want to use a skill, you MUST read its `SKILL.md` file carefully.
## sample_skill
用于演示的示例智能体技能
Check "sample_skill/SKILL.md" for how to use this skill

当然，我们也可以在创建 `Toolkit` 实例时自定义提示词模板。

custom_toolkit = Toolkit(
    # 向智能体/大语言模型介绍如何使用技能的指令
    agent_skill_instruction="<system-info>为你提供了一组技能，每个技能都在一个目录中，并由 SKILL.md 文件进行描述。</system-info>",
    # 用于格式化每个技能提示词的模板，必须包含 {name}、{description} 和 {dir} 字段
    agent_skill_template="- {name}(in directory '{dir}'): {description}",
)

custom_toolkit.register_agent_skill("sample_skill")
agent_skill_prompt = custom_toolkit.get_agent_skill_prompt()
print("自定义智能体技能提示词:")
print(agent_skill_prompt)

自定义智能体技能提示词:
<system-info>为你提供了一组技能，每个技能都在一个目录中，并由 SKILL.md 文件进行描述。</system-info>
- sample_skill(in directory 'sample_skill'): 用于演示的示例智能体技能

在 ReActAgent 中集成智能体技能[¶](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#reactagent "Link to this heading")
--------------------------------------------------------------------------------------------------------------------------

AgentScope 中的 ReActAgent 类会自动将智能体技能提示词附加到系统提示词中。

我们可以按如下方式创建一个带有已注册智能体技能的 ReAct 智能体：

重要

使用智能体技能时，智能体必须配备文本文件读取或 shell 命令工具，以便访问 SKILL.md 文件中的技能指令。

agent = ReActAgent(
    name="Friday",
    sys_prompt="你是一个名为 Friday 的智能助手。",
    model=DashScopeChatModel(
        model_name="qwen3-235b-a22b-instruct-2507",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    ),
    memory=InMemoryMemory(),
    formatter=DashScopeChatFormatter(),
    toolkit=toolkit,
)

print("带有智能体技能的系统提示词:")
print(agent.sys_prompt)

带有智能体技能的系统提示词:
你是一个名为 Friday 的智能助手。

# Agent Skills
The agent skills are a collection of folds of instructions, scripts, and resources that you can load dynamically to improve performance on specialized tasks. Each agent skill has a `SKILL.md` file in its folder that describes how to use the skill. If you want to use a skill, you MUST read its `SKILL.md` file carefully.
## sample_skill
用于演示的示例智能体技能
Check "sample_skill/SKILL.md" for how to use this skill

**Total running time of the script:** (0 minutes 0.004 seconds)

[`Download Jupyter notebook: task_agent_skill.ipynb`](https://doc.agentscope.io/zh_CN/_downloads/49581b9ec49c0c26280f5d25c406d988/task_agent_skill.ipynb)

[`Download Python source code: task_agent_skill.py`](https://doc.agentscope.io/zh_CN/_downloads/35b30bfc3e633d45683a88db7706cfed/task_agent_skill.py)

[`Download zipped: task_agent_skill.zip`](https://doc.agentscope.io/zh_CN/_downloads/58ae75edb07add7bd1f8d02d87449bd2/task_agent_skill.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io/)

[Next 智能体](https://doc.agentscope.io/zh_CN/tutorial/task_agent.html)[Previous MCP](https://doc.agentscope.io/zh_CN/tutorial/task_mcp.html)

 Copyright © 2025, Alibaba 

 Made with [Sphinx](https://www.sphinx-doc.org/) and [@pradyunsg](https://pradyunsg.me/)'s [Furo](https://github.com/pradyunsg/furo)

[](https://github.com/agentscope-ai/agentscope)[](https://discord.gg/eYMpfnkG8h)[](https://qr.dingtalk.com/action/joingroup?code=v1,k1,OmDlBXpjW+I2vWjKDsjvI9dhcXjGZi3bQiojOq3dlDw=&_dt_no_comment=1&origin=11)

 On this page 

*   [智能体技能](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#)
    *   [注册智能体技能](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#id3)
    *   [在 ReActAgent 中集成智能体技能](https://doc.agentscope.io/zh_CN/tutorial/task_agent_skill.html#reactagent)