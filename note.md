# CompanyCopilot 后端实现说明

这份笔记的目标不是逐行讲源码，而是把“这个后端现在到底是怎么搭起来的、每一部分在做什么、整体怎么跑起来”讲清楚。写法会尽量像给一个有一定 Python、Web、数据库基础的大一学生解释。

---

## 1. 后端技术栈先概括一下

当前后端主要是下面这套组合：

- Web 框架：`FastAPI`
- 数据校验：`Pydantic`
- 大模型调用：
  - `openai` SDK
  - 阿里云 `DashScope` 的 OpenAI 兼容接口
- Agent 框架：`AgentScope`
- 网页搜索工具：`Firecrawl`
- RAG 检索：
  - 向量数据库：`Qdrant`
  - 嵌入模型：`qwen3-vl-embedding`
- 文件解析：
  - PDF、Word、Excel、PPT、TXT、CSV
  - 图片走视觉模型做 OCR + 内容理解

一句话概括：

这个后端其实做了三件事：

1. 提供普通的 HTTP API。
2. 用多个 Agent 做企业调研。
3. 做一个本地知识库，支持上传文档、向量化、语义检索和问答。

---

## 2. 整个后端目录在做什么

后端代码主要在 `backend/` 下面。

### 2.1 核心目录

- `backend/app/main.py`
  - FastAPI 入口
  - 所有主要 API 都在这里注册
- `backend/app/config.py`
  - 读取环境变量
- `backend/app/schemas.py`
  - 普通企业调研相关的数据模型
- `backend/app/agent_service.py`
  - 老版本的“简单尽调”服务
- `backend/app/agent_pipeline.py`
  - 新版本的三阶段企业调研流水线
- `backend/app/tools.py`
  - 给 Agent 用的工具函数
- `backend/app/agents/`
  - 放各种 Agent 实现
- `backend/app/rag/`
  - 放知识库、向量检索、文件处理相关代码
- `backend/skills/`
  - 给 AgentScope 用的技能说明文件
- `backend/qdrant_data/`
  - 向量库和元数据存储位置
- `backend/uploaded_files/`
  - 用户上传原始文件存储位置

---

## 3. 配置层怎么实现

文件：`backend/app/config.py`

这个文件很短，但很重要，因为它决定整个程序启动时需要哪些配置。

代码里定义了一个 `Settings` 数据类，里面现在有 4 个字段：

- `dashscope_api_key`
- `dashscope_model`
- `firecrawl_api_key`
- `firecrawl_api_url`

然后通过 `get_settings()` 统一读取环境变量。

### 3.1 它的设计思路

这个实现很朴素：

1. 从环境变量里取值。
2. 把值组装成 `Settings` 对象。
3. 其他模块只认 `Settings`，不再自己到处读 `os.getenv()`。

### 3.2 为什么这样写

好处是：

- 配置集中管理
- 以后加新配置容易
- 代码别的地方不用关心环境变量细节

### 3.3 当前必须配置什么

严格来说，代码里最硬性的要求是：

- `DASHSCOPE_API_KEY` 必须存在

如果没有，`get_settings()` 会直接抛出 `ValueError`。

默认模型是：

- `qwen-max`

Firecrawl 相关配置是可选的，但如果你要让调研 Agent 真正联网搜索，通常也需要配置。

---

## 4. FastAPI 入口是怎么组织的

文件：`backend/app/main.py`

这是整个后端最核心的文件。可以把它理解成：

- 对外接口总入口
- 各种服务对象的组装处
- 企业调研、聊天、RAG API 的统一接线板

---

## 5. `main.py` 最前面的几件事

### 5.1 先加载 `.env`

```python
from dotenv import load_dotenv
load_dotenv()
```

这一步的作用就是让本地 `.env` 文件里的环境变量进入进程环境。

### 5.2 创建 FastAPI 应用

```python
app = FastAPI(title="CompanyCopilot Agent API", version="0.1.0")
```

就是标准写法，后面所有 `@app.get`、`@app.post` 都挂在这个对象上。

### 5.3 延迟初始化几个全局服务

代码里有几个全局变量：

- `_knowledge_service`
- `_image_processor`
- `_file_storage`
- `_rag_agent_factory`

它们一开始都是 `None`。

然后通过下面这些函数按需创建：

- `get_knowledge_service()`
- `get_image_processor()`
- `get_file_storage()`
- `get_rag_agent_factory()`

### 5.4 为什么要这么写

这是一个很常见的“懒加载单例”思路。

意思是：

- 服务第一次真的被用到时才创建
- 创建后重复复用同一个实例

这样做主要有两个好处：

1. 启动更轻，服务不用一上来全初始化。
2. 像 Qdrant 客户端、嵌入模型这种对象比较重，复用更合理。

---

## 6. `main.py` 里的基础数据模型

虽然大部分数据模型分散在别的文件里，但 `main.py` 自己也定义了几个请求体模型。

### 6.1 `DueDiligenceRequest`

用于老版本尽调接口：

- `company_name`
- `prompt`

### 6.2 `CompanyResearchRequest`

用于企业研究接口：

- `domain`

这里要求传公司域名，比如：

- `apple.com`
- `notion.so`

### 6.3 `ChatMessage`

这是给 `/v1/chat/completions` 用的，兼容 OpenAI 的消息格式。

字段有：

- `role`
- `content`

其中 `content` 允许是：

- 普通字符串
- OpenAI 风格的内容块数组

### 6.4 `ChatCompletionRequest`

用于聊天接口，字段包括：

- `model`
- `messages`
- `temperature`
- `stream`
- `max_tokens`
- `top_p`

也就是说，这个接口做成了“看起来像 OpenAI API”的样子，方便前端直接按兼容方式调用。

---

## 7. `extract_text_content()` 在干什么

这个函数是个小工具函数，但很关键。

它做的是：

- 如果 `content` 本来就是字符串，直接返回
- 如果 `content` 是内容块数组，就把其中 `type == "text"` 的块拼起来

为什么要有它？

因为前端或者兼容 OpenAI 的调用里，消息内容不一定永远是单纯字符串，可能是：

```json
[
  {"type": "text", "text": "你好"},
  {"type": "image_url", "image_url": {...}}
]
```

后端如果只想做“文本理解”，就需要先抽纯文本。

---

## 8. 最简单的接口：健康检查

接口：

- `GET /health`

返回：

```json
{"status": "ok"}
```

这个就是最标准的健康检查接口，用来确认服务活着没有。

---

## 9. 老版本尽调接口怎么实现

接口：

- `POST /api/due-diligence`

对应服务：

- `backend/app/agent_service.py`

---

## 10. `DueDiligenceAgentService` 是怎么工作的

这个类可以理解成“旧版单 Agent 尽调服务”。

它不是三阶段流水线，而是一个 ReActAgent 自己搜、自己整理、自己输出。

### 10.1 初始化过程

构造函数会做两件事：

1. `_build_toolkit()`
2. `_build_agent()`

### 10.2 工具包里注册了什么

#### `view_text_file`

这是 AgentScope 自带的文件查看工具。

它的作用是让 Agent 可以读技能文件 `SKILL.md`。

#### `firecrawl_search`

这是项目自己定义在 `backend/app/tools.py` 里的联网搜索工具。

注册时会把：

- `api_url`
- `api_key`

预先塞进去，这样模型在调用工具时不用看到密钥。

#### `demo_sleep_tool`

这是一个测试并行工具调用的小工具，本质就是异步睡眠几秒再返回文本。

它不承担业务逻辑，更像调试/演示用。

#### `due_diligence` 技能

如果 `backend/skills/due_diligence/` 存在，就注册进去。

这意味着 Agent 在运行时可以先读取技能文件，再按技能文件里的规则搜资料。

### 10.3 Agent 怎么创建

底层模型是 `OpenAIChatModel`，但真正访问的是 DashScope 的 OpenAI 兼容接口：

- `base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"`

同时设置了：

- `enable_thinking = False`
- `parallel_tool_calls = True`

也就是说：

- 不开“思考模式”
- 允许模型并行调用工具

### 10.4 Agent 类型

这里用的是 `ReActAgent`。

你可以把 ReAct 理解成一种经典 Agent 模式：

1. 先思考
2. 需要时调用工具
3. 根据工具结果继续思考
4. 最后给答案

### 10.5 真正执行时做什么

`run_due_diligence()` 会：

1. 生成用户 prompt
2. 调用 Agent
3. 如果 `structured=True`，强制按 `DueDiligenceSummary` 输出结构化结果
4. 最后返回：
   - `text`
   - `metadata`

这里的 `DueDiligenceSummary` 在 `backend/app/schemas.py` 中定义，包含：

- 公司名
- 成立年份
- 总部
- 核心业务
- 高管
- 风险信号
- 来源链接

### 10.6 这个接口现在的定位

它属于兼容保留接口。

因为项目后来又做了新的三阶段企业调研流水线，所以这个接口更像“旧版本仍保留”。

---

## 11. 新版企业调研接口怎么实现

主要有四个接口：

- `POST /api/company-research`
- `POST /api/due-diligence-research`
- `POST /api/company-research/stream`
- `POST /api/due-diligence-research/stream`

它们都依赖：

- `backend/app/agent_pipeline.py`

---

## 12. `CompanyResearchPipeline` 的整体思想

这个类是现在企业调研的主流程。

它把任务拆成三步：

1. `ResearcherAgent` 收集原始资料
2. `AnalystAgent` 做分析提炼
3. `CompilerAgent` 生成最后报告

这其实就是一种很典型的“多 Agent 分工”：

- 调研的人负责搜资料
- 分析的人负责解释资料
- 写报告的人负责整理成可读文档

相比“一个 Agent 一把梭”，这种方式的优点是职责更清楚。

---

## 13. `CompanyResearchPipeline` 的懒加载

这个类内部也不是一上来就把所有 Agent 全建好，而是通过属性做懒加载：

- `researcher`
- `analyst`
- `compiler`
- `streaming_compiler`

第一次访问时才真正 `build()`。

这个思想和 `main.py` 的单例服务是一样的：避免不必要初始化。

---

## 14. 全量报告 `run()` 怎么跑

### 14.1 参数检查

先检查：

- `domain` 不能为空
- `mode` 必须是 `"full"` 或 `"quick"`

### 14.2 组装第一阶段提示词

如果 `mode == "full"`：

- 要做 8 维调研
- 要求系统性搜索
- 强调可靠来源
- 关注风险信号

如果 `mode == "quick"`：

- 只做产品相关快速尽调
- 重点放在产品、定价、商业模式

### 14.3 执行三个阶段

按顺序调用：

1. `researcher_result = await self.researcher(research_msg)`
2. `analyst_result = await self.analyst(researcher_result)`
3. `compiler_result = await self.compiler(..., structured_model=CompanyReport)`

注意第三步传了 `structured_model=CompanyReport`。

这表示系统希望最后结果尽量符合 `CompanyReport` 结构。

### 14.4 `CompanyReport` 里有哪些字段

定义在 `backend/app/schemas.py`，包括：

- `company_name`
- `overview`
- `products`
- `market`
- `personnel`
- `financials`
- `tech_stack`
- `news`
- `competitors`
- `risk_signals`
- `source_urls`
- `report`

其中最重要的是：

- `report`

它是一整份 Markdown 报告。

### 14.5 为什么最后优先读 `metadata["report"]`

因为结构化输出时，AgentScope 有可能把结构化结果塞到 metadata 里。

所以代码先尝试：

- `compiler_result.metadata.get("report")`

如果取不到，再退回：

- `compiler_result.content`

### 14.6 错误处理风格

这里有个特点：

流水线内部如果出错，并不是直接把异常往外抛，而是返回一份“带错误说明的 Markdown 报告”。

这样前端至少还能显示一份结果，不至于整个页面炸掉。

---

## 15. 流式报告 `run_streaming()` 怎么跑

这个方法和 `run()` 的区别在于：

- 前两个阶段还是正常一次性跑完
- 只有第三阶段报告生成改成流式输出

也就是：

1. Researcher 先把原始材料准备好
2. Analyst 先把分析准备好
3. 最后 `StreamingCompilerAgent` 一段一段吐报告

这样的设计比较现实，因为：

- 搜资料和分析这两步本身不太适合边做边给用户看碎片
- 最终写报告很适合流式显示

---

## 16. 参与流水线的几个 Agent 分别怎么写

### 16.1 `ResearcherAgent`

文件：`backend/app/agents/researcher.py`

这个 Agent 是“调研员”。

#### 它注册了什么

- `view_text_file`
- `firecrawl_search`
- `company_research` 技能
- `due_diligence` 技能

也就是说，它是一个带工具、带技能、能联网搜资料的 Agent。

#### 它的系统提示词在强调什么

- 先选对技能
- 按维度系统调研
- 尽量并行调用工具
- 信息必须有证据
- 客观中立

#### 它为什么最适合做第一阶段

因为它既能读技能说明，又能调用搜索工具，所以最适合干“信息采集”这件事。

---

### 16.2 `AnalystAgent`

文件：`backend/app/agents/analyst.py`

这个 Agent 是“分析师”。

特点很明确：

- 没有工具
- 不联网
- 只做分析和洞察提炼

系统提示词主要强调：

- 基于事实
- 多维度分析
- 识别趋势
- 识别风险
- 保持客观

这个设计很合理，因为分析阶段不应该继续无节制联网搜索，否则流程边界会变得很乱。

---

### 16.3 `CompilerAgent`

文件：`backend/app/agents/compiler.py`

这个 Agent 是“报告撰写员”。

同样：

- 没有工具
- 专心写报告

系统提示词要求它：

- 按标准章节结构组织内容
- 用 Markdown
- 单独列风险信号和来源
- 语气客观

它是非流式场景下的最终报告生成器。

---

### 16.4 `StreamingCompilerAgent`

文件：`backend/app/agents/streaming_compiler.py`

这个类不是通过 AgentScope 的 `ReActAgent` 实现，而是直接调用 `openai.AsyncOpenAI`。

这是因为开发者想要更直接地控制流式输出。

#### 它怎么工作

1. 构造系统提示词
2. 根据 `mode` 选择报告结构要求
3. 把分析师结果拼成 `user_content`
4. 调用 DashScope 的流式聊天接口
5. 把每个 token/片段 yield 出去

#### 为什么还写了 `_is_report_content()`

因为流式输出时有时会混进：

- JSON
- 工具调用痕迹
- thinking / analysis 这类元信息

这个函数就是做简单过滤，尽量只把真正的报告正文往外发。

#### 它的定位

它是当前流式企业调研接口真正用到的最终输出器。

---

### 16.5 `GuidanceAgent`

文件：`backend/app/agents/guidance.py`

这个 Agent 的角色是“前台引导员”。

它的职责不是直接调研，而是：

- 判断用户是不是想查公司
- 引导用户使用 `/research`
- 介绍系统功能

不过要注意：

虽然 `main.py` 里导入了 `GuidanceAgent`，但当前真正的 `/v1/chat/completions` 并没有直接实例化这个类，而是自己写了一套“引导型 system prompt + DashScope 调用”的逻辑。

也就是说：

- `GuidanceAgent` 这个类存在
- 但当前入口没有真正把它接上

你可以把它理解成“保留实现”。

---

### 16.6 `RAGAgent` 和 `RAGAgentFactory`

文件：`backend/app/agents/rag_agent.py`

这个类本意是做“知识库问答 Agent”。

它会注册：

- `view_text_file`
- `retrieve_knowledge`
- `knowledge_base` 技能

也就是说，它原本设计成：

1. 先检索知识库
2. 再基于检索结果回答

而且还专门写了一个 `RAGAgentFactory`，用于延迟创建这个 Agent。

但是当前 `main.py` 里的 RAG 问答接口实际走的不是这个 Agent，而是：

1. `KnowledgeService.retrieve()` 先检索
2. 再直接调用 DashScope 生成回答

所以这里也属于：

- 代码已实现
- 但当前主接口没有真正接入

---

### 16.7 `StreamingReportBuilder`

文件：`backend/app/agents/streaming_report_builder.py`

这个类是一个更“理想化”的流式报告方案。

它的思路不是等研究和分析全做完再输出，而是：

- 有一点增量数据就更新一点报告
- 某个维度分析完了，就立刻补该章节

为此它配合了：

- `backend/app/incremental_data.py`

里面定义了：

- 数据维度枚举 `DataDimension`
- 数据状态枚举 `DataStatus`
- 单维度数据 `DimensionData`
- 整体增量数据容器 `IncrementalCompanyData`

### 16.8 这套增量系统现在有没有真正接上

目前没有。

也就是说：

- 代码写了
- 思路也很完整
- 但当前 API 路由还没有实际使用它

所以如果你画“真实运行图”，可以把它归类到“预留/实验性设计”。

---

## 17. `tools.py` 里的工具层怎么实现

文件：`backend/app/tools.py`

这里定义了两个工具函数。

### 17.1 `firecrawl_search`

这是整个企业调研 Agent 最重要的外部工具。

它会：

1. 拼 Firecrawl 搜索接口地址 `/v1/search`
2. 组装查询参数：
   - `query`
   - `limit`
   - `lang = "en"`
3. 用 `httpx.AsyncClient` 发 POST 请求
4. 把返回 JSON 包成 `ToolResponse`

为什么返回 `ToolResponse`？

因为 AgentScope 的工具体系希望工具返回一种标准格式，方便 Agent 继续阅读工具结果。

### 17.2 `demo_sleep_tool`

这个就是异步 sleep 一下，然后返回一句文本。

本质上是在帮助验证“并行工具调用”是否真的生效。

---

## 18. `/v1/chat/completions` 是怎么做成 OpenAI 兼容接口的

接口：

- `POST /v1/chat/completions`

这个接口非常值得讲，因为它不是单纯聊天，而是“引导型聊天”。

### 18.1 它的目标

不是让模型自由聊天，而是让它扮演 CompanyCopilot 的引导员，告诉用户：

- 如果想做企业调研，用什么命令
- 如果想做快速尽调，用什么命令
- 如果想查知识库，用什么命令

### 18.2 它怎么处理历史消息

先遍历 `payload.messages`，把对话整理成中文文本：

- 用户消息变成 `用户：xxx`
- 助手消息变成 `助手：xxx`

然后提取最后一条用户消息作为当前问题。

如果历史消息很多，就拼成：

- 对话历史
- 当前用户问题

### 18.3 它的 system prompt 在做什么

这个 prompt 很长，本质是在给模型一套“客服接待话术”：

- 介绍 `/research`
- 介绍 `/due-diligence`
- 介绍 `/kg`
- 要求不要直接做调研
- 只负责引导用户用正确命令

### 18.4 真正调用模型时用什么

用的是：

- `openai.AsyncOpenAI`

但 `base_url` 指向 DashScope 兼容接口。

### 18.5 流式和非流式怎么分

#### 如果 `payload.stream == True`

返回 `StreamingResponse`，里面走 `generate_real_openai_stream()`。

这个函数会：

1. 调用 DashScope 流式接口
2. 把流式结果包装成 OpenAI SSE 格式
3. 最后发 `[DONE]`

#### 如果 `payload.stream == False`

就一次性请求模型，再把返回整理成 OpenAI 风格 JSON。

### 18.6 为什么还保留了 `generate_openai_stream()`

这是旧的“假流式”实现：

- 把完整文本切成词块
- 人工 sleep
- 模拟流式

现在已经弃用，但代码还留着，方便回滚。

---

## 19. RAG 模块是当前后端的另一个大头

这里的 RAG，可以简单理解成：

1. 用户上传文档
2. 系统把文档切块
3. 每块做向量化
4. 存到 Qdrant
5. 用户提问时先检索相关块
6. 再让模型根据这些块回答

RAG 相关代码主要在：

- `backend/app/rag/schemas.py`
- `backend/app/rag/knowledge_service.py`
- `backend/app/rag/image_processor.py`
- `backend/app/rag/file_storage.py`

---

## 20. RAG 的数据模型怎么设计

文件：`backend/app/rag/schemas.py`

### 20.1 文档类型 `DocumentType`

支持：

- PDF
- TXT
- MD
- DOCX
- XLSX
- PPTX
- CSV
- IMAGE

### 20.2 文档状态 `DocumentStatus`

有四种：

- `PENDING`
- `PROCESSING`
- `COMPLETED`
- `FAILED`

### 20.3 文档元数据 `DocumentMetadata`

这是最核心的文档对象，记录：

- 文档 id
- 文件名
- 文件类型
- 文件大小
- 状态
- chunk 数量
- 创建/更新时间
- user_id
- workspace_id
- knowledge_base_id
- description
- error_message

如果是图片，还多一个：

- `image_description`

### 20.4 文档块 `DocumentChunk`

表示“检索出来的一小块文本”，里面有：

- `doc_id`
- `chunk_index`
- `content`
- `metadata`
- `score`

### 20.5 知识库模型 `KnowledgeBase`

记录：

- id
- name
- description
- user_id
- workspace_id
- document_count
- created_at
- updated_at

### 20.6 其他响应模型

还定义了很多接口响应模型，比如：

- `DocumentListResponse`
- `DeleteDocumentResponse`
- `KnowledgeBaseListResponse`
- `KnowledgeBaseDetailResponse`
- `DocumentChunksResponse`

这些模型的意义是：

- 让接口返回结构更稳定
- FastAPI 自动生成更清楚的 OpenAPI 文档

---

## 21. `KnowledgeService` 是整个知识库的核心

文件：`backend/app/rag/knowledge_service.py`

这一个类几乎负责了 RAG 的大部分后端逻辑。

你可以把它理解成“知识库总管家”。

它负责：

- 多知识库管理
- 文档处理
- 向量化
- 检索
- 元数据持久化
- 文档删除

---

## 22. `KnowledgeService` 初始化时做了哪些准备

### 22.1 设置存储目录

默认根目录是：

- `backend/qdrant_data`

下面又分成几个目录：

- `metadata/`
- `chunks/`
- `vector_store/`

分别存：

- 知识库和文档元数据 JSON
- 每个文档切出来的 chunk JSON
- Qdrant 向量数据

### 22.2 内存中的缓存结构

初始化时还会准备：

- `_knowledge_instances`
  - 每个知识库对应一个 `SimpleKnowledge`
- `_ready_knowledge_bases`
  - 标记哪些知识库索引已经确认可用
- `_embedding_model`
  - 共用一个嵌入模型实例
- `_qdrant_client`
  - 共用一个 Qdrant 客户端
- `_knowledge_bases`
  - 所有知识库元数据
- `_documents`
  - 所有文档元数据

### 22.3 为什么要共享 embedding model 和 qdrant client

因为这两个对象都比较重。

尤其 Qdrant 这里走的是本地文件存储，共用客户端可以尽量减少文件锁冲突。

---

## 23. 嵌入模型和向量库怎么接

### 23.1 嵌入模型

`_get_embedding_model()` 使用的是：

- `DashScopeMultiModalEmbedding`
- 模型名：`qwen3-vl-embedding`
- 向量维度：`2560`

注意这里是多模态 embedding 模型，但当前主要还是用来编码文本块和图片描述文本。

### 23.2 Qdrant 客户端

`_get_qdrant_client()` 使用的是：

- `AsyncQdrantClient(path=...)`

这说明当前是本地持久化模式，不是纯远端托管。

### 23.3 每个知识库如何区分

每个知识库对应一个 collection，命名规则是：

- `kb_{kb_id}`

比如默认知识库就可能叫：

- `kb_default`

---

## 24. `SimpleKnowledge` 是怎么构造的

在 `_get_knowledge_instance()` 里，代码会：

1. 先创建一个 `QdrantStore(location=":memory:")`
2. 再强行把它的 `_client` 替换成共享的本地 Qdrant 客户端
3. 用这个 store + embedding model 组装 `SimpleKnowledge`

这段实现说明开发者在借 AgentScope 的封装，但又想自己控制真正的持久化客户端。

你可以理解成：

- 表面上还是走 AgentScope 的知识库抽象
- 底层实际用共享本地 Qdrant 客户端承载

---

## 25. 为什么还有“重建索引”逻辑

`KnowledgeService` 里有一套很重要的机制：

- `_build_documents_from_chunk_data()`
- `_build_documents_for_reindex()`
- `_get_expected_point_count()`
- `_ensure_knowledge_instance()`

这套逻辑的目的是：

如果元数据还在，但 Qdrant collection 状态不对，就能根据本地保存的 chunk 数据重新建索引。

### 25.1 具体怎么判断

在 `_ensure_knowledge_instance()` 里会检查：

1. collection 是否存在
2. 如果存在，里面 point 数量和当前文档 chunk 总数是否一致

如果不一致，就：

- 删除 collection
- 用本地 chunk 数据重建

### 25.2 为什么这很重要

因为知识库系统最怕“元数据还在，但向量索引坏了”。

这套设计实际上是在做一种简化版“自愈”。

---

## 26. 写入 embedding 时为什么还要分批和重试

相关函数：

- `_is_retryable_embedding_error()`
- `_add_documents_with_retry()`
- `_add_documents_in_batches()`

### 26.1 分批

代码里配置了：

- 每批 10 个文档
- 批次间隔 0.3 秒

这是为了控制 DashScope embedding 的 QPS，避免限流。

### 26.2 重试

如果出现：

- 502
- 503
- 504
- 超时
- 连接错误

这类上游故障，就会指数退避式重试。

### 26.3 为什么这一层很必要

因为“上传文档 -> 做 embedding”是最容易因为外部服务不稳定出问题的地方。

没有重试的话，用户体验会很差。

---

## 27. 元数据怎么持久化

### 27.1 加载

`_load_metadata()` 启动时会读两个 JSON：

- `knowledge_bases.json`
- `documents.json`

然后恢复成：

- `KnowledgeBase`
- `DocumentMetadata`

对象。

### 27.2 保存

`_save_metadata()` 每次修改后会把内存中的对象重新写回 JSON。

### 27.3 默认知识库

如果加载后发现没有默认知识库，就自动创建：

- id: `default`
- name: `默认知识库`

这让旧接口 `/api/rag/upload` 能直接工作，不要求用户先手动创建知识库。

---

## 28. 文档 chunk 为什么单独存 JSON

相关函数：

- `_save_chunks()`
- `get_document_chunks()`
- `_delete_chunks()`

设计思路是：

- 向量库存的是向量索引
- 但系统还想保留“原始切块文本”

所以每个文档的 chunk 会额外存成一个 JSON 文件。

这有两个用途：

1. 可以给前端查看文档被切成了哪些块
2. 如果索引丢了，还能拿 chunk 重建

---

## 29. 多知识库管理是怎么做的

### 29.1 创建知识库

`create_knowledge_base()` 会：

1. 生成 16 位十六进制 id
2. 创建 `KnowledgeBase`
3. 存到 `_knowledge_bases`
4. 写回元数据 JSON

### 29.2 查询和列表

- `get_knowledge_base()`
- `list_knowledge_bases()`

支持按：

- `user_id`
- `workspace_id`

过滤。

默认知识库会尽量排在前面。

### 29.3 更新知识库

`update_knowledge_base()` 的写法比较 Pydantic 风格：

- 先把旧对象 dump 成 dict
- 改字段
- 再新建一个 `KnowledgeBase`

### 29.4 删除知识库

`delete_knowledge_base()` 会：

1. 不允许删默认知识库
2. 删除该知识库下所有文档
3. 删除对应 Qdrant collection
4. 清理缓存和元数据

这个删除流程算是比较完整的。

---

## 30. 上传文本文档时后端怎么处理

主入口函数：

- `process_document()`

### 30.1 第一步：识别文件类型

通过扩展名判断：

- `.pdf`
- `.txt`
- `.md`
- `.docx`
- `.xlsx`
- `.pptx`
- `.csv`
- 图片格式

### 30.2 第二步：确定知识库

如果没传 `knowledge_base_id`，默认放进：

- `default`

### 30.3 第三步：生成文档 id

`generate_doc_id()` 的做法是：

1. 先对文件内容做 md5
2. 再把 `文件名 + 内容长度 + md5` 拼起来
3. 再做一次 sha256
4. 取前 32 位

这样做的目标是尽量稳定识别“同一个文件”。

### 30.4 第四步：去重

如果这个 `doc_id` 已经存在，而且状态是 `COMPLETED`，就直接返回已有元数据，不重复处理。

### 30.5 第五步：记录 PROCESSING 状态

先创建一条 `DocumentMetadata`，状态设为：

- `PROCESSING`

这样即使处理中途失败，也能留下失败痕迹。

### 30.6 第六步：真正解析文档

如果不是图片，就走 `_process_text_document()`。

这个函数会先把文件写成临时文件，然后根据类型选择不同 Reader：

- PDF -> `PDFReader`
- TXT / MD / CSV -> `TextReader`
- DOCX -> `WordReader`
- XLSX -> `ExcelReader`
- PPTX -> `PowerPointReader`

### 30.7 第七步：统一改写 doc_id

Reader 自己可能会生成自己的 doc_id。

代码会把每个 chunk 的 `doc.metadata.doc_id` 强制改成我们自己生成的文档 id。

这样后面检索结果才能准确关联回项目自己的 `_documents` 元数据字典。

### 30.8 第八步：保存 chunk JSON

把每个 chunk 的：

- `chunk_id`
- `content`

存到本地 JSON。

### 30.9 第九步：写入向量库

通过 `knowledge.add_documents(documents)`，把所有 chunk 做 embedding 并写进 Qdrant。

### 30.10 第十步：更新状态和知识库计数

成功后：

- 文档状态改成 `COMPLETED`
- `chunk_count` 记录块数
- 知识库的 `document_count + 1`
- 最后保存元数据

### 30.11 失败怎么处理

如果中途异常：

- 文档状态改成 `FAILED`
- `error_message` 记录错误
- 保存元数据
- 再把异常抛出去

---

## 31. 图片文档怎么处理

文件：`backend/app/rag/image_processor.py`

图片不走普通文本 Reader，而是单独走视觉模型。

### 31.1 为什么单独做

因为图片不能直接按文本切块，它需要先做：

- OCR
- 图表/表格理解
- 内容描述

### 31.2 `ImageProcessor` 初始化时配置了什么

它内部写死了一个视觉模型名：

- `qwen3.5-flash-2026-02-23`

访问地址是 DashScope 的兼容聊天接口。

### 31.3 `process_image()` 怎么跑

1. 检查扩展名是不是支持的图片格式
2. 确定知识库 id
3. 生成文档 id
4. 如果已处理过，直接返回旧结果
5. 创建 `DocumentMetadata`
6. 调 `_extract_image_content()` 让 VL 模型读图
7. 把模型返回的描述文本存进 `image_description`
8. 调 `_create_documents_from_description()` 生成一个文本 `Document`
9. 把这个“图片描述文档”写入知识库
10. 更新状态、计数、元数据

### 31.4 `_extract_image_content()` 具体做了什么

它会：

1. 把图片字节转成 base64
2. 拼成 `data:image/...;base64,...`
3. 发送给多模态聊天接口
4. system prompt 要求模型输出：
   - 图片中的文字
   - 图表/表格描述
   - 整体概述

最终拿到一段长文本，这段文本就相当于“图片内容的文字版说明”。

### 31.5 `_create_documents_from_description()` 的作用

把图片描述包装成一个 `Document`，让它也能像普通文本块一样进入向量检索体系。

所以系统对图片的处理思路其实是：

- 先看图
- 再把图转成文字
- 最后把这段文字放进知识库

### 31.6 这段代码里一个值得注意的点

它给 `DocMetadata` 额外塞了：

- `source_file`
- `source_doc_id`
- `content_type`

而 `KnowledgeService` 在文本处理部分反复强调“不要往 `DocMetadata` 里加自定义字段”，因为后续重建时可能不兼容。

所以这里可以看出：

- 当前图片处理实现和文本处理实现的元数据策略并不完全一致

从“讲实现”的角度，你只需要知道现在代码就是这样写的。

---

## 32. 文件物理存储怎么做

文件：`backend/app/rag/file_storage.py`

这个类很像一个“本地文件管理员”。

它管三类目录：

- `documents/`
- `images/`
- `temp/`

### 32.1 它提供的能力

- `save_file()`
- `get_file()`
- `read_file()`
- `delete_file()`
- `save_temp_file()`
- `delete_temp_file()`
- `cleanup_temp()`
- `get_storage_stats()`

### 32.2 为什么还要单独存原始文件

因为向量库只负责检索，不适合承担“原始文件归档”的职责。

所以这个项目把“原始文件存储”和“向量检索”分开做了。

---

## 33. RAG 检索是怎么实现的

核心函数：

- `KnowledgeService.retrieve()`

### 33.1 检索流程

1. 确定要搜哪些知识库
2. 对每个知识库执行 `knowledge.retrieve()`
3. 把返回的结果转成统一的 `DocumentChunk`
4. 补充元数据：
   - 源文件名
   - 知识库 id
   - 总 chunk 数
5. 合并所有知识库结果
6. 按相关分数排序
7. 截断到 `limit`
8. 组装 `formatted_context`

### 33.2 `formatted_context` 是什么

它是一段给 LLM 用的上下文文本，大概长这样：

```text
[1] (来源: xxx.pdf, 知识库: 默认知识库, 相关度: 0.82)
这里是检索到的文本块
```

把多个块用分隔线拼起来。

这样后续模型回答时，就能直接读这段上下文。

---

## 34. 文档查询和删除怎么做

### 34.1 查询

- `get_document()`
- `list_documents()`

就是从内存字典 `_documents` 里读，并支持按：

- 知识库
- 用户
- 工作空间

过滤。

### 34.2 删除文档

`delete_document()` 做的事情比较全：

1. 从 `_documents` 删除元数据
2. 更新知识库文档数
3. 保存元数据
4. 删除 chunk JSON
5. 尝试从 Qdrant 删除对应向量点

删除向量时用的是 Qdrant filter，按：

- `doc_id == 当前文档 id`

来删除。

如果向量删除失败，代码不会让整个删除失败，只会打 warning。

这是一种比较务实的错误处理：先保证主流程删掉。

---

## 35. RAG API 在 `main.py` 里怎么接上

这一部分接口很多，分成“知识库管理”和“文档管理”两大类。

---

## 36. 知识库管理 API

### 36.1 创建知识库

- `POST /api/rag/knowledge-bases`

调用：

- `knowledge_service.create_knowledge_base()`

### 36.2 列出知识库

- `GET /api/rag/knowledge-bases`

支持：

- `user_id`
- `workspace_id`

### 36.3 获取知识库详情

- `GET /api/rag/knowledge-bases/{kb_id}`

除了知识库本身，还会附带：

- 该知识库下的文档列表

### 36.4 更新知识库

- `PUT /api/rag/knowledge-bases/{kb_id}`

### 36.5 删除知识库

- `DELETE /api/rag/knowledge-bases/{kb_id}`

这里有两个额外动作：

1. 调 `KnowledgeService.delete_knowledge_base()`
2. 再通过 `FileStorage` 把本地原始文件一起删掉

这意味着删除知识库时，不只是删向量数据，也会删磁盘上的源文件。

---

## 37. 文档上传 API

### 37.1 上传到指定知识库

- `POST /api/rag/knowledge-bases/{kb_id}/upload`

流程是：

1. 检查知识库存在
2. 读上传文件字节
3. 判断是不是图片
4. 图片走 `ImageProcessor`
5. 普通文档走 `KnowledgeService.process_document()`
6. 最后用 `FileStorage.save_file()` 保存原始文件

### 37.2 旧的默认上传接口

- `POST /api/rag/upload`

这个接口跟上面逻辑差不多，只是默认上传到 `default` 知识库。

它属于兼容旧前端/旧调用方式的保留接口。

---

## 38. RAG 问答接口怎么实现

接口：

- `POST /api/rag/query/stream`

### 38.1 它不是直接用 `RAGAgent`

当前实现是“手工写的两段式流程”：

1. 先检索
2. 再生成回答

### 38.2 先检索

调用：

- `knowledge_service.retrieve()`

如果一个相关 chunk 都没找到，就走：

- `generate_no_context_response()`

返回一段流式 SSE，明确告诉用户没查到内容，并给出建议。

### 38.3 如果检索到了内容

就走：

- `generate_rag_response_with_sources()`

这个函数会分两步流式输出：

#### 第一步：先发 sources

先把每个 chunk 对应的：

- 内容摘要
- 分数
- 源文件
- 知识库 id / 名称
- chunk_index
- doc_id

组成一个：

```json
{"type": "sources", "sources": [...]}
```

先发给前端。

这很实用，因为前端可以先把“引用来源列表”渲染出来。

#### 第二步：再发模型答案

调用：

- `generate_rag_response()`

### 38.4 `generate_rag_response()` 做什么

1. 组装一个 system prompt，强调：
   - 必须基于参考资料回答
   - 不够就直说
   - 适当引用来源
2. 把 `formatted_context + 用户问题` 组成 user prompt
3. 调 DashScope 流式生成
4. 按 OpenAI SSE chunk 格式不断往外发

所以这个接口的本质就是：

- 检索增强生成

而不是纯聊天。

---

## 39. 文档管理 API

### 39.1 列出文档

- `GET /api/rag/documents`

### 39.2 删除文档

- `DELETE /api/rag/documents/{doc_id}`

除了删元数据和向量外，还会删本地原始文件。

### 39.3 获取文档详情

- `GET /api/rag/documents/{doc_id}`

### 39.4 获取文档 chunks

- `GET /api/rag/documents/{doc_id}/chunks`

这个接口很适合调试，因为可以直接看到“文档切块后的内容长什么样”。

---

## 40. 技能文件在后端里起什么作用

文件：

- `backend/skills/company_research/SKILL.md`
- `backend/skills/due_diligence/SKILL.md`
- `backend/skills/knowledge_base/SKILL.md`

这些文件不是 Python 代码，但它们在 AgentScope 体系里很重要。

你可以把它们理解成：

- 给 Agent 的“岗位说明书 + 作业流程”

---

## 41. `company_research` 技能写了什么

它定义了 8 个调研维度：

1. 公司概览
2. 产品服务
3. 市场表现
4. 人员架构
5. 融资财务
6. 技术栈
7. 近期动态
8. 竞争格局

还详细写了：

- 每个维度该搜什么
- 搜索关键词怎么设计
- 信息源优先级
- 风险信号怎么看

这会直接影响 `ResearcherAgent` 的行为。

---

## 42. `due_diligence` 技能写了什么

它是轻量版，只盯产品与服务维度。

强调：

- 只做两次核心搜索
- 重点看产品和定价
- 控制成本和响应时间

这就是为什么 quick 模式能比 full 模式更快。

---

## 43. `knowledge_base` 技能写了什么

它规定了知识库问答的工作流：

1. 理解用户问题
2. 调 `retrieve_knowledge`
3. 整合信息
4. 生成回答

同时要求：

- 标注来源
- 信息不足要诚实说明
- 不要编造

虽然当前主 RAG 接口没真正走 `RAGAgent`，但这个技能文件已经为以后接入 Agent 版本打好了基础。

---

## 44. 普通企业调研相关的数据模型

文件：`backend/app/schemas.py`

这里主要有两个结构化输出模型：

### 44.1 `DueDiligenceSummary`

用于老版本简单尽调。

字段偏简洁，适合直接给一个“摘要式结果”。

### 44.2 `CompanyReport`

用于新版完整企业调研。

这个模型很重要，因为它相当于告诉编译 Agent：

- 最终报告至少应该包含哪些部分

所以它既是“数据模型”，也是“输出约束”。

---

## 45. `main.py` 里的异常处理风格

从整体上看，当前项目很喜欢把异常分成两类：

1. `ValueError`
   - 转成 `400`
2. 其他异常
   - 转成 `500`

这种风格比较直接。

优点是简单清晰；
缺点是业务错误分类还不够细。

不过对于当前项目阶段来说，是够用的。

---

## 46. 真实运行时，几个主要业务流程长什么样

这一节很适合建立全局理解。

---

## 47. 流程一：全面企业调研

用户请求：

- `POST /api/company-research`
  或
- `POST /api/company-research/stream`

后端流程：

1. FastAPI 收到域名
2. 创建 `CompanyResearchPipeline`
3. `ResearcherAgent` 联网搜 8 个维度资料
4. `AnalystAgent` 整理并做商业分析
5. `CompilerAgent` 或 `StreamingCompilerAgent` 生成报告
6. 返回 Markdown 报告

---

## 48. 流程二：快速产品尽调

用户请求：

- `POST /api/due-diligence-research`
  或
- `POST /api/due-diligence-research/stream`

后端流程基本一样，只是：

- Researcher 读的是 `due_diligence` 技能
- 关注的是产品和定价
- 最终报告结构更轻量

---

## 49. 流程三：上传文档到知识库

用户请求：

- `POST /api/rag/upload`
  或
- `POST /api/rag/knowledge-bases/{kb_id}/upload`

后端流程：

1. 读取文件字节
2. 判断文件类型
3. 文本文件：
   - Reader 切块
   - 保存 chunk JSON
   - embedding
   - 写入 Qdrant
4. 图片文件：
   - VL 模型理解图片
   - 生成描述文本
   - 描述文本向量化
5. 保存原始文件到磁盘
6. 返回文档元数据

---

## 50. 流程四：知识库问答

用户请求：

- `POST /api/rag/query/stream`

后端流程：

1. `KnowledgeService.retrieve()` 检索相关 chunk
2. 如果没找到，直接返回“未检索到相关信息”
3. 如果找到了，先把 sources 发给前端
4. 再把上下文喂给模型
5. 模型流式输出回答

---

## 51. 流程五：聊天引导

用户请求：

- `POST /v1/chat/completions`

后端流程：

1. 整理消息历史
2. 提取最后一个用户问题
3. 套用“引导员”系统提示词
4. 调 DashScope
5. 用 OpenAI 兼容格式返回

这个接口本质不是知识问答接口，而是“命令路由引导接口”。

---

## 52. 当前后端里哪些代码是“已经接上的”

已经真正接入主要 API 的有：

- `config.py`
- `main.py`
- `agent_service.py`
- `agent_pipeline.py`
- `tools.py`
- `agents/researcher.py`
- `agents/analyst.py`
- `agents/compiler.py`
- `agents/streaming_compiler.py`
- `rag/knowledge_service.py`
- `rag/image_processor.py`
- `rag/file_storage.py`
- `rag/schemas.py`
- `skills/*.md`

---

## 53. 哪些代码更像“预留能力”或“暂未主路由使用”

主要有：

- `agents/guidance.py`
  - 类已实现，但 `/v1/chat/completions` 没直接用它
- `agents/rag_agent.py`
  - 类和工厂都写了，但主 RAG 接口没直接用它
- `incremental_data.py`
  - 增量数据结构已写
- `agents/streaming_report_builder.py`
  - 增量流式报告构建器已写，但主流程未接入

这部分如果以后扩展，很可能会重新接上。

---

## 54. 从“工程结构”角度看，这个后端的优点是什么

### 54.1 模块边界相对清楚

- API 入口在 `main.py`
- Agent 流程在 `agent_pipeline.py`
- 单 Agent 服务在 `agent_service.py`
- 知识库逻辑在 `rag/`

### 54.2 多知识库设计已经成型

不是只做一个默认库，而是已经支持：

- 创建多个知识库
- 按用户/工作空间过滤

### 54.3 文档元数据和向量索引分开存

这是比较合理的做法：

- 元数据 JSON 可读、好调试
- 向量库负责相似度搜索
- 原始文件再单独存磁盘

### 54.4 有一定容错能力

例如：

- embedding 写入有重试
- collection 异常时可重建
- 删除向量失败不阻断主删除流程

---

## 55. 如果你要把这套后端讲给同学听，可以怎么一句话概括

一句比较准确的话是：

这个后端是一个“FastAPI + AgentScope + DashScope + Qdrant”的 AI 应用后端，它一边用多 Agent 做企业调研报告，一边用 RAG 做文档知识库问答，还额外做了一个 OpenAI 兼容聊天接口来引导前端使用正确命令。

---

## 56. 最后做一个总总结

当前后端实际上由两条主线组成：

### 第一条主线：企业调研

- `ResearcherAgent` 负责搜
- `AnalystAgent` 负责分析
- `CompilerAgent` / `StreamingCompilerAgent` 负责写报告

### 第二条主线：知识库问答

- `KnowledgeService` 负责文档处理、向量化、检索
- `ImageProcessor` 负责图片理解
- `FileStorage` 负责原始文件存储
- RAG 接口负责“先检索，再生成”

而 `main.py` 就像总调度室，把所有东西接成 HTTP API。

如果你把它当成一个课程项目来看，它已经不是“单文件 demo”了，而是一个比较完整的后端雏形：

- 有配置层
- 有服务层
- 有 Agent 层
- 有 RAG 层
- 有存储层
- 有流式接口
- 有兼容 OpenAI 的接口

所以理解它的最好方式，不是死记每一行，而是先抓住这几个核心问题：

1. 请求从哪里进来？
2. 进来后走哪个服务？
3. 服务内部是单 Agent、流水线，还是 RAG？
4. 数据最后存到哪里？
5. 返回给前端的是普通 JSON 还是流式 SSE？

只要这五个问题想清楚，整个后端你基本就能看懂八成了。
