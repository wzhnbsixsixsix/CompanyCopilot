# CompanyCopilot 项目文档

CompanyCopilot 是一个基于 LLM 的企业级智能助手平台，旨在通过自然语言交互提升公司内部效率。项目前端采用开源的 `chatbot-ui`，后端基于 `AgentScope` 框架构建多智能体系统。

## 核心架构

- **前端**: [Chatbot UI](https://github.com/mckaywrigley/chatbot-ui) (Next.js)
- **后端**: [AgentScope](https://github.com/modelscope/agentscope) (Python 多智能体框架)
- **数据处理**: RAG (检索增强生成) + 向量数据库
- **外部工具**: Firecrawl (网页爬虫)

## 功能模块

### 1. 企业知识库问答 (RAG)
基于向量数据库构建的企业大脑，支持员工通过自然语言查询内部信息。

可以上传文档如pdf,照片,md文件,csv文件，来形成企业知识库

- **业务流程解答**: 快速获取公司报销、请假、审批等流程指引。
- **公司背景信息**: 查询公司历史、架构、文化价值观等。
- **客户信息查询**: 检索CRM中的客户基础资料和历史交互记录。

### 2. 企业背景调查助手 (/research 命令)
**新增功能**: 基于 3-Agent 流水线的全面企业背调系统

#### 使用方法
在聊天界面输入：`/research <公司域名>`

例如：
- `/research apple.com`
- `/research openai.com`
- `/research alibaba.com`

#### 功能特点
- **3-Agent 协作**: ResearcherAgent (数据收集) → AnalystAgent (深度分析) → CompilerAgent (报告生成)
- **8维度调研**: 涵盖公司概览、产品服务、市场表现、团队架构、融资财务、技术栈、近期动态、竞争格局
- **并行搜索**: 利用 Firecrawl API 并行收集多维度数据，提高调研效率
- **风险识别**: 自动识别法律风险、财务风险、声誉风险等关键信号
- **来源标注**: 所有关键信息附带可验证的来源链接

#### 报告内容
生成包含以下8个章节的 Markdown 格式报告：

1. **公司概览与基础信息** - 成立时间、总部、行业分类、使命愿景
2. **产品与服务体系** - 主要产品线、商业模式、定价策略
3. **市场表现与受众分析** - 访问量、市场份额、用户画像
4. **关键人员与组织架构** - 核心高管、创始团队背景
5. **融资历史与财务状况** - 融资轮次、投资方、营收状况
6. **技术栈与数字化水平** - 技术架构、基础设施
7. **近期动态与发展趋势** - 最新新闻、产品发布、战略调整
8. **竞争格局与市场地位** - 主要竞争对手、差异化优势

### 3. 简单背调助手 (原有功能)
保留原有的单一智能体背调功能作为快速查询选项。

---

## 技术架构

### 后端 (Python + AgentScope)
```
backend/
├── app/
│   ├── agents/                    # 3个专职智能体
│   │   ├── researcher.py         # 数据收集专家
│   │   ├── analyst.py            # 数据分析专家
│   │   └── compiler.py           # 报告生成专家
│   ├── agent_pipeline.py         # 3-Agent 流水线编排
│   ├── agent_service.py          # 原有单一智能体服务
│   ├── main.py                   # FastAPI 应用入口
│   ├── tools.py                  # 工具函数 (Firecrawl)
│   ├── schemas.py                # 数据模型定义
│   └── config.py                 # 配置管理
├── skills/
│   ├── due_diligence/            # 原有背调技能
│   └── company_research/         # 新增企业调研技能
│       └── SKILL.md              # 8维度调研规范
└── requirements.txt
```

### 前端 (Next.js + Chatbot UI)
```
chatbot-ui/
├── app/
│   └── api/
│       ├── company-research/     # 新增企业背调API路由
│       └── chat/                 # 原有聊天API路由
├── components/
│   └── chat/
│       ├── chat-helpers/         # 聊天辅助函数
│       └── chat-hooks/           # React Hooks
└── ...
```

## 使用说明

### 企业背景调查 (/research)
在聊天界面输入 `/research <域名>` 即可启动全面企业调研：
```bash
/research apple.com
```

### 普通聊天对话 (Custom Model 配置)
要让普通聊天也使用 AgentScope 后端而不是外部 LLM 服务，请在 chatbot-ui 中配置自定义模型：

1. **启动后端服务**
   ```bash
   cd backend
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
   ```

2. **在前端界面配置 Custom Model**
   - 打开聊天界面，点击 **Settings** → **Models**
   - 点击 **"+ Add Custom Model"**
   - 填写以下配置：
     - **Model Name**: `CompanyCopilot Agent`
     - **Model ID**: `guidance-agent` (任意值，后端会忽略)
     - **Base URL**: `http://localhost:8001/v1`
     - **API Key**: `dummy` (必填字段，但后端不验证)
     - **Context Length**: `4096`

3. **选择模型进行对话**
   - 在聊天侧边栏选择刚创建的 `CompanyCopilot Agent` 模型
   - 现在普通聊天消息将通过 AgentScope 的 GuidanceAgent 处理
   - 该 Agent 会识别用户意图，引导使用 `/research` 命令进行企业调研

### 意图识别与引导功能
GuidanceAgent 的主要功能：
- 当用户询问企业相关信息时，会引导使用 `/research <域名>` 命令
- 回答关于 CompanyCopilot 系统功能的一般性问题
- 确保用户了解正确的企业调研流程

---

## 部署说明

### 后端启动
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 前端启动
```bash
cd chatbot-ui
npm run dev
```

### 环境变量配置
**后端 (.env)**
```
DASHSCOPE_API_KEY=sk-...        # 阿里云DashScope API密钥
DASHSCOPE_MODEL=qwen3.5-plus    # 模型名称
FIRECRAWL_API_KEY=fc-...        # Firecrawl API密钥
FIRECRAWL_API_URL=https://api.firecrawl.dev
```

**前端 (.env.local)**
```
NEXT_PUBLIC_SUPABASE_URL=...    # Supabase项目URL
NEXT_PUBLIC_SUPABASE_ANON_KEY=... # Supabase公开密钥
SUPABASE_SERVICE_ROLE_KEY=...   # Supabase服务角色密钥
```

---

## 目录结构说明

- `/backend`: Python 后端代码，基于 AgentScope 框架
- `/chatbot-ui`: Next.js 前端代码
- `/docs`: 项目文档及参考资料
  - `/crewai_example`: CrewAI 设计参考文档
  - `/agent_scope_official_docs`: AgentScope 官方文档
- `README.md`: 本项目总览
