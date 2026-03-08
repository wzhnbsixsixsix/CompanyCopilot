# CompanyCopilot

CompanyCopilot 是一个面向企业场景的 AI 助手项目，当前聚焦三类能力：

- 企业研究：基于多 Agent 流水线输出结构化公司研究报告
- 快速尽调：以产品和商业模式为重点的轻量调研
- 知识库问答：上传企业文档后进行检索增强问答（RAG）

项目采用前后端分离架构：

- 前端：`chatbot-ui`，基于 Next.js 14、TypeScript、Tailwind CSS、Supabase
- 后端：`backend`，基于 FastAPI、AgentScope、DashScope、Qdrant

## 当前能力

| 能力 | 用户命令 | 前端代理路由 | 后端接口 |
| --- | --- | --- | --- |
| 企业研究（完整） | `/research apple.com` | `/api/company-research-stream` | `/api/company-research/stream` |
| 快速尽调 | `/due-diligence apple.com` | `/api/due-diligence-research-stream` | `/api/due-diligence-research/stream` |
| 知识库问答 | `/kg 这个产品的定价策略是什么？` | `/api/rag-query` | `/api/rag/query/stream` |
| 引导式聊天 | 普通对话 | `/api/chat/companycopilot` | `/v1/chat/completions` |

说明：

- `/research` 走完整企业研究流程，输出 8 个维度的调研结果
- `/due-diligence` 走快速模式，更适合验证产品、定价和商业模式
- `/kg` 从已上传文档中检索答案，并返回引用来源
- 普通聊天默认由 `CompanyCopilot Agent` 引导用户选择合适命令，不直接代替研究流程

## 仓库结构

```text
CompanyCopilot/
├── backend/                  # FastAPI + AgentScope 后端
│   ├── app/
│   │   ├── agents/           # researcher / analyst / compiler / guidance / rag agent
│   │   ├── rag/              # 知识库、文档处理、向量检索
│   │   ├── agent_pipeline.py # 企业研究流水线
│   │   ├── agent_service.py  # 兼容旧版尽调服务
│   │   ├── config.py         # 环境变量读取
│   │   └── main.py           # API 入口
│   ├── skills/               # research / due diligence / knowledge base 技能定义
│   └── requirements.txt
├── chatbot-ui/               # Next.js 前端
│   ├── app/api/              # 前端代理路由
│   ├── components/           # 聊天、知识库、侧边栏 UI
│   ├── lib/                  # 模型与工具封装
│   ├── context/              # 全局状态管理
│   └── supabase/             # 本地 Supabase 配置与迁移
├── docs/                     # 设计文档与参考资料
├── AGENTS.md                 # 仓库内 AI 开发约束
└── README.md
```

## 技术架构

### 后端

- FastAPI 提供研究、尽调、RAG、OpenAI 兼容聊天接口
- AgentScope 负责 Agent 组织与 RAG 组件集成
- DashScope 提供对话与多模态能力
- Qdrant 用于向量索引

### 前端

- 基于 Chatbot UI 二次开发
- 通过 Next.js Route Handlers 转发到 Python 后端
- Supabase 负责用户、工作区、聊天记录和文件元数据
- 内置 `CompanyCopilot Agent` 模型，调用后端 `/v1/chat/completions`

## 快速启动

### 1. 准备环境

建议环境：

- Node.js 18+
- npm
- Python 3.11+
- Docker + Supabase CLI（运行完整前端功能时需要）

### 2. 启动后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

在 `backend/.env` 中创建并填写：

```bash
DASHSCOPE_API_KEY=sk-xxx
DASHSCOPE_MODEL=qwen-max
FIRECRAWL_API_KEY=fc-xxx
FIRECRAWL_API_URL=https://api.firecrawl.dev
```

启动服务：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

健康检查：

```bash
curl http://localhost:8000/health
```

### 3. 启动前端

```bash
cd chatbot-ui
npm install
cp .env.local.example .env.local
```

在 `chatbot-ui/.env.local` 中至少补齐这些变量：

```bash
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
SUPABASE_SERVICE_ROLE_KEY=...
```

如果你需要完整的本地登录、工作区和聊天持久化能力，先启动 Supabase：

```bash
supabase start
```

然后启动前端：

```bash
npm run dev
```

或使用仓库已有脚本一键启动本地 Supabase + 类型生成 + 开发服务：

```bash
npm run chat
```

默认访问地址：

- 前端：[http://localhost:3000](http://localhost:3000)
- 后端：[http://localhost:8000](http://localhost:8000)

## 使用方式

### 方式一：在前端界面直接使用

前端已经内置 `CompanyCopilot Agent` 模型。启动前后端后，在聊天界面可直接：

- 输入普通问题，让引导 Agent 推荐命令
- 输入 `/research <域名>` 触发完整企业研究
- 输入 `/due-diligence <域名>` 触发快速尽调
- 输入 `/kg <问题>` 查询已上传知识库文档

示例：

```text
/research openai.com
/due-diligence stripe.com
/kg 这份合同里的付款周期是什么？
```

### 方式二：直接调用后端 API

完整企业研究：

```bash
curl -X POST http://localhost:8000/api/company-research/stream \
  -H "Content-Type: application/json" \
  -d '{"domain":"openai.com"}'
```

快速尽调：

```bash
curl -X POST http://localhost:8000/api/due-diligence-research/stream \
  -H "Content-Type: application/json" \
  -d '{"domain":"stripe.com"}'
```

OpenAI 兼容聊天：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "companycopilot-agent",
    "messages": [{"role":"user","content":"帮我看看 OpenAI"}],
    "stream": true
  }'
```

## 知识库（RAG）

### 支持的文件类型

- 文本与文档：`pdf`、`txt`、`md`、`docx`、`xlsx`、`pptx`、`csv`
- 图片：`jpg`、`jpeg`、`png`、`webp`

### 典型流程

1. 创建知识库
2. 上传文档到默认知识库或指定知识库
3. 通过 `/kg` 或 RAG API 发起查询
4. 在回答中查看引用来源和文档片段

### 常用 API

创建知识库：

```bash
curl -X POST http://localhost:8000/api/rag/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"name":"Sales Docs","description":"销售资料库"}'
```

上传文档到默认知识库：

```bash
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@./example.pdf"
```

上传文档到指定知识库：

```bash
curl -X POST http://localhost:8000/api/rag/knowledge-bases/<kb_id>/upload \
  -F "file=@./example.pdf"
```

查询知识库：

```bash
curl -X POST http://localhost:8000/api/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"这份文档的关键条款是什么？","limit":5}'
```

## 主要接口一览

### 后端

- `GET /health`
- `POST /api/company-research`
- `POST /api/company-research/stream`
- `POST /api/due-diligence`
- `POST /api/due-diligence-research`
- `POST /api/due-diligence-research/stream`
- `POST /v1/chat/completions`
- `POST /api/rag/knowledge-bases`
- `GET /api/rag/knowledge-bases`
- `GET /api/rag/knowledge-bases/{kb_id}`
- `PUT /api/rag/knowledge-bases/{kb_id}`
- `DELETE /api/rag/knowledge-bases/{kb_id}`
- `POST /api/rag/knowledge-bases/{kb_id}/upload`
- `POST /api/rag/upload`
- `POST /api/rag/query/stream`
- `GET /api/rag/documents`
- `GET /api/rag/documents/{doc_id}`
- `GET /api/rag/documents/{doc_id}/chunks`
- `DELETE /api/rag/documents/{doc_id}`

### 前端代理路由

- `/api/chat/companycopilot`
- `/api/company-research`
- `/api/company-research-stream`
- `/api/due-diligence-research`
- `/api/due-diligence-research-stream`
- `/api/knowledge-bases`
- `/api/knowledge-bases/[id]`
- `/api/knowledge-bases/[id]/upload`
- `/api/rag-upload`
- `/api/rag-query`
- `/api/rag-documents/[docId]`
- `/api/rag-chunks/[docId]`

## 开发命令

### 前端

```bash
cd chatbot-ui
npm run dev
npm run build
npm run lint
npm run type-check
npm run test
```

### 后端

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
python -m pytest
```

## 常见问题

### 1. 后端启动报错 `DASHSCOPE_API_KEY is required`

`backend/app/config.py` 会在启动时强校验 `DASHSCOPE_API_KEY`，必须先配置 `backend/.env`。

### 2. 前端请求不到后端

检查 `chatbot-ui/.env.local` 中是否显式设置：

```bash
BACKEND_URL=http://localhost:8000
```

前端多个代理路由都会读取这个变量；如果没配，会默认打到本地 `8000`。

### 3. 只启动前端但登录或工作区异常

这是因为当前前端依赖 Supabase。要完整体验聊天持久化、工作区和文件记录，需要先运行本地或远程 Supabase。

### 4. RAG 上传成功但问答效果差

优先检查：

- 文档是否真正完成解析和分块
- 查询是否过于宽泛
- DashScope 与嵌入权限是否可用
- 图片文档是否提取到了 `image_description`

## 相关文档

- [根目录 AGENTS 指南](./AGENTS.md)
- [后端说明](./backend/README.md)
- [前端原始说明](./chatbot-ui/README.md)
- [设计与参考文档](./docs)
