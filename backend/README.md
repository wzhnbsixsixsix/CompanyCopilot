# CompanyCopilot Backend (AgentScope)

基于 AgentScope 的企业背调和知识库问答后端服务。

## 快速启动

```bash
# 1. 进入 backend 目录
cd backend

# 2. 激活虚拟环境（如果还没激活）
source .venv/bin/activate
# 或者如果使用项目根目录的 venv
source ../venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填写 DASHSCOPE_API_KEY 等

# 5. 启动服务（端口 8000）
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 或者使用 --reload 开发模式
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**注意**: 启动命令必须使用 `app.main:app`（模块路径），而不是 `main:app`。

## 环境变量配置

在 `.env` 文件中配置：

```bash
# 必需
DASHSCOPE_API_KEY=sk-xxx          # 阿里云 DashScope API Key
DASHSCOPE_MODEL=qwen3.5-flash     # 对话模型（默认 qwen3.5-flash）

# 可选
FIRECRAWL_API_KEY=fc-xxx          # Firecrawl API Key（用于网页抓取）
FIRECRAWL_API_URL=https://api.firecrawl.dev
```

## 功能模块

### 1. 企业背调 (Company Research)

- **全面调研** `/research domain.com` - 8维度企业背景调查
- **快速尽调** `/due-diligence domain.com` - 产品维度快速调研

### 2. RAG 知识库问答

基于 AgentScope RAG 组件实现，支持：
- 多知识库管理
- 文档上传（PDF、Word、Excel、PPT、TXT、MD、CSV、图片）
- 语义检索和问答

**技术栈**：
- 嵌入模型: DashScope `qwen3-vl-embedding` (2560维)
- 向量存储: Qdrant 内存模式
- 分块策略: paragraph split, 512 tokens

## API 端点

### 健康检查
```bash
curl http://localhost:8000/health
```

### 企业背调
```bash
# 全面调研（流式）
curl -X POST http://localhost:8000/api/company-research/stream \
  -H "Content-Type: application/json" \
  -d '{"domain": "apple.com"}'

# 快速尽调（流式）
curl -X POST http://localhost:8000/api/due-diligence-research/stream \
  -H "Content-Type: application/json" \
  -d '{"domain": "apple.com"}'
```

### 知识库管理
```bash
# 列出知识库
curl http://localhost:8000/api/rag/knowledge-bases

# 创建知识库
curl -X POST http://localhost:8000/api/rag/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"name": "项目文档", "description": "项目相关文档"}'

# 上传文档到默认知识库
curl -X POST http://localhost:8000/api/rag/upload \
  -F "file=@document.pdf" \
  -F "description=项目说明文档"

# 上传文档到指定知识库
curl -X POST http://localhost:8000/api/rag/knowledge-bases/{kb_id}/upload \
  -F "file=@document.pdf"

# RAG 问答（流式）
curl -X POST http://localhost:8000/api/rag/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "文档的主要内容是什么？", "limit": 5}'

# 列出文档
curl http://localhost:8000/api/rag/documents

# 删除文档
curl -X DELETE http://localhost:8000/api/rag/documents/{doc_id}
```

### OpenAI 兼容接口
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-plus",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

## 项目结构

```
backend/
├── app/
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── agent_service.py     # Agent 服务
│   ├── agent_pipeline.py    # 企业调研流水线
│   ├── tools.py             # 工具定义
│   ├── schemas.py           # 数据模型
│   ├── agents/              # Agent 实现
│   │   ├── researcher.py    # 调研 Agent
│   │   ├── analyst.py       # 分析 Agent
│   │   ├── compiler.py      # 编译 Agent
│   │   ├── guidance.py      # 引导 Agent
│   │   └── rag_agent.py     # RAG Agent
│   └── rag/                 # RAG 模块
│       ├── knowledge_service.py  # 知识库服务
│       ├── image_processor.py    # 图片处理
│       ├── file_storage.py       # 文件存储
│       └── schemas.py            # RAG 数据模型
├── skills/                  # Agent 技能定义
├── qdrant_data/            # Qdrant 元数据存储
├── uploaded_files/         # 上传文件存储
├── requirements.txt        # Python 依赖
├── .env                    # 环境变量（需自行创建）
└── README.md               # 本文件
```

## 已知限制

1. **向量数据不持久化**: 当前使用 Qdrant 内存模式，服务重启后向量数据会丢失（元数据会保留）
2. **嵌入模型限制**: 需要 DashScope 账户开通 `qwen3-vl-embedding` 权限

## 故障排除

### 启动失败: "Could not import module main"
确保使用正确的模块路径：
```bash
# 正确
python -m uvicorn app.main:app --port 8000

# 错误
python -m uvicorn main:app --port 8000
```

### 嵌入模型 403 错误
检查 DashScope API Key 是否有 `qwen3-vl-embedding` 权限。可在阿里云控制台开通。

### 端口被占用
```bash
# 查看占用端口的进程
lsof -i :8000

# 杀掉进程
kill $(lsof -t -i :8000)
```
