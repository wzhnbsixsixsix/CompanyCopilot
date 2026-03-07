# CompanyCopilot Backend (AgentScope)

This backend implements a Due Diligence Agent using AgentScope + Firecrawl, aligned with the official AgentScope docs in `docs/agent_scope_official_docs`.

## What is implemented

- ReActAgent-based service (`ReActAgent`)
- Toolkit tool registration with preset secret kwargs
- Async tool sample for parallel tool calls
- Structured output via Pydantic model (`structured_model`)
- Agent skill registration (`register_agent_skill`) with `SKILL.md`
- Optional AgentScope Studio integration via `AGENTSCOPE_STUDIO_URL`
- FastAPI endpoint: `POST /api/due-diligence`

## Setup

1. Create venv and install deps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure env:

```bash
cp .env.example .env
```

Fill at least:

- `DASHSCOPE_API_KEY`
- `FIRECRAWL_API_KEY` (recommended for search tool)

3. Run API:

```bash
uvicorn app.main:app --reload --port 8001
```

4. Test endpoint:

```bash
curl -X POST http://localhost:8001/api/due-diligence \
  -H "Content-Type: application/json" \
  -d '{"company_name":"OpenAI"}'
```

## File map

- `app/main.py`: FastAPI app and routes
- `app/agent_service.py`: AgentScope ReAct agent construction
- `app/tools.py`: Firecrawl + async demo tools
- `app/schemas.py`: Structured output schema
- `skills/due_diligence/SKILL.md`: Agent skill spec
