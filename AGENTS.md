# AGENTS.md - CompanyCopilot Development Guide

> Guidelines for AI coding agents working in this repository.

## Project Overview

CompanyCopilot is an enterprise AI assistant with:
- **Knowledge Base (RAG)**: Document upload, vector search, semantic Q&A
- **Company Research**: 8-dimension enterprise investigation via 3-agent pipeline
- **Due Diligence**: Quick product-focused research

## Architecture

| Layer | Stack |
|-------|-------|
| **Frontend** | Next.js 14, TypeScript, Tailwind CSS, Supabase |
| **Backend** | Python FastAPI, AgentScope, Qdrant (vector DB), DashScope LLM |
| **Database** | Supabase (PostgreSQL) for app data, Qdrant for vectors |

## Build & Run Commands

### Backend (Python)

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start dev server (with hot reload)
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start production server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend (Next.js)

```bash
cd chatbot-ui

# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Full startup (Supabase + types + dev)
npm run chat
```

## Testing Commands

### Frontend Unit Tests (Jest)

```bash
cd chatbot-ui

# Run all tests
npm run test

# Run single test file
npm run test -- __tests__/lib/openapi-conversion.test.ts

# Run tests matching pattern
npm run test -- --testNamePattern="should convert"

# Run with coverage
npm run test -- --coverage
```

### E2E Tests (Playwright)

```bash
cd chatbot-ui/__tests__/playwright-test

# Run all E2E tests
npm run integration

# Run with UI
npm run integration:open

# Generate test code
npm run integration:codegen
```

### Backend Tests

```bash
cd backend

# Run pytest (if tests exist)
python -m pytest

# Run single test file
python -m pytest test_file.py

# Run specific test
python -m pytest test_file.py::test_function_name -v
```

## Linting & Formatting

### Frontend

```bash
cd chatbot-ui

npm run lint          # ESLint check
npm run lint:fix      # ESLint auto-fix
npm run format:check  # Prettier check
npm run format:write  # Prettier auto-fix
npm run type-check    # TypeScript check
npm run clean         # lint:fix + format:write
```

**Pre-commit hook** (`.husky/pre-commit`): Runs `lint:fix` + `format:write` automatically.

### Backend

Python code uses implicit Black-style formatting (88-100 char lines, 4-space indent).
Use `# noqa: BLE001` for intentional bare exception handling.

## Code Style Guidelines

### Python (backend/app/)

**Imports** - Order: stdlib → third-party → local (relative with `.`)
```python
import asyncio
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .config import get_settings
from .rag import KnowledgeService
```

**Type Hints** - Use extensively; both `str | None` and `Optional[str]` accepted
```python
async def process(content: bytes, filename: str) -> DocumentMetadata:
    ...
```

**Naming Conventions**
| Element | Style | Example |
|---------|-------|---------|
| Classes | PascalCase | `KnowledgeService` |
| Functions | snake_case | `get_document_chunks` |
| Constants | UPPER_SNAKE | `DEFAULT_KB_ID` |
| Private | Leading `_` | `_save_metadata()` |

**Error Handling** - Chain exceptions with `from exc`
```python
except ValueError as exc:
    raise HTTPException(status_code=400, detail=str(exc)) from exc
except Exception as exc:  # noqa: BLE001
    raise HTTPException(status_code=500, detail=f"Error: {exc}") from exc
```

**Docstrings** - Google-style (Chinese descriptions OK)
```python
def process_document(self, content: bytes) -> DocumentMetadata:
    """处理上传的文档

    Args:
        content: 文件内容（字节）

    Returns:
        文档元数据

    Raises:
        ValueError: 不支持的文件类型
    """
```

### TypeScript (chatbot-ui/)

**Imports** - Auto-sorted by Prettier; use `@/` path aliases
```typescript
"use client"

import { FC, useState } from "react"
import { Button } from "@/components/ui/button"
import { KnowledgeBase } from "@/types/knowledge-base"
```

**Naming Conventions**
| Element | Style | Example |
|---------|-------|---------|
| Components | PascalCase | `KnowledgeBaseItem` |
| Files | kebab-case | `knowledge-base-item.tsx` |
| Functions | camelCase | `handleUploadClick` |
| Interfaces | PascalCase | `KnowledgeBaseProps` |

**Components** - Use `FC` type with explicit props interface
```typescript
interface Props {
  knowledgeBase: KnowledgeBase
  onDelete: () => void
}

export const KnowledgeBaseItem: FC<Props> = ({ knowledgeBase, onDelete }) => {
  ...
}
```

**Error Handling** - Try-catch with typed errors
```typescript
try {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed: ${response.status}`)
} catch (error: any) {
  console.error("API error:", error)
  return new Response(JSON.stringify({ error: error.message }), { status: 500 })
}
```

**Formatting** - Prettier config: no semicolons, double quotes, 2-space indent

## Key Directories

```
backend/
├── app/
│   ├── main.py           # FastAPI entry, all API endpoints
│   ├── config.py         # Settings (env vars)
│   ├── agents/           # AgentScope agents (researcher, analyst, etc.)
│   └── rag/              # RAG module (knowledge_service, schemas)
├── qdrant_data/          # Vector DB storage
└── requirements.txt

chatbot-ui/
├── app/
│   ├── api/              # Next.js API routes (proxy to backend)
│   └── [locale]/         # i18n pages
├── components/
│   ├── chat/             # Chat UI components
│   ├── knowledge/        # Knowledge base components
│   └── ui/               # Shared UI primitives
├── lib/                  # Utilities, API helpers
├── types/                # TypeScript interfaces
└── __tests__/            # Jest + Playwright tests
```

## Environment Variables

**Backend** (`backend/.env`):
```
DASHSCOPE_API_KEY=sk-xxx       # Alibaba Cloud LLM
FIRECRAWL_API_KEY=fc-xxx       # Web scraping
```

**Frontend** (`chatbot-ui/.env.local`):
```
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_SUPABASE_URL=...
NEXT_PUBLIC_SUPABASE_ANON_KEY=...
```

## Important Patterns

1. **API Routing**: Frontend `/api/*` routes proxy to backend `:8000`
2. **RAG Flow**: Upload → TextReader chunks → DashScope embedding → Qdrant store
3. **Agent Pipeline**: Researcher → Analyst → Compiler (streaming output)
4. **State Management**: React Context (`ChatbotUIContext`) for global state
