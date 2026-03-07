from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str
    dashscope_model: str
    firecrawl_api_key: Optional[str]
    firecrawl_api_url: str


def get_settings() -> Settings:
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not dashscope_api_key:
        raise ValueError("DASHSCOPE_API_KEY is required.")

    return Settings(
        dashscope_api_key=dashscope_api_key,
        dashscope_model=os.getenv("DASHSCOPE_MODEL", "qwen-max"),
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY"),
        firecrawl_api_url=os.getenv("FIRECRAWL_API_URL", "https://api.firecrawl.dev"),
    )
