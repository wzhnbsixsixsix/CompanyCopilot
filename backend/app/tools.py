from __future__ import annotations

import asyncio
from typing import Any

import httpx
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def firecrawl_search(
    query: str,
    api_url: str,
    api_key: str | None,
    limit: int = 5,
) -> ToolResponse:
    """Search web content via Firecrawl.

    Args:
        query (str): Search keywords for due diligence.
        api_url (str): Firecrawl API base url.
        api_key (str): Firecrawl API key.
        limit (int): Max number of search results.
    """
    endpoint = f"{api_url.rstrip('/')}/v1/search"
    payload: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "lang": "en",
    }

    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(endpoint, headers=_headers(api_key), json=payload)
        response.raise_for_status()
        data = response.json()

    return ToolResponse(
        content=[TextBlock(type="text", text=str(data))],
    )


async def demo_sleep_tool(tag: str, wait_seconds: int = 2) -> ToolResponse:
    """A demo async tool for parallel tool-calls.

    Args:
        tag (str): Task tag.
        wait_seconds (int): Sleep time to simulate IO task.
    """
    await asyncio.sleep(wait_seconds)
    return ToolResponse(
        content=[TextBlock(type="text", text=f"{tag} done in {wait_seconds}s")],
    )
