"""웹 검색 도구: web_search (로컬 SearXNG 인스턴스)."""

import requests

from .. import config
from .registry import tool


@tool(
    name="web_search",
    description="Search the public web for current, external, or real-time "
                "information (news, recent events, facts not in the user's "
                "local documents). Returns titles, snippets, and URLs.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "search query"},
            "num_results": {"type": "integer", "description": "default 5"},
        },
        "required": ["query"],
    },
)
def web_search(query: str, num_results: int = 5) -> str:
    """로컬 SearXNG로 웹을 검색해 상위 결과의 제목·요약·URL을 반환."""
    try:
        resp = requests.get(
            config.SEARXNG_URL, params={"q": query, "format": "json"}, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return "Error: search timed out. Try a simpler query."
    except requests.exceptions.RequestException as e:
        return f"Error: search failed ({e})"
    except ValueError:
        return "Error: could not parse search response as JSON."

    results = data.get("results", [])[:num_results]
    if not results:
        return f"No web results found for: {query}"
    out = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        content = r.get("content", "").strip()
        url = r.get("url", "")
        out.append(f"{i}. {title}\n   {content}\n   ({url})")
    return "\n\n".join(out)
