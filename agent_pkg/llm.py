"""LLM 호출 래퍼 — llama-server(OpenAI 호환)를 한 번 호출한다."""

from openai import OpenAI

from . import config

_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def chat(messages, tools=None):
    """messages로 한 번 호출. tools를 주면 tool calling 활성화."""
    kwargs = {"model": config.MODEL, "messages": messages, "temperature": config.TEMPERATURE}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return _client.chat.completions.create(**kwargs)
