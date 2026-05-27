"""메모리 — 하이브리드 압축(요약 + 도구결과 비우기)."""

from . import config
from .llm import chat


def compress_memory(messages, verbose=True):
    """
    너무 길면: ① 오래된 구간의 긴 tool 결과를 비우고 ② 그 구간을 LLM으로 요약.
    system 메시지와 최근 KEEP_RECENT_MSGS개는 보존.
    """
    if len(messages) <= config.COMPRESS_THRESHOLD:
        return messages

    system_msg = messages[0]
    recent = messages[-config.KEEP_RECENT_MSGS:]
    middle = messages[1:-config.KEEP_RECENT_MSGS]
    if not middle:
        return messages

    # 안전장치: 최근 구간이 tool 메시지로 시작하면(짝인 assistant가 잘려나가면)
    # API가 에러를 내므로, 그 tool 메시지를 요약 대상(middle)으로 되돌린다.
    while recent and recent[0].get("role") == "tool":
        middle.append(recent.pop(0))
    if not recent:
        return messages

    if verbose:
        print(f"   🗜️  메모리 압축 시작: 중간 {len(middle)}개 메시지 → 요약")

    # ① 긴 tool 결과 비우기
    for m in middle:
        if m.get("role") == "tool" and len(str(m.get("content", ""))) > 80:
            m["content"] = "[이전 도구 결과 생략됨]"

    # ② 요약
    transcript = []
    for m in middle:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            transcript.append(f"User: {content}")
        elif role == "assistant":
            if m.get("tool_calls"):
                names = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                transcript.append(f"Assistant: (도구 호출: {names}) {content}")
            else:
                transcript.append(f"Assistant: {content}")
        elif role == "tool":
            transcript.append(f"ToolResult: {content}")
    transcript_text = "\n".join(transcript)

    summary_resp = chat([
        {"role": "system", "content":
            "Summarize the following conversation excerpt concisely in Korean. "
            "Preserve key facts, decisions, commands, and any answers the user "
            "received, so the conversation can continue coherently. "
            "Be compact — a few sentences."},
        {"role": "user", "content": transcript_text},
    ])
    summary_msg = {
        "role": "system",
        "content": f"[이전 대화 요약]\n{summary_resp.choices[0].message.content}",
    }
    new_messages = [system_msg, summary_msg] + recent
    if verbose:
        print(f"   🗜️  압축 완료: {len(messages)}개 → {len(new_messages)}개")
    return new_messages
