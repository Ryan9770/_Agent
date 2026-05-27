"""에이전트 루프 (ReAct + 가드레일: 중복차단 / step한계 / 재시도).

중복차단은 두 겹:
 (A1) 완전 동일한 (도구, 인자) 반복 차단
 (A2) 의미 기반 차단 — 검색계 도구의 query가 직전 검색어들과
      임베딩 유사도로 너무 비슷하면(SEMANTIC_DUP_THRESHOLD) 차단.
      → 검색어만 살짝 바꿔 같은 검색을 반복하는 패턴을 잡는다.
"""

import json
import time

from . import config
from .llm import chat
from .tools import TOOLS, TOOL_FUNCS
from .embeddings import encode, cosine

# 검색어의 의미가 이 값 이상으로 비슷하면 '사실상 같은 검색'으로 본다(보수적).
SEMANTIC_DUP_THRESHOLD = 0.9
# 의미 비교를 적용할 검색계 도구들(query 인자를 갖는 도구)
_SEARCH_TOOLS = {"web_search", "search_docs", "search_calendar_events"}

# 턴 단위로만 살아야 하는 상태. planner가 매 턴 reset_turn_state()로 비운다.
_seen_calls_this_turn = set()        # 완전 동일 호출 추적 (A1)
_seen_queries_this_turn = []         # (도구명, 검색어, 검색어벡터) 목록 (A2)


def reset_turn_state():
    """새 턴 시작 시 중복 추적 상태를 비운다."""
    _seen_calls_this_turn.clear()
    _seen_queries_this_turn.clear()


def _semantic_duplicate(name, args):
    """검색계 도구의 query가 이번 턴의 기존 검색어와 의미적으로 너무 비슷하면
    (기존검색어, 유사도)를 반환, 아니면 None."""
    if name not in _SEARCH_TOOLS:
        return None
    query = args.get("query")
    if not query:
        return None
    q_vec = encode([query])[0]
    for prev_name, prev_query, prev_vec in _seen_queries_this_turn:
        if prev_name != name:
            continue
        sim = cosine(q_vec, prev_vec)
        if sim >= SEMANTIC_DUP_THRESHOLD:
            return (prev_query, sim)
    # 중복이 아니면 이번 검색어를 기록해두고 통과
    _seen_queries_this_turn.append((name, query, q_vec))
    return None


def _run_tool_with_retry(func, args, name, verbose, retries=2):
    """도구 실행. 결과가 'Error:'로 시작하면 일시 오류로 보고 점증 백오프 재시도."""
    for attempt in range(retries + 1):
        try:
            result = func(**args)
        except Exception as e:
            result = f"Error: tool '{name}' raised {type(e).__name__}: {e}"
        if not str(result).startswith("Error:"):
            return result
        if attempt < retries:
            if verbose:
                print(f"           ↻ 재시도 {attempt + 1}/{retries} ({name})")
            time.sleep(1.0 * (attempt + 1))
    return result


def run_agent(messages, max_steps=config.MAX_STEPS, verbose=True):
    """
    messages: 이번 user 메시지까지 들어 있는 대화 전체(system 포함).
    반환: (최종답변, 갱신된 messages)
    """
    for step in range(max_steps):
        is_last = (step == max_steps - 1)
        if is_last:
            messages.append({
                "role": "system",
                "content": "You have reached the step limit. "
                           "Answer NOW with the information you have. Do not call tools.",
            })
            resp = chat(messages)
        else:
            resp = chat(messages, tools=TOOLS)

        msg = resp.choices[0].message

        if not msg.tool_calls:
            if verbose:
                print(f"[step {step}] ✅ 최종 답변")
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, messages

        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments
            args = json.loads(raw_args)

            # (A1) 완전 동일 호출 차단
            call_key = (name, raw_args)
            if call_key in _seen_calls_this_turn:
                if verbose:
                    print(f"[step {step}] ⛔ 중복 차단(동일): {name}({args})")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": "You already made this exact call. Do not repeat it. "
                               "Use the previous result, try a different approach, or answer now.",
                })
                continue

            # (A2) 의미 기반 중복 차단
            dup = _semantic_duplicate(name, args)
            if dup is not None:
                prev_query, sim = dup
                if verbose:
                    print(f"[step {step}] ⛔ 중복 차단(의미 {sim:.2f}): "
                          f"'{args.get('query')}' ≈ '{prev_query}'")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"This search is semantically almost identical to your "
                               f"earlier search '{prev_query}' (similarity {sim:.2f}). "
                               f"Rephrasing won't help. Use the previous results, or "
                               f"answer with what you have.",
                })
                continue

            _seen_calls_this_turn.add(call_key)

            if verbose:
                print(f"[step {step}] 🔧 {name}({args})")

            func = TOOL_FUNCS.get(name)
            result = (f"Error: unknown tool '{name}'" if func is None
                      else _run_tool_with_retry(func, args, name, verbose))

            if verbose:
                preview = str(result)[:120].replace("\n", " ")
                print(f"           → {preview}{'...' if len(str(result)) > 120 else ''}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    messages.append({"role": "assistant", "content": "최대 스텝 도달 — 답을 못 냈습니다."})
    return "최대 스텝 도달 — 답을 못 냈습니다.", messages