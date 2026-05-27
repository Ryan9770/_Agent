"""계획 (Plan-and-Execute): 계획을 세운 뒤 그 계획을 길잡이로 run_agent 실행."""

from .llm import chat
from .tools import TOOL_FUNCS
from .agent import run_agent, reset_turn_state
from .memory import compress_memory


def make_plan(user_message, verbose=True):
    """도구를 주지 않고, 요청을 푸는 단계별 계획만 생성한다."""
    tool_names = ", ".join(TOOL_FUNCS.keys())
    planning_messages = [
        {"role": "system", "content":
            "You are a planning module. Given a user request, produce a SHORT "
            "numbered plan (2-5 steps) describing how to fulfill it. "
            f"Available tools you can plan around: {tool_names}. "
            "For each step, note which tool (if any) would be used. "
            "Output ONLY the plan, no preamble, no answer yet. "
            "If the request is trivial (one step, no tools), say 'DIRECT' and nothing else."},
        {"role": "user", "content": user_message},
    ]
    plan = chat(planning_messages).choices[0].message.content.strip()
    if verbose:
        print(f"\n📋 계획:\n{plan}\n")
    return plan


def run_planning_agent(messages, user_input, verbose=True):
    """계획을 세운 뒤, 그 계획을 길잡이로 run_agent를 실행한다."""
    reset_turn_state()   # 이번 턴의 중복 추적 초기화

    plan = make_plan(user_input, verbose=verbose)
    messages.append({"role": "user", "content": user_input})

    if plan != "DIRECT":
        messages.append({
            "role": "system",
            "content": f"Follow this plan to answer the user (adapt if needed):\n{plan}",
        })

    answer, messages = run_agent(messages, verbose=verbose)

    # 일회용 계획 힌트는 대화 기록에서 제거(메모리 오염 방지)
    messages = [m for m in messages
                if not (m.get("role") == "system"
                        and str(m.get("content", "")).startswith("Follow this plan"))]

    messages = compress_memory(messages, verbose=verbose)
    return answer, messages
