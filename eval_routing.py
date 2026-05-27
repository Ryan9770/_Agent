"""
라우팅 eval 파이프라인

supervisor가 각 질문을 '올바른 전문가에게 위임했는지'를 자동 채점한다.
도구 선택 eval과 골격이 동일 — 평가 대상이 단일 에이전트 → supervisor로,
'기대 도구' → '기대 전문가'로 바뀐 것뿐.

전문가가 도구(ask_research_expert / ask_math_expert)로 감싸져 있으므로,
supervisor의 tool_calls에서 어떤 전문가를 불렀는지 추출하면 된다.
"""

import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
import os 
from dotenv import load_dotenv


load_dotenv()

AI_SERVER = os.getenv("AI_SERVER")

# ══════════════════════════════════════════════════════
#  모델
# ══════════════════════════════════════════════════════
model = ChatOpenAI(
    base_url=AI_SERVER,
    api_key="not-needed",
    model="local-model",
    temperature=0,
)


# ── 레벨 1: 진짜 도구 ──
@tool
def web_search(query: str) -> str:
    """Search the public web for information."""
    try:
        resp = requests.get("http://localhost:8080/search",
                            params={"q": query, "format": "json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return f"Error: search failed ({e})"
    results = data.get("results", [])[:3]
    return "\n".join(f"{r.get('title','')}: {r.get('content','')[:100]}" for r in results) or "No results"


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


# ── 레벨 2: 전문가 에이전트 ──
research_agent = create_agent(
    model, [web_search],
    system_prompt="You are a research expert. Use web_search to find facts.",
)
math_agent = create_agent(
    model, [calculator],
    system_prompt="You are a math expert. Always use calculator for arithmetic.",
)


# ── 레벨 3: 전문가를 도구로 감싸기 ──
@tool
def ask_research_expert(task: str) -> str:
    """Delegate a research/fact-finding task to the research expert.
    Use for anything requiring web search or current factual information."""
    result = research_agent.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content


@tool
def ask_math_expert(task: str) -> str:
    """Delegate an arithmetic/calculation task to the math expert.
    Use for any computation, summation, or numeric work."""
    result = math_agent.invoke({"messages": [{"role": "user", "content": task}]})
    return result["messages"][-1].content


# ── 레벨 4: supervisor ──
supervisor = create_agent(
    model, [ask_research_expert, ask_math_expert],
    system_prompt=(
        "You are a supervisor coordinating two experts:\n"
        "- ask_research_expert: for finding facts via web search.\n"
        "- ask_math_expert: for any arithmetic/calculation.\n"
        "Delegate each sub-task to the right expert. Do NOT answer research or "
        "math yourself — always delegate. For arithmetic, ALWAYS use the math "
        "expert even if it looks simple."
    ),
)


# ══════════════════════════════════════════════════════
#  1) 테스트 케이스: 질문 + 기대 전문가
# ══════════════════════════════════════════════════════
TEST_CASES = [
    {"q": "애플의 최신 직원 수가 몇 명이야?",            "expected": {"ask_research_expert"}},
    {"q": "345 곱하기 678을 계산해줘",                  "expected": {"ask_math_expert"}},
    {"q": "최신 쿠버네티스 버전이 뭐야?",               "expected": {"ask_research_expert"}},
    {"q": "(100 + 250) 나누기 7은 얼마야?",            "expected": {"ask_math_expert"}},
    {"q": "비트코인 현재 가격 알려줘",                  "expected": {"ask_research_expert"}},
    {"q": "9876 빼기 1234는?",                         "expected": {"ask_math_expert"}},
    # 복합: 둘 다 위임해야 하는 케이스
    {"q": "애플과 마이크로소프트 직원 수를 찾아서 합을 계산해줘",
     "expected": {"ask_research_expert", "ask_math_expert"}},
]


# ══════════════════════════════════════════════════════
#  2) 실행 + 3) 위임 추출
# ══════════════════════════════════════════════════════
def extract_delegated_experts(result) -> set:
    """supervisor가 호출한 전문가(도구로 감싼) 이름을 모은다."""
    delegated = set()
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            if tc["name"] in ("ask_research_expert", "ask_math_expert"):
                delegated.add(tc["name"])
    return delegated


def run_one(case):
    try:
        result = supervisor.invoke({"messages": [{"role": "user", "content": case["q"]}]})
        delegated = extract_delegated_experts(result)
    except Exception as e:
        return {"delegated": set(), "passed": False, "error": str(e)}
    expected = case["expected"]
    passed = expected.issubset(delegated)
    return {"delegated": delegated, "passed": passed, "error": None}


# ══════════════════════════════════════════════════════
#  4) 집계 + 출력
# ══════════════════════════════════════════════════════
def run_eval():
    print("=" * 60)
    print("라우팅 eval (supervisor 위임 정확도)")
    print("=" * 60)

    passed_count = 0
    for i, case in enumerate(TEST_CASES, 1):
        r = run_one(case)
        mark = "✅" if r["passed"] else "❌"
        if r["passed"]:
            passed_count += 1
        exp = ", ".join(sorted(case["expected"]))
        got = ", ".join(sorted(r["delegated"])) if r["delegated"] else "(없음)"
        print(f"\n{mark} [{i}] {case['q']}")
        print(f"     기대: {{{exp}}}")
        print(f"     실제: {{{got}}}")
        if r["error"]:
            print(f"     ⚠️ 에러: {r['error']}")

    total = len(TEST_CASES)
    acc = passed_count / total * 100
    print("\n" + "=" * 60)
    print(f"결과: {passed_count}/{total} 통과  (라우팅 정확도 {acc:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()