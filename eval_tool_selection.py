"""
도구 선택 eval 파이프라인

에이전트가 각 질문에 '올바른 도구를 불렀는지'를 자동 채점한다.
그동안 눈으로 보던 '검색 건너뛰기' 같은 문제를 숫자로 잡는다.

파이프라인 4단계:
    1) 질문 + 기대 도구 세트 (테스트 케이스)
    2) 각 질문을 에이전트에 실행
    3) 실제 호출한 도구를 추출해 기대값과 비교 (채점)
    4) 정확도 집계 + 케이스별 상세 출력
"""
import os
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv


load_dotenv()

AI_SERVER = os.getenv("AI_SERVER")

# ══════════════════════════════════════════════════════
#  평가 대상 에이전트 (도구 3개를 가진 단일 에이전트)
# ══════════════════════════════════════════════════════
model = ChatOpenAI(
    base_url=AI_SERVER,
    api_key="not-needed",
    model="local-model",
    temperature=0,   # eval은 재현성이 중요 → 0으로 고정
)


@tool
def web_search(query: str) -> str:
    """Search the public web for current or external information (news, facts, prices, latest versions)."""
    try:
        resp = requests.get("http://localhost:8080/search",
                            params={"q": query, "format": "json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return f"Error: search failed ({e})"
    results = data.get("results", [])[:3]
    if not results:
        return f"No results for: {query}"
    return "\n".join(f"{r.get('title','')}: {r.get('content','')[:100]}" for r in results)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. e.g. '12 * (3 + 4)'"""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def get_time() -> str:
    """Get the current UTC time. Use when asked about the current date or time."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


TOOLS = [web_search, calculator, get_time]
agent = create_agent(
    model, TOOLS,
    system_prompt="You are a helpful assistant. Use the right tool for each task. "
                  "For current/external facts use web_search, for current time use get_time. "
                  "IMPORTANT: For ANY arithmetic, ALWAYS use the calculator tool — "
                  "even simple-looking calculations. Never compute in your head, "
                  "because mental math is error-prone.",   # ← 계산 건너뛰기 방지
)


# ══════════════════════════════════════════════════════
#  1) 테스트 케이스: 질문 + 기대 도구
#  expected_tools: 이 질문에서 반드시 불려야 하는 도구들(집합)
# ══════════════════════════════════════════════════════
TEST_CASES = [
    {"q": "지금 몇 시야?",                        "expected": {"get_time"}},
    {"q": "123 곱하기 456은 얼마야?",              "expected": {"calculator"}},
    {"q": "최신 쿠버네티스 버전이 뭐야?",          "expected": {"web_search"}},
    {"q": "오늘 한국 주요 뉴스 알려줘",            "expected": {"web_search"}},
    {"q": "9999 더하기 8888 계산해줘",            "expected": {"calculator"}},
    {"q": "비트코인 현재 시세 알려줘",             "expected": {"web_search"}},
    {"q": "(15 + 27) 나누기 6은?",                "expected": {"calculator"}},
    {"q": "지금 날짜가 며칠이야?",                 "expected": {"get_time"}},
]


# ══════════════════════════════════════════════════════
#  2) 실행 + 3) 도구 추출
# ══════════════════════════════════════════════════════
def extract_called_tools(result) -> set:
    """에이전트 결과 메시지에서 실제 호출된 도구 이름을 모은다."""
    called = set()
    for m in result["messages"]:
        for tc in getattr(m, "tool_calls", None) or []:
            called.add(tc["name"])
    return called


def run_one(case):
    """한 케이스를 실행하고 채점 결과를 반환."""
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": case["q"]}]})
        called = extract_called_tools(result)
    except Exception as e:
        return {"called": set(), "passed": False, "error": str(e)}

    expected = case["expected"]
    # 채점 기준: 기대 도구가 실제 호출 도구에 포함되면 통과
    # (다른 도구를 추가로 불러도 기대 도구만 있으면 OK로 본다 — 엄격히 하려면 == 사용)
    passed = expected.issubset(called)
    return {"called": called, "passed": passed, "error": None}


# ══════════════════════════════════════════════════════
#  4) 집계 + 출력
# ══════════════════════════════════════════════════════
def run_eval():
    print("=" * 60)
    print("도구 선택 eval")
    print("=" * 60)

    passed_count = 0
    for i, case in enumerate(TEST_CASES, 1):
        r = run_one(case)
        mark = "✅" if r["passed"] else "❌"
        if r["passed"]:
            passed_count += 1

        expected_str = ", ".join(sorted(case["expected"]))
        called_str = ", ".join(sorted(r["called"])) if r["called"] else "(없음)"
        print(f"\n{mark} [{i}] {case['q']}")
        print(f"     기대: {{{expected_str}}}  |  실제: {{{called_str}}}")
        if r["error"]:
            print(f"     ⚠️ 에러: {r['error']}")

    total = len(TEST_CASES)
    acc = passed_count / total * 100
    print("\n" + "=" * 60)
    print(f"결과: {passed_count}/{total} 통과  (정확도 {acc:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()