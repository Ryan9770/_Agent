"""
멀티에이전트 supervisor (tool-calling 방식, 라이브러리 없이)

구조:
    supervisor (관리자)
      ├─ 도구로 감싼: research_agent  (web_search 보유)
      └─ 도구로 감싼: math_agent      (calculator 보유)

핵심 발상: 전문 에이전트를 supervisor의 '도구'로 감싼다.
supervisor 입장에선 전문가 호출이 calculator 호출과 똑같아 보인다.
본인이 아는 tool calling 그대로인데, 그 도구가 함수가 아니라 '또 다른 에이전트'.
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
#  모델
# ══════════════════════════════════════════════════════
model = ChatOpenAI(
    base_url=AI_SERVER,
    api_key="not-needed",
    model="local-model",
    temperature=0,
)


# ══════════════════════════════════════════════════════
#  레벨 1: 전문가들이 쓸 '진짜 도구'
# ══════════════════════════════════════════════════════
@tool
def web_search(query: str) -> str:
    """Search the public web for information. Returns titles, snippets, URLs."""
    try:
        resp = requests.get("http://localhost:8080/search",
                            params={"q": query, "format": "json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return f"Error: search failed ({e})"
    results = data.get("results", [])[:5]
    if not results:
        return f"No web results found for: {query}"
    return "\n\n".join(
        f"{i}. {r.get('title','')}\n   {r.get('content','').strip()}\n   ({r.get('url','')})"
        for i, r in enumerate(results, 1)
    )


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. e.g. '67317 + 164000'"""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════
#  레벨 2: 전문가 에이전트들 (각자 도구를 가진 독립 에이전트)
# ══════════════════════════════════════════════════════
research_agent = create_agent(
    model, [web_search],
    system_prompt="You are a research expert. Use web_search to find factual "
                  "information. Report the facts you found clearly and concisely.",
)

math_agent = create_agent(
    model, [calculator],
    system_prompt="You are a math expert. Use the calculator tool for any "
                  "arithmetic. Always compute with the tool, never in your head.",
)


# ══════════════════════════════════════════════════════
#  레벨 3: 전문가를 '도구로 감싸기'
#  여기가 핵심 — 에이전트 호출을 일반 함수 도구처럼 보이게 만든다.
# ══════════════════════════════════════════════════════
@tool
def ask_research_expert(task: str) -> str:
    """Delegate a research/fact-finding task to the research expert.
    Use for anything requiring web search or current factual information."""
    print(f"   📨 → 리서치 전문가에게: {task[:80]}")
    result = research_agent.invoke({"messages": [{"role": "user", "content": task}]})
    answer = result["messages"][-1].content
    print(f"   📩 ← 리서치 전문가: {answer[:80]}...")
    return answer


@tool
def ask_math_expert(task: str) -> str:
    """Delegate an arithmetic/calculation task to the math expert.
    Use for any computation, summation, or numeric work."""
    print(f"   📨 → 계산 전문가에게: {task[:80]}")
    result = math_agent.invoke({"messages": [{"role": "user", "content": task}]})
    answer = result["messages"][-1].content
    print(f"   📩 ← 계산 전문가: {answer[:80]}...")
    return answer


# ══════════════════════════════════════════════════════
#  레벨 4: supervisor (전문가 도구들을 지휘)
# ══════════════════════════════════════════════════════
supervisor = create_agent(
    model, [ask_research_expert, ask_math_expert],
    system_prompt=(
        "You are a supervisor coordinating two experts:\n"
        "- ask_research_expert: for finding facts via web search.\n"
        "- ask_math_expert: for any arithmetic/calculation.\n"
        "Break the user's request into sub-tasks, delegate each to the right "
        "expert, then combine their results into a final answer. "
        "Do NOT do research or math yourself — always delegate."
    ),
)


if __name__ == "__main__":
    # 리서치(직원 수 찾기)와 계산(합산)을 둘 다 요구하는 질문
    question = "너의 최신 학습 지점은 언제야?"
    print(f"🎯 질문: {question}\n")
    result = supervisor.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"\n{'='*50}\n🤖 supervisor 최종 답변:\n{result['messages'][-1].content}")