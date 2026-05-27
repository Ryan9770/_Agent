# lg_agent.py — LangGraph 저수준 포팅 (본인 run_agent를 그래프로)
from langchain_openai import ChatOpenAI

# 본인 config.py의 값 그대로
model = ChatOpenAI(
    base_url="http://10.1.10.111:8080/v1",
    api_key="not-needed",
    model="local-model",
    temperature=0.3,
)

from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. e.g. '12 * (3 + 4)'"""
    print(f"\n[calculator] {expression} 계산 중...")
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

# 본인 web_search도 똑같이 — docstring이 곧 description이 됨
import requests
@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the public web for current or external information. Returns titles, snippets, URLs."""
    print(f"\n[web_search] {query} 검색 중...")
    try:
        resp = requests.get("http://localhost:8080/search",
                            params={"q": query, "format": "json"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return f"Error: search failed ({e})"
    results = data.get("results", [])[:num_results]
    if not results:
        return f"No web results found for: {query}"
    return "\n\n".join(
        f"{i}. {r.get('title','')}\n   {r.get('content','').strip()}\n   ({r.get('url','')})"
        for i, r in enumerate(results, 1)
    )

# lg_agent.py 에 이어서 — State 확장 + 커스텀 도구 노드
from langchain_core.messages import ToolMessage
import json
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import tools_condition
from typing import TypedDict, Annotated, List, Tuple, Union
import operator

class PlanExecute(TypedDict):
    input: str                                      # 원래 목표
    plan: List[str]                                 # 남은 계획 스텝들
    past_steps: Annotated[List[Tuple], operator.add]  # (스텝, 결과) 누적
    response: str                                   # 최종 답 (있으면 종료)



tools = [calculator, web_search]   # 일단 두 개로 시작

# MessagesState를 확장: messages + seen_calls
class AgentState(MessagesState):
    seen_calls: list      # 이번 실행에서 실행한 (도구명, 인자) 기록

# 도구 이름 → 함수 매핑 (본인의 TOOL_FUNCS와 같은 것)
tools_by_name = {t.name: t for t in tools}


def guarded_tool_node(state: AgentState):
    """
    ToolNode를 대체하는 커스텀 노드.
    도구를 실행하기 전에 '이번 실행에서 이미 부른 호출인지' 검사한다.
    (본인 run_agent의 중복 차단 로직을 노드 안으로 옮긴 것)
    """
    last_message = state["messages"][-1]      # 방금 agent 노드가 만든 tool_calls
    seen = state.get("seen_calls", [])
    outputs = []

    for call in last_message.tool_calls:
        name = call["name"]
        args = call["args"]
        call_key = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

        # ── 중복 차단 (본인의 _seen_calls_this_turn 검사) ──
        if call_key in seen:
            print(f"[guard] ⛔ 중복 차단: {name}({args})")
            result = ("You already made this exact call. Do not repeat it. "
                      "Use the previous result or answer now.")
        else:
            seen.append(call_key)
            print(f"[guard] 🔧 {name}({args})")
            result = tools_by_name[name].invoke(args)   # 실제 도구 실행

        # 도구 결과를 ToolMessage로 (LangChain 포맷)
        outputs.append(ToolMessage(
            content=str(result),
            name=name,
            tool_call_id=call["id"],
        ))

    # messages에 결과들 추가 + seen_calls 갱신해서 State에 돌려줌
    return {"messages": outputs, "seen_calls": seen}

model_with_tools = model.bind_tools(tools)   # 본인의 chat(messages, tools=TOOLS)

# "agent" 노드: LLM 호출 = 본인 run_agent의 chat() 부분
def call_model(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(AgentState)                    # ← MessagesState 대신 AgentState
builder.add_node("agent", call_model)
builder.add_node("tools", guarded_tool_node)        # ← ToolNode 대신 커스텀 노드
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()

if __name__ == "__main__":
    # 중복을 유도하기 쉬운 질문 (특정 날짜 뉴스 등, 모델이 재검색하려는)
    result = graph.invoke({
        "messages": [{"role": "user", "content": "오늘 한국 주요 뉴스 5개를 웹에서 검색해서 알려줘"}],
        "seen_calls": [],     # ← 초기 상태: 빈 기록
    })
    print("\n" + result["messages"][-1].content)