"""
동적 재계획 에이전트 (LangGraph)

정적 plan-execute(한 번 계획 → 끝까지 실행)와 달리,
한 스텝 실행할 때마다 replan 노드가 '남은 계획을 고칠지 / 끝낼지' 판단한다.

그래프 흐름:
    START → planner → executor → replan → (executor 또는 END)

State 필드:
    input       원래 목표
    plan        남은 계획 스텝들 (실행하면서 줄어듦)
    past_steps  (스텝, 결과) 누적 기록  ← operator.add 로 누적
    response    최종 답 (채워지면 종료)
"""

import operator
from typing import TypedDict, Annotated, List, Tuple, Union, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

import requests
import datetime
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

# ══════════════════════════════════════════════════════
#  도구 (web_search + calendar 영역)
# ══════════════════════════════════════════════════════
@tool
def web_search(query: str) -> str:
    """Search the public web for current or external information. Returns titles, snippets, URLs."""
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
def get_time() -> str:
    """Get the current UTC time. Useful to know what 'today', 'next week' means."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# (캘린더는 google 인증이 필요하므로, 이 실험 파일에서는 데모용 스텁을 둔다.
#  실제 연동 시 본인 agent_pkg/tools/calendar.py의 함수를 import해서 교체하면 된다.)
@tool
def list_calendar_events(days_ahead: int = 7) -> str:
    """List the user's upcoming calendar events within the given number of days."""
    # TODO: 실제 캘린더 연동으로 교체. 지금은 데모 데이터.
    return ("- 2026-05-27 10:00 | 팀 회의\n"
            "- 2026-05-29 15:00 | 치과 예약")


tools = [web_search, get_time, list_calendar_events]

# executor 노드가 쓸 ReAct 에이전트 (한 스텝을 실제로 수행하는 일꾼)
executor_agent = create_react_agent(model, tools)


# ══════════════════════════════════════════════════════
#  구조화된 출력: Plan / Response
# ══════════════════════════════════════════════════════
class Plan(BaseModel):
    """단계별 계획. 아직 할 일이 남았을 때 사용."""
    steps: List[str] = Field(
        description="따라야 할 단계들. 각 단계는 구체적이고 한 번에 실행 가능해야 함. "
                    "번호(1. 2.)를 붙이지 말 것."
    )


class Response(BaseModel):
    """사용자에게 줄 최종 답변. 더 할 일이 없을 때 사용."""
    response: str = Field(description="최종 답변 텍스트")


class Act(BaseModel):
    """다음 행동: 계획을 더 진행(Plan)하거나, 최종 답을 내거나(Response)."""
    action: Union[Plan, Response] = Field(
        description="더 할 일이 있으면 Plan을, 목표를 달성해 끝낼 수 있으면 Response를 선택."
    )


class Critique(BaseModel):
    """초안에 대한 비평과 다음 행동 결정."""
    verdict: Literal["need_research", "need_rewrite", "good_enough"] = Field(
        description="need_research: 정보가 부족해 더 조사 필요. "
                    "need_rewrite: 정보는 충분하나 글의 질이 부족. "
                    "good_enough: 충분히 좋음, 종료."
    )
    feedback: str = Field(description="구체적인 개선 지시 (무엇이 왜 부족한지)")

class ReflectState(TypedDict):
    objective: str                                    # 원래 요청
    research_notes: Annotated[List[str], operator.add]  # 모은 정보 (누적)
    draft: str                                        # 현재 초안
    feedback: str                                     # 직전 비평 (draft가 참고)
    iterations: int                                   # 반복 횟수 (회로 차단기용)
    final: str                                        # 최종 답 (채워지면 종료)

# ══════════════════════════════════════════════════════
#  State
# ══════════════════════════════════════════════════════
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]  # 누적
    response: str


# ══════════════════════════════════════════════════════
#  노드들
# ══════════════════════════════════════════════════════
TOOL_DESC = "web_search(웹 검색), get_time(현재 시각), list_calendar_events(다가오는 일정)"

planner_model = model.with_structured_output(Plan)
replanner_model = model.with_structured_output(Act)


def plan_node(state: PlanExecute):
    """목표를 받아 초기 계획을 세운다 (본인 make_plan에 해당)."""
    prompt = (
        f"You are a planner. Make a concise step-by-step plan to achieve the objective.\n"
        f"Available tools: {TOOL_DESC}.\n"
        f"Each step should be concrete and executable with one tool call. "
        f"Do not number the steps.\n\n"
        f"Objective: {state['input']}"
    )
    plan = planner_model.invoke(prompt)
    print(f"\n📋 초기 계획:")
    for i, s in enumerate(plan.steps, 1):
        print(f"   {i}. {s}")
    return {"plan": plan.steps}


def execute_node(state: PlanExecute):
    """계획의 '첫 스텝 하나'만 ReAct 에이전트로 실행한다."""
    if not state["plan"]:
        return {}
    task = state["plan"][0]
    print(f"\n⚙️  실행: {task}")
    result = executor_agent.invoke({"messages": [{"role": "user", "content": task}]})
    answer = result["messages"][-1].content
    print(f"   → {answer[:150]}{'...' if len(answer) > 150 else ''}")
    # 이 스텝을 past_steps에 누적 (operator.add 덕분에 더해짐)
    return {"past_steps": [(task, answer)]}


def replan_node(state: PlanExecute):
    """지금까지 한 것을 보고, 남은 계획을 고치거나 최종 답을 낸다."""
    done = "\n".join(f"- {step}: {result}" for step, result in state["past_steps"])
    remaining = "\n".join(f"- {s}" for s in state["plan"][1:]) or "(없음)"
    prompt = (
        f"Objective: {state['input']}\n\n"
        f"Already completed steps and their results:\n{done}\n\n"
        f"Remaining planned steps:\n{remaining}\n\n"
        f"Decide the next action:\n"
        f"- If the objective is now fully achieved, return a Response with the final answer.\n"
        f"- Otherwise, return a Plan with the remaining steps still needed "
        f"(you may revise them based on what you learned). Do not repeat completed steps."
    )
    output = replanner_model.invoke(prompt)
    if isinstance(output.action, Response):
        print(f"\n✅ 재계획 판단: 종료")
        return {"response": output.action.response}
    else:
        print(f"\n🔄 재계획 판단: 계속 (남은 스텝 {len(output.action.steps)}개)")
        for i, s in enumerate(output.action.steps, 1):
            print(f"   {i}. {s}")
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    """response가 채워졌으면 종료, 아니면 다시 실행."""
    if state.get("response"):
        return END
    return "executor"


# ══════════════════════════════════════════════════════
#  그래프 조립
# ══════════════════════════════════════════════════════
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_node)
workflow.add_node("executor", execute_node)
workflow.add_node("replan", replan_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "replan")
workflow.add_conditional_edges("replan", should_end, {"executor": "executor", END: END})

app = workflow.compile()


if __name__ == "__main__":
    objective = "내 다음 주 일정을 확인하고, 일정이 있는 날 중 하나를 골라 그날 무슨 요일인지 알려줘"
    print(f"🎯 목표: {objective}")
    result = app.invoke(
        {"input": objective, "plan": [], "past_steps": [], "response": ""},
        config={"recursion_limit": 20},   # 회로 차단기: 무한 루프 방지
    )
    print(f"\n{'='*50}\n🤖 최종 답변:\n{result['response']}")