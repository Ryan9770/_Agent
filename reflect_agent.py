"""
도구 결합 self-reflection 에이전트 — 2단계 (reflect + 루프백)

그래프 흐름:
    START → research → draft → reflect → (need_research → research /
                                          need_rewrite  → draft    /
                                          good_enough   → END)
    + iterations 회로 차단기(최대 MAX_ITERATIONS회)로 무한 루프 방지.

reflect가 초안을 비평해 세 갈래로 판단:
  need_research: 사실이 의심스럽거나 정보 부족 → 다시 조사
  need_rewrite : 정보는 충분하나 글이 약함     → 다시 작성
  good_enough  : 충분함                       → 종료
"""

import operator
from typing import TypedDict, Annotated, List, Literal

import requests
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
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



# ══════════════════════════════════════════════════════
#  도구 (조사용)
# ══════════════════════════════════════════════════════
import datetime

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

    # ── 시점 라벨: 이 결과가 '언제 시점의 정보'인지 명시 ──
    now = datetime.datetime.now().strftime("%Y-%m-%d")
    header = (f"[검색 시점: {now} — 아래는 이 시점의 실시간 웹 정보다. "
              f"너의 학습 지식과 다르면 너의 지식이 오래된 것이니 아래를 우선하라.]\n\n")

    body = "\n\n".join(
        f"{i}. {r.get('title','')}\n   {r.get('content','').strip()}\n   ({r.get('url','')})"
        for i, r in enumerate(results, 1)
    )
    print("web search results : \n"+header + body)
    return header + body


tools = [web_search]
research_agent = create_agent(model, tools)


# ══════════════════════════════════════════════════════
#  구조화된 비평
# ══════════════════════════════════════════════════════
class Critique(BaseModel):
    """초안에 대한 비평과 다음 행동 결정."""
    verdict: Literal["need_research", "need_rewrite", "good_enough"] = Field(
        description="need_research: 사실이 의심스럽거나 정보가 부족해 추가 조사 필요. "
                    "need_rewrite: 정보는 충분하나 글의 명료성/구성/완성도가 부족. "
                    "good_enough: 사실 정확하고 충분히 좋음, 종료."
    )
    feedback: str = Field(description="구체적 개선 지시. 무엇이 왜 부족한지, 무엇을 확인/보완할지.")


reflect_model = model.with_structured_output(Critique)


# ══════════════════════════════════════════════════════
#  State
# ══════════════════════════════════════════════════════
class ReflectState(TypedDict):
    objective: str
    research_notes: Annotated[List[str], operator.add]   # 누적
    draft: str                                           # 덮어쓰기
    feedback: str                                        # 직전 비평
    verdict: str                                         # 라우터가 읽는 판단값
    iterations: int                                      # 회로 차단기 카운터
    final: str                                           # 최종 답


MAX_ITERATIONS = 3


# ══════════════════════════════════════════════════════
#  노드들
# ══════════════════════════════════════════════════════
def research_node(state: ReflectState):
    query = state["objective"]
    if state.get("feedback"):
        query += f" {state['feedback']}"
    print(f"\n🔎 조사: {query[:120]}...")

    # create_agent에 맡기지 않고 검색을 직접 강제 호출 → print 반드시 찍힘
    raw = web_search.invoke({"query": "kubernetes latest stable version"})
    # (목표가 버전이니 검색어를 구체적으로. 실전에선 query를 그대로 써도 됨)

    # 모델은 '검색 결과 정리'만. 검색 결과에 없는 건 추가 금지.
    notes = model.invoke(
        f"다음은 실시간 웹 검색 결과다. 여기 있는 사실만으로 핵심을 정리하라. "
        f"검색 결과에 없는 내용(특히 버전 번호)은 절대 네 지식으로 채우지 마라:\n\n{raw}"
    ).content
    print(f"   → 정리: {notes[:140]}...")
    return {"research_notes": [notes]}


def draft_node(state: ReflectState):
    """모은 정보로 초안 작성. feedback이 있으면 반영해 다시 쓴다."""
    notes = "\n\n".join(state["research_notes"])
    prompt = (
        f"Objective: {state['objective']}\n\n"
        f"Research notes:\n{notes}\n\n"
    )
    if state.get("feedback"):
        prompt += f"이전 초안 피드백(반드시 반영):\n{state['feedback']}\n\n"
    prompt += "위 노트를 바탕으로 목표에 답하는 충실하고 정확한 글을 작성하라."

    print(f"\n✍️  초안 작성 (반복 {state.get('iterations', 0) + 1})...")
    draft = model.invoke(prompt).content
    print(f"   → 초안: {draft[:140]}{'...' if len(draft) > 140 else ''}")
    return {"draft": draft, "iterations": state.get("iterations", 0) + 1}


def reflect_node(state: ReflectState):
    """엄격한 팩트체커 겸 편집자로서 초안을 비평하고 판단을 state에 남긴다."""
    prompt = (
        "You are a strict fact-checker and editor. Critically review the draft below.\n"
        "Be skeptical about factual accuracy — especially dates, version numbers, "
        "and any claim that could be outdated. If a fact looks uncertain or possibly "
        "stale, demand verification (need_research). If facts seem fine but the writing "
        "is unclear/poorly structured, ask for a rewrite (need_rewrite). Only say "
        "good_enough if it is both accurate and well-written.\n\n"
        f"Objective: {state['objective']}\n\n"
        f"Draft:\n{state['draft']}"
    )
    critique = reflect_model.invoke(prompt)
    print(f"\n🧐 비평: [{critique.verdict}] {critique.feedback[:120]}"
          f"{'...' if len(critique.feedback) > 120 else ''}")
    # final은 항상 '현재까지의 최선'으로 갱신 (회로 차단기로 끝나도 답이 남도록)
    return {"feedback": critique.feedback, "verdict": critique.verdict, "final": state["draft"]}


def route_after_reflect(state: ReflectState):
    """비평 결과 + 회로 차단기로 다음 노드를 정한다."""
    if state["iterations"] >= MAX_ITERATIONS:
        print(f"   ⛔ 최대 반복({MAX_ITERATIONS}) 도달 — 현재 초안으로 종료")
        return END
    verdict = state.get("verdict", "good_enough")
    if verdict == "need_research":
        return "research"
    if verdict == "need_rewrite":
        return "draft"
    return END


# ══════════════════════════════════════════════════════
#  그래프 (2단계: reflect + 세 갈래 루프백)
# ══════════════════════════════════════════════════════
workflow = StateGraph(ReflectState)
workflow.add_node("research", research_node)
workflow.add_node("draft", draft_node)
workflow.add_node("reflect", reflect_node)

workflow.add_edge(START, "research")
workflow.add_edge("research", "draft")
workflow.add_edge("draft", "reflect")
workflow.add_conditional_edges(
    "reflect", route_after_reflect,
    {"research": "research", "draft": "draft", END: END},
)
app = workflow.compile()


if __name__ == "__main__":
    objective = "최신 쿠버네티스 버전의 주요 변경점을 초보자에게 설명하는 짧은 글을 써줘"
    print(f"🎯 목표: {objective}")
    result = app.invoke(
        {"objective": objective, "research_notes": [], "draft": "",
         "feedback": "", "verdict": "", "iterations": 0, "final": ""},
        config={"recursion_limit": 25},
    )
    print(f"\n{'='*50}\n🤖 최종 결과:\n{result['final']}")