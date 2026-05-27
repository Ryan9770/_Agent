"""
병렬 실행 데모 (fan-out / fan-in)

여러 독립적인 작업을 동시에 실행하고 결과를 모은다.
본인 llama-server의 --parallel 4 덕분에 진짜 동시 처리된다.

그래프 구조:
        ┌─→ worker(애플) ─┐
   START┤  ┌→ worker(MS) ─┤→ combine → END
        └──┘  ...         ┘
   (fan-out: 동시 실행)   (fan-in: operator.add로 결과 취합)

핵심:
  - Send API로 한 노드에서 여러 worker를 동시에 띄운다 (fan-out)
  - results 필드에 operator.add reducer → 동시 결과가 덮어쓰지 않고 누적 (fan-in)
"""

import time
import operator
from typing import TypedDict, Annotated, List

import requests
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
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


def web_search_raw(query: str) -> str:
    try:
        resp = requests.get("http://localhost:8080/search",
                            params={"q": query, "format": "json"}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    results = data.get("results", [])[:3]
    return "\n".join(f"{r.get('title','')}: {r.get('content','')[:80]}" for r in results) or "No results"


# ══════════════════════════════════════════════════════
#  State
# ══════════════════════════════════════════════════════
class ParallelState(TypedDict):
    topics: List[str]                              # 조사할 주제들
    results: Annotated[List[str], operator.add]    # 결과 누적 (병렬 안전)
    summary: str


# ══════════════════════════════════════════════════════
#  worker 노드 — 주제 하나를 처리 (여러 개가 동시에 실행됨)
# ══════════════════════════════════════════════════════
def worker_node(state: dict):
    """단일 주제를 검색 + 요약. fan-out으로 여러 개가 병렬 실행된다.
    Send로 호출되므로 state는 {'topic': ...} 형태로 받는다."""
    topic = state["topic"]
    print(f"   ⚙️  [{time.strftime('%H:%M:%S')}] worker 시작: {topic}")
    raw = web_search_raw(topic)
    answer = model.invoke(
        f"다음 검색 결과에서 '{topic}'에 대한 핵심을 한 문장으로 요약하라:\n{raw}"
    ).content
    print(f"   ✅ [{time.strftime('%H:%M:%S')}] worker 완료: {topic}")
    # operator.add 덕분에 여러 worker의 결과가 results 리스트에 안전하게 합쳐짐
    return {"results": [f"[{topic}] {answer}"]}


# ══════════════════════════════════════════════════════
#  dispatch — fan-out: 주제마다 worker를 동시에 띄운다
# ══════════════════════════════════════════════════════
def dispatch(state: ParallelState):
    """각 주제를 worker로 동시 전송 (Send API).
    반환하는 Send 리스트만큼 worker가 병렬 실행된다."""
    print(f"\n🚀 fan-out: {len(state['topics'])}개 worker 동시 실행")
    return [Send("worker", {"topic": t}) for t in state["topics"]]


# ══════════════════════════════════════════════════════
#  combine — fan-in: 모인 결과를 종합
# ══════════════════════════════════════════════════════
def combine_node(state: ParallelState):
    print(f"\n🔗 fan-in: {len(state['results'])}개 결과 취합")
    joined = "\n".join(state["results"])
    summary = model.invoke(
        f"다음 조사 결과들을 종합해 간단히 정리하라:\n{joined}"
    ).content
    return {"summary": summary}


# ══════════════════════════════════════════════════════
#  그래프
# ══════════════════════════════════════════════════════
builder = StateGraph(ParallelState)
builder.add_node("worker", worker_node)
builder.add_node("combine", combine_node)
# START에서 조건부로 fan-out (dispatch가 Send 리스트 반환)
builder.add_conditional_edges(START, dispatch, ["worker"])
builder.add_edge("worker", "combine")
builder.add_edge("combine", END)
app = builder.compile()


if __name__ == "__main__":
    topics = [
        "최신 쿠버네티스 버전",
        "비트코인 현재 가격",
        "최신 파이썬 버전",
    ]
    print(f"🎯 {len(topics)}개 주제 병렬 조사: {topics}")

    start = time.time()
    result = app.invoke({"topics": topics, "results": [], "summary": ""})
    elapsed = time.time() - start

    print(f"\n{'='*55}")
    print(f"⏱️  총 소요: {elapsed:.1f}초 (worker {len(topics)}개 병렬)")
    print(f"{'='*55}")
    print(f"🤖 종합 결과:\n{result['summary']}")