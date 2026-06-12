"""
HITL 최소 예제 — interrupt() / Command(resume=...) 메커니즘만 깨끗이 보기

도구도 멀티에이전트도 없는 가장 단순한 그래프:
    START → propose → approve(여기서 멈춤) → execute → END

propose가 행동을 제안하면, approve 노드의 interrupt()가 그래프를 멈춘다.
사람이 yes/edit/no로 응답하면 Command(resume=...)로 재개.
응답에 따라 execute가 다른 행동을 한다.

핵심:
  - interrupt()는 Checkpointer가 있어야 작동한다 (멈춘 상태 저장용)
  - 같은 thread_id로 invoke해야 그 지점부터 재개됨
  - interrupt()의 반환값 = 사람이 Command(resume=...)로 보낸 값

설치 필요: pip install langgraph-checkpoint-sqlite (이미 했으면 skip)
"""

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver


# ══════════════════════════════════════════════════════
#  State — 제안된 행동, 사람의 결정, 실행 결과
# ══════════════════════════════════════════════════════
class HITLState(TypedDict):
    proposed_action: str    # 에이전트가 제안한 행동
    decision: str           # 사람의 결정 ("approve" / "edit" / "reject")
    final_action: str       # 최종 실행될 행동 (edit이면 수정됨)
    result: str             # 실행 결과


# ══════════════════════════════════════════════════════
#  노드 1: propose — 행동을 제안
# ══════════════════════════════════════════════════════
def propose_node(state: HITLState):
    """실전이면 LLM이 제안하겠지만, 데모라 고정 행동."""
    action = "철수에게 '내일 회의 잡자' 메일 보내기"
    print(f"\n💡 propose: 다음 행동을 제안합니다 → {action}")
    return {"proposed_action": action}


# ══════════════════════════════════════════════════════
#  노드 2: approve — 여기서 멈춰서 사람에게 묻는다 (HITL의 심장)
# ══════════════════════════════════════════════════════
def approve_node(state: HITLState):
    """interrupt()로 그래프를 멈추고 사람의 결정을 기다린다.
    재개되면 interrupt()의 반환값으로 사람의 응답이 들어온다."""

    # 이 줄에서 그래프가 멈춤. payload({"action": ...})가 호출자에게 전달됨.
    human_response = interrupt({
        "question": "이 행동을 승인하시겠습니까?",
        "proposed_action": state["proposed_action"],
        "options": ["approve", "edit", "reject"],
    })
    # ↑↑↑ 여기서 정확히 멈추고, 호출자가 Command(resume=...)로 재개하면
    #      그때 보낸 값이 human_response에 담겨서 아래로 흐름.

    print(f"   ↩️  사람 응답 받음: {human_response}")

    # human_response가 dict인지 str인지에 따라 처리
    if isinstance(human_response, dict):
        decision = human_response.get("decision", "reject")
        edited = human_response.get("edited_action", state["proposed_action"])
    else:
        decision = human_response
        edited = state["proposed_action"]

    return {
        "decision": decision,
        "final_action": edited if decision == "edit" else state["proposed_action"],
    }


# ══════════════════════════════════════════════════════
#  노드 3: execute — 결정에 따라 분기 실행
# ══════════════════════════════════════════════════════
def execute_node(state: HITLState):
    decision = state["decision"]
    if decision == "reject":
        print(f"\n🚫 execute: 거부됨 — 행동 취소")
        return {"result": "취소됨"}

    action = state["final_action"]
    label = "수정된 행동 실행" if decision == "edit" else "행동 실행"
    print(f"\n✅ execute: {label} → {action}")
    # 실전이면 여기서 진짜 메일 발송 등. 데모라 출력만.
    return {"result": f"완료: {action}"}


# ══════════════════════════════════════════════════════
#  그래프 조립
# ══════════════════════════════════════════════════════
builder = StateGraph(HITLState)
builder.add_node("propose", propose_node)
builder.add_node("approve", approve_node)
builder.add_node("execute", execute_node)
builder.add_edge(START, "propose")
builder.add_edge("propose", "approve")
builder.add_edge("approve", "execute")
builder.add_edge("execute", END)


# ══════════════════════════════════════════════════════
#  실행 — Checkpointer가 반드시 필요 (interrupt 동작 조건)
# ══════════════════════════════════════════════════════
def main():
    with SqliteSaver.from_conn_string("hitl_demo.db") as checkpointer:
        app = builder.compile(checkpointer=checkpointer)

        # 매 실행 새 스레드로 (이전 실험과 안 섞이게)
        import time
        thread_id = f"hitl_{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}

        print(f"=== HITL 데모 (thread={thread_id}) ===")

        # ── 1차 invoke: interrupt까지 실행되고 멈춤 ──
        result = app.invoke({"proposed_action": "", "decision": "",
                             "final_action": "", "result": ""}, config=config)

        # 멈췄으면 '__interrupt__' 키에 interrupt payload가 들어 있다
        if "__interrupt__" in result:
            interrupt_payload = result["__interrupt__"][0].value
            print(f"\n⏸️  그래프 멈춤. 호출자가 받은 페이로드:")
            print(f"   질문: {interrupt_payload['question']}")
            print(f"   제안: {interrupt_payload['proposed_action']}")
            print(f"   선택지: {interrupt_payload['options']}")

            # ── 사람 입력 받기 ──
            print()
            choice = input("결정 (approve / edit / reject): ").strip().lower()

            if choice == "edit":
                edited = input("수정된 행동을 입력하세요: ").strip()
                resume_value = {"decision": "edit", "edited_action": edited}
            else:
                resume_value = choice if choice in ("approve", "reject") else "reject"

            # ── 2차 invoke: Command(resume=...)로 그 지점부터 재개 ──
            print(f"\n▶️  Command(resume={resume_value})로 재개합니다...")
            result = app.invoke(Command(resume=resume_value), config=config)

        # ── 최종 결과 ──
        print(f"\n{'='*50}")
        print(f"🎯 최종 결과: {result.get('result', '(없음)')}")
        print(f"   결정: {result.get('decision', '?')}")
        print(f"   실행된 행동: {result.get('final_action', '(없음)')}")


if __name__ == "__main__":
    main()