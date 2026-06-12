"""
HITL + 캘린더 쓰기 통합 — 에이전트가 처음으로 '세상을 바꾸는' 작업

그래프 흐름:
    START → parse_request → approve(여기서 멈춤) → execute → END
            (LLM 파싱)      (사람이 확인)         (실제 캘린더 추가)

핵심:
  - parse_request: LLM이 자연어 → EventProposal(structured output) 변환
                   현재 시각을 프롬프트에 박아 '다음 주' 등을 정확히 계산
  - approve: interrupt()로 멈춤. approve/edit/reject 받음
             edit이면 사용자가 새 자연어를 주고 → parse_request로 루프백
  - execute: token_write.json으로 실제 일정 추가 (또는 거부)

전제: test_gcal_write.py가 통과해서 token_write.json이 이미 있음.
"""

import os
import datetime
from typing import TypedDict, Literal, Optional

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


# ══════════════════════════════════════════════════════
#  모델
# ══════════════════════════════════════════════════════
model = ChatOpenAI(
    base_url="http://10.1.10.111:8080/v1",
    api_key="not-needed",
    model="local-model",
    temperature=0,   # 일정 파싱은 결정적이어야 함
)


# ══════════════════════════════════════════════════════
#  Structured output: 일정 제안 구조
# ══════════════════════════════════════════════════════
class EventProposal(BaseModel):
    """사용자 자연어 요청에서 추출한 일정 제안."""
    title: str = Field(description="일정 제목 (간결하게)")
    start_iso: str = Field(description="시작 시각 ISO 8601 형식, 예: '2026-06-03T15:00:00'")
    end_iso: str = Field(description="종료 시각 ISO 8601 형식. 명시 없으면 시작+1시간")
    description: str = Field(default="", description="일정 설명 (선택)")


parser_model = model.with_structured_output(EventProposal)


# ══════════════════════════════════════════════════════
#  캘린더 서비스 (token_write.json 사용, 단독 테스트에서 검증됨)
# ══════════════════════════════════════════════════════
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
TOKEN_FILE = "token_write.json"

_calendar_service = None

def get_calendar_service():
    global _calendar_service
    if _calendar_service is not None:
        return _calendar_service
    if not os.path.exists(TOKEN_FILE):
        raise RuntimeError(f"{TOKEN_FILE}이 없습니다. test_gcal_write.py를 먼저 돌리세요.")
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            raise RuntimeError("토큰이 만료/무효. token_write.json을 지우고 재인증하세요.")
    _calendar_service = build("calendar", "v3", credentials=creds)
    return _calendar_service


# ══════════════════════════════════════════════════════
#  State
# ══════════════════════════════════════════════════════
class CalendarHITLState(TypedDict):
    user_request: str         # 사용자 자연어 요청
    proposal: dict            # EventProposal을 dict로 (직렬화 가능)
    decision: str             # approve / edit / reject
    final_proposal: dict      # 실제 추가될 최종 일정
    result: str               # 실행 결과 (성공/실패 메시지)


# ══════════════════════════════════════════════════════
#  노드 1: parse_request — LLM이 자연어 → 구조화된 일정
# ══════════════════════════════════════════════════════
def parse_request_node(state: CalendarHITLState):
    """현재 시각을 박아주고 LLM이 자연어를 EventProposal로 변환."""
    now = datetime.datetime.now().astimezone()
    now_str = now.isoformat()
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]

    prompt = (
        f"현재 시각: {now_str} ({weekday_kr}요일)\n\n"
        f"위 현재 시각을 기준으로, 다음 사용자 요청을 캘린더 일정으로 변환하라. "
        f"'다음 주', '내일' 같은 상대 시간 표현은 현재 시각 기준으로 정확히 계산하라. "
        f"종료 시각이 명시되지 않으면 시작 시각 + 1시간으로 잡아라.\n\n"
        f"사용자 요청: {state['user_request']}"
    )

    print(f"\n🧠 parse: 자연어를 일정으로 변환 중...")
    proposal: EventProposal = parser_model.invoke(prompt)
    print(f"   📋 파싱 결과:")
    print(f"      제목: {proposal.title}")
    print(f"      시작: {proposal.start_iso}")
    print(f"      종료: {proposal.end_iso}")

    return {"proposal": proposal.model_dump()}


# ══════════════════════════════════════════════════════
#  노드 2: approve — interrupt로 멈춰서 사람에게 묻기
# ══════════════════════════════════════════════════════
def approve_node(state: CalendarHITLState):
    """제안된 일정을 사람에게 보여주고 approve/edit/reject 받는다."""
    human_response = interrupt({
        "question": "이 일정을 추가하시겠습니까?",
        "proposal": state["proposal"],
        "options": ["approve", "edit", "reject"],
    })
    print(f"   ↩️  사람 응답: {human_response if isinstance(human_response, str) else human_response.get('decision', '?')}")

    if isinstance(human_response, dict):
        decision = human_response.get("decision", "reject")
        edited_text = human_response.get("edited_request")
    else:
        decision = human_response
        edited_text = None

    if decision == "edit" and edited_text:
        # 새 요청으로 user_request를 갈아끼우고 parse로 다시 보낼 거다
        return {"decision": "edit", "user_request": edited_text}

    return {"decision": decision, "final_proposal": state["proposal"]}


# ══════════════════════════════════════════════════════
#  노드 3: execute — 결정에 따라 실제 행동
# ══════════════════════════════════════════════════════
def execute_node(state: CalendarHITLState):
    decision = state["decision"]
    if decision == "reject":
        print(f"\n🚫 execute: 거부됨 — 캘린더 변경 없음")
        return {"result": "취소됨 (캘린더에 추가 안 함)"}

    proposal = state["final_proposal"]
    print(f"\n📅 execute: 실제 캘린더에 추가 중...")

    try:
        service = get_calendar_service()
        event_body = {
            "summary": proposal["title"],
            "description": proposal.get("description", ""),
            "start": {"dateTime": proposal["start_iso"], "timeZone": "Asia/Seoul"},
            "end":   {"dateTime": proposal["end_iso"],   "timeZone": "Asia/Seoul"},
        }
        created = service.events().insert(calendarId="primary", body=event_body).execute()
        link = created.get("htmlLink", "(링크 없음)")
        print(f"   ✅ 추가 완료! 링크: {link}")
        return {"result": f"추가 성공: {proposal['title']} ({link})"}
    except Exception as e:
        print(f"   ❌ 실패: {e}")
        return {"result": f"실패: {e}"}


# ══════════════════════════════════════════════════════
#  라우팅: approve가 'edit'이면 parse로 루프백, 아니면 execute로
# ══════════════════════════════════════════════════════
def route_after_approve(state: CalendarHITLState):
    if state["decision"] == "edit":
        return "parse_request"   # 재파싱 → 또 approve
    return "execute"


# ══════════════════════════════════════════════════════
#  그래프
# ══════════════════════════════════════════════════════
builder = StateGraph(CalendarHITLState)
builder.add_node("parse_request", parse_request_node)
builder.add_node("approve", approve_node)
builder.add_node("execute", execute_node)
builder.add_edge(START, "parse_request")
builder.add_edge("parse_request", "approve")
builder.add_conditional_edges("approve", route_after_approve,
                              {"parse_request": "parse_request", "execute": "execute"})
builder.add_edge("execute", END)


# ══════════════════════════════════════════════════════
#  실행 루프 (interrupt 처리)
# ══════════════════════════════════════════════════════
def main():
    user_request = input("📝 추가하고 싶은 일정을 자연어로 입력하세요\n   (예: '다음 주 화요일 오후 3시에 치과 예약'): ").strip()
    if not user_request:
        print("입력이 비었습니다.")
        return

    import time
    thread_id = f"cal_hitl_{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}

    with SqliteSaver.from_conn_string("calendar_hitl.db") as checkpointer:
        app = builder.compile(checkpointer=checkpointer)

        initial = {"user_request": user_request, "proposal": {}, "decision": "",
                   "final_proposal": {}, "result": ""}
        result = app.invoke(initial, config=config)

        # interrupt 처리 루프: edit이면 다시 interrupt가 뜰 수 있음
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"\n⏸️  사람 확인 필요")
            print(f"   질문: {payload['question']}")
            p = payload["proposal"]
            print(f"   제안된 일정:")
            print(f"     • 제목: {p['title']}")
            print(f"     • 시작: {p['start_iso']}")
            print(f"     • 종료: {p['end_iso']}")
            if p.get("description"):
                print(f"     • 설명: {p['description']}")

            choice = input(f"\n   결정 (approve / edit / reject): ").strip().lower()

            if choice == "edit":
                new_text = input("   수정된 요청을 자연어로 다시 입력: ").strip()
                resume_value = {"decision": "edit", "edited_request": new_text}
            elif choice in ("approve", "reject"):
                resume_value = choice
            else:
                print(f"   알 수 없는 입력 '{choice}', reject로 처리합니다.")
                resume_value = "reject"

            print(f"\n▶️  재개...")
            result = app.invoke(Command(resume=resume_value), config=config)

        print(f"\n{'='*55}")
        print(f"🎯 최종: {result.get('result', '(결과 없음)')}")


if __name__ == "__main__":
    main()