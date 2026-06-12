"""
캘린더 쓰기 인증 단독 테스트 (token_write.json 분리 발급)

목적: HITL 통합으로 가기 전에 두 가지를 단독 검증한다.
  1) calendar.events 스코프로 새 OAuth 토큰을 받을 수 있는가
  2) 그 토큰으로 실제 캘린더에 일정을 추가할 수 있는가

기존 token.json(readonly)은 건드리지 않는다.
처음 실행 시 브라우저가 뜨고 새 동의 화면이 나옴 (쓰기 권한 요청).
"""

import os
import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# ── 설정 (메인 에이전트의 readonly 토큰과 완전히 분리) ──
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]   # 일정 CRUD만
CREDS_FILE = "credentials.json"      # 기존 것 그대로 사용
TOKEN_FILE = "token_write.json"      # ← 새 파일! token.json과 별개


def get_calendar_service_write():
    """쓰기 권한 캘린더 서비스. 첫 호출 시 브라우저 인증."""
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("🔐 새 권한(쓰기)으로 인증을 시작합니다. 브라우저가 열립니다...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
        print(f"✅ 새 토큰 저장: {TOKEN_FILE}")

    return build("calendar", "v3", credentials=creds)


def add_test_event():
    """오늘로부터 7일 뒤 오전 10시에 1시간짜리 테스트 일정을 추가."""
    service = get_calendar_service_write()

    # 시작 시각: 7일 후 오전 10시 (UTC)
    start_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=7)
    start_dt = start_dt.replace(hour=10, minute=0, second=0, microsecond=0)
    end_dt = start_dt + datetime.timedelta(hours=1)

    event_body = {
        "summary": "[테스트] HITL 캘린더 쓰기 검증",
        "description": "test_gcal_write.py가 만든 테스트 일정. 안전하게 삭제 가능.",
        "start": {"dateTime": start_dt.isoformat(), "timeZone": "UTC"},
        "end":   {"dateTime": end_dt.isoformat(),   "timeZone": "UTC"},
    }

    print(f"\n📅 일정 추가 시도:")
    print(f"   제목: {event_body['summary']}")
    print(f"   시작: {start_dt.isoformat()}")
    print(f"   종료: {end_dt.isoformat()}")

    created = service.events().insert(calendarId="primary", body=event_body).execute()

    print(f"\n✅ 일정 추가 성공!")
    print(f"   ID: {created.get('id')}")
    print(f"   링크: {created.get('htmlLink')}")
    print(f"\n👉 본인 구글 캘린더에서 7일 뒤 오전 10시에 '[테스트] HITL ...' 일정이 보이면 성공입니다.")
    print(f"   확인 후 삭제하셔도 됩니다 (이 코드 다시 안 돌리면 또 안 생김).")


if __name__ == "__main__":
    add_test_event()