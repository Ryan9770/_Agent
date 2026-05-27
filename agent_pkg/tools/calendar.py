"""구글 캘린더 도구: list_calendar_events (읽기 전용)."""

import os
import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .. import config
from .registry import tool

_calendar_service = None   # 첫 호출 시 lazy 생성 후 재사용


def get_calendar_service():
    global _calendar_service
    if _calendar_service is not None:
        return _calendar_service

    creds = None
    if os.path.exists(config.GCAL_TOKEN):
        creds = Credentials.from_authorized_user_file(config.GCAL_TOKEN, config.GCAL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(config.GCAL_CREDS, config.GCAL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(config.GCAL_TOKEN, "w") as f:
            f.write(creds.to_json())

    _calendar_service = build("calendar", "v3", credentials=creds)
    return _calendar_service


@tool(
    name="list_calendar_events",
    description="List the user's upcoming Google Calendar events within a given "
                "number of days from now. Use when the user asks about their "
                "schedule, appointments, or what's coming up.",
    parameters={
        "type": "object",
        "properties": {
            "days_ahead": {"type": "integer", "description": "how many days ahead to look, default 7"},
            "max_results": {"type": "integer", "description": "max events to return, default 10"},
        },
    },
)
def list_calendar_events(days_ahead: int = 7, max_results: int = 10) -> str:
    """지금부터 days_ahead일 이내의 일정을 시간순으로 조회한다."""
    try:
        service = get_calendar_service()
        now = datetime.datetime.now(datetime.timezone.utc)
        result = service.events().list(
            calendarId="primary",
            timeMin=now.isoformat(),
            timeMax=(now + datetime.timedelta(days=days_ahead)).isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
    except Exception as e:
        return f"Error: calendar fetch failed ({type(e).__name__}: {e})"

    if not events:
        return f"No events in the next {days_ahead} days."
    lines = []
    for e in events:
        start = e["start"].get("dateTime", e["start"].get("date"))
        lines.append(f"- {start} | {e.get('summary', '(제목 없음)')}")
    return "\n".join(lines)


@tool(
    name="search_calendar_events",
    description="Search the user's Google Calendar for events matching a "
                "keyword (e.g. a birthday, anniversary, a person's name, or a "
                "specific meeting title). Use this for the user's PERSONAL dates "
                "and recurring events like birthdays — NOT just upcoming items. "
                "Searches a wide time window by default.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "keyword to search event titles, e.g. '생일'"},
            "days_window": {"type": "integer", "description": "how many days around now to search, default 365"},
            "max_results": {"type": "integer", "description": "max events, default 10"},
        },
        "required": ["query"],
    },
)
def search_calendar_events(query: str, days_window: int = 365, max_results: int = 10) -> str:
    """캘린더에서 키워드(q)로 일정을 검색한다. 생일·기념일 등 반복/개인 일정용."""
    try:
        service = get_calendar_service()
        now = datetime.datetime.now(datetime.timezone.utc)
        result = service.events().list(
            calendarId="primary",
            q=query,                                   # ← 텍스트 검색 파라미터
            timeMin=(now - datetime.timedelta(days=days_window)).isoformat(),
            timeMax=(now + datetime.timedelta(days=days_window)).isoformat(),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = result.get("items", [])
    except Exception as e:
        return f"Error: calendar search failed ({type(e).__name__}: {e})"

    if not events:
        return f"No calendar events matching '{query}'."
    lines = []
    for e in events:
        start = e["start"].get("dateTime", e["start"].get("date"))
        lines.append(f"- {start} | {e.get('summary', '(제목 없음)')}")
    return "\n".join(lines)