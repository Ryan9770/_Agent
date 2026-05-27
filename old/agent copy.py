"""
로컬 llama-server 에이전트 학습용 골격
1단계: bare client (chat 함수)
2단계: ReAct 에이전트 루프 (run_agent 함수)

실행 전: pip install openai
"""

from openai.types.beta.chatkit import chatkit_thread_user_message_item
from openai import OpenAI
import json

# ── 설정 ──────────────────────────────────────────────
# llama-server가 OpenAI 호환 모드로 떠 있다고 가정 (--api-key 없이)
client = OpenAI(
    base_url="http://10.1.10.111:8080/v1",  # 본인 환경에 맞게 수정
    api_key="not-needed",                  # 로컬이면 아무 값
)
MODEL = "local-model"  # llama-server는 보통 모델명 무시하지만 명시


# ── 1단계: bare client ────────────────────────────────
def chat(messages, tools=None):
    """llama-server를 한 번 호출하는 가장 단순한 래퍼."""
    kwargs = {"model": MODEL, "messages": messages, "temperature": 0.3}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


# ── 도구 정의 ─────────────────────────────────────────
# (a) 실제 파이썬 함수
def calculator(expression: str) -> str:
    """안전한 산술 계산만 허용."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

def get_time(timezone: str = "UTC") -> str:
    from datetime import datetime, timezone as tz
    return datetime.now(tz.utc).isoformat()

# (b) 모델에게 알려줄 도구 스키마 (OpenAI tool calling 포맷)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '12 * (3 + 4)'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current UTC time.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        },
    },
]

# (c) 이름 → 함수 매핑 (모델이 부른 이름으로 실제 함수를 찾음)
TOOL_FUNCS = {"calculator": calculator, "get_time": get_time}


# ══════════════════════════════════════════════════════
#  run_agent: 가드레일 강화 버전
#  - (A) 중복 도구 호출 차단
#  - (B) step 한계 도달 시 강제 마무리
#  - (C) 도구 실패 시 재시도 (네트워크 도구용)
#  - (D) 호출 로그 정리
# ══════════════════════════════════════════════════════
import time

SYSTEM_PROMPT = (
    "You are a helpful agent with access to several tools. "
    "Choose the right tool for each question:\n"
    "- search_docs: for the USER'S OWN documents, notes, internal/team commands.\n"
    "- web_search: for CURRENT or EXTERNAL info — news, recent events, public facts.\n"
    "- calculator / get_time: for arithmetic and current time.\n\n"
    "Rules:\n"
    "1. Do NOT repeat the same search you already made. If a tool result is "
    "   insufficient, either try a MEANINGFULLY different query or answer with "
    "   what you have.\n"
    "2. Once you have enough information, answer immediately. Don't over-call tools.\n"
    "3. Base answers on tool results and cite the source (document name or URL). "
    "   If you lack enough info, say so honestly."
)

# 턴 단위로 살아야 하는 상태. run_agent를 부르기 직전에 초기화한다.
_seen_calls_this_turn = set()

def chat_turn(messages, user_input, verbose=True):
    """한 턴을 처리: user 메시지 추가 → 턴 상태 초기화 → run_agent 실행."""
    global _seen_calls_this_turn
    _seen_calls_this_turn = set()          # 이번 턴의 중복 추적만 리셋
    messages.append({"role": "user", "content": user_input})
    answer, messages = run_agent(messages, verbose=verbose)
    messages = compress_memory(messages, verbose=verbose)   # ← 턴 끝에 압축
    return answer, messages

# ══════════════════════════════════════════════════════
#  run_agent: 대화 메모리 버전
#  messages를 인자로 받고, 갱신된 messages를 반환한다.
#  → 호출하는 쪽이 messages를 보관하면 멀티턴이 된다.
# ══════════════════════════════════════════════════════
def run_agent(messages, max_steps=6, verbose=True):
    """
    messages: 지금까지의 대화 전체 (system 포함). 이번 user 메시지까지
              이미 들어 있는 상태로 받는다.
    반환: (최종답변 문자열, 갱신된 messages)
    """
    for step in range(max_steps):
        is_last = (step == max_steps - 1)
        if is_last:
            messages.append({
                "role": "system",
                "content": "You have reached the step limit. "
                           "Answer NOW with the information you have. Do not call tools.",
            })
            resp = chat(messages)
        else:
            resp = chat(messages, tools=TOOLS)

        msg = resp.choices[0].message

        # 도구 호출 없음 → 최종 답변. assistant 답변도 messages에 기록하고 반환.
        if not msg.tool_calls:
            if verbose:
                print(f"[step {step}] ✅ 최종 답변")
            messages.append({"role": "assistant", "content": msg.content})
            return msg.content, messages

        # 도구 호출 기록
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        # ⚠️ seen_calls를 step 루프 바깥에 두면 턴을 넘어 누적되어
        #    다음 턴의 정상 검색까지 막는다. 그래서 이번 턴 동안만 유지하도록
        #    매 run_agent 호출의 첫 진입에서 초기화한다 (아래 설명 참고).
        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments
            args = json.loads(raw_args)

            call_key = (name, raw_args)
            if call_key in _seen_calls_this_turn:
                if verbose:
                    print(f"[step {step}] ⛔ 중복 차단: {name}({args})")
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": "You already made this exact call. Do not repeat it. "
                               "Use the previous result, try a different approach, or answer now.",
                })
                continue
            _seen_calls_this_turn.add(call_key)

            if verbose:
                print(f"[step {step}] 🔧 {name}({args})")

            func = TOOL_FUNCS.get(name)
            result = (f"Error: unknown tool '{name}'" if func is None
                      else _run_tool_with_retry(func, args, name, verbose))

            if verbose:
                preview = str(result)[:120].replace("\n", " ")
                print(f"           → {preview}{'...' if len(str(result)) > 120 else ''}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    messages.append({"role": "assistant", "content": "최대 스텝 도달 — 답을 못 냈습니다."})
    return "최대 스텝 도달 — 답을 못 냈습니다.", messages


# ── (C) 재시도 헬퍼 ───────────────────────────────────
def _run_tool_with_retry(func, args, name, verbose, retries=2):
    """
    도구 실행. 결과가 'Error:'로 시작하면(우리 도구들의 실패 규약)
    네트워크성 일시 오류로 보고 잠깐 쉬었다 재시도한다.
    """
    for attempt in range(retries + 1):
        try:
            result = func(**args)
        except Exception as e:
            result = f"Error: tool '{name}' raised {type(e).__name__}: {e}"

        # 성공으로 보이면 즉시 반환
        if not str(result).startswith("Error:"):
            return result

        # 실패인데 재시도 여지가 있으면 잠깐 쉬고 다시
        if attempt < retries:
            if verbose:
                print(f"           ↻ 재시도 {attempt + 1}/{retries} ({name})")
            time.sleep(1.0 * (attempt + 1))   # 1초, 2초 점증 백오프

    return result   # 끝까지 실패하면 마지막 에러 메시지 반환


# ── 진단: tool calling 지원 여부 확인 ─────────────────
def diagnose():
    print("=" * 50)
    print("진단 1: 기본 chat 동작 확인")
    print("=" * 50)
    r = chat([{"role": "user", "content": "Say 'pong' and nothing else."}])
    print("응답:", r.choices[0].message.content)

    print("\n" + "=" * 50)
    print("진단 2: 네이티브 tool calling 지원 확인")
    print("=" * 50)
    r = chat(
        [{"role": "user", "content": "What is 47 * 89? Use the calculator tool."}],
        tools=TOOLS,
    )
    m = r.choices[0].message
    if m.tool_calls:
        print("✅ 네이티브 tool calling 지원됨")
        for tc in m.tool_calls:
            print(f"   모델이 호출: {tc.function.name}({tc.function.arguments})")
    else:
        print("❌ tool_calls 없음 — 네이티브 미지원일 수 있음")
        print("   모델 응답:", m.content)
        print("   → 프롬프트 기반 JSON 강제 방식으로 우회 필요")

# ══════════════════════════════════════════════════════
#  4-A단계: 키워드 기반 로컬 문서 검색 (RAG의 첫 버전)
#  기존 agent.py에 이어 붙이세요. run_agent는 수정 불필요.
# ══════════════════════════════════════════════════════
import os
import glob

# ── 문서 적재 & 청킹 ──────────────────────────────────
DOCS_DIR = "./docs"   # 여기에 .txt / .md 파일을 넣어두세요

def load_and_chunk(docs_dir=DOCS_DIR, chunk_size=500, overlap=50):
    """
    폴더 안의 텍스트 파일들을 읽어서 일정 크기 조각(chunk)으로 자른다.
    overlap: 조각 경계에서 문맥이 끊기지 않도록 겹치는 글자 수.
    반환: [{"source": 파일명, "text": 조각내용}, ...]
    """
    chunks = []
    paths = glob.glob(os.path.join(docs_dir, "*.txt")) + \
            glob.glob(os.path.join(docs_dir, "*.md"))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # 글자 단위로 잘라낸다 (가장 단순한 방식)
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size]
            chunks.append({"source": os.path.basename(path), "text": chunk})
            start += chunk_size - overlap
    return chunks

# 프로그램 시작 시 한 번 적재해서 메모리에 들고 있는다
DOC_CHUNKS = load_and_chunk()
print(f"[RAG] {len(DOC_CHUNKS)}개 조각 적재 완료 (from {DOCS_DIR})")

# ══════════════════════════════════════════════════════
#  4-B단계: 임베딩 기반 로컬 문서 검색
#  load_and_chunk() 함수는 그대로 두고,
#  그 아래 적재부 + search_docs만 교체
# ══════════════════════════════════════════════════════
from sentence_transformers import SentenceTransformer
import numpy as np

# ── 임베딩 모델 로드 (시작 시 한 번) ──────────────────
# 한국어+영어 섞인 기술 문서용 다국어 모델. GPU로 로드.
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
print(f"[RAG] 임베딩 모델 로딩 중: {EMBED_MODEL_NAME} ...")
embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
print(f"[RAG] 임베딩 모델 로드 완료 (device: {embedder.device})")

# ── 문서 적재 + 임베딩 사전 계산 ──────────────────────
DOC_CHUNKS = load_and_chunk()   # 4-A의 청킹 함수 재사용
print(f"[RAG] {len(DOC_CHUNKS)}개 조각 적재 완료")

# 모든 조각을 미리 벡터로 변환해 메모리에 들고 있는다.
# normalize_embeddings=True → 코사인 유사도를 내적(dot)으로 간단 계산 가능
_texts = [c["text"] for c in DOC_CHUNKS]
DOC_EMBEDDINGS = embedder.encode(
    _texts, normalize_embeddings=True, show_progress_bar=False
)
print(f"[RAG] 임베딩 사전 계산 완료: shape={DOC_EMBEDDINGS.shape}")


# ── 검색 도구 (임베딩 버전) ───────────────────────────
def search_docs(query: str, top_k: int = 3) -> str:
    """
    질문을 같은 임베딩 공간으로 변환해, 의미적으로 가장 가까운
    조각을 코사인 유사도로 찾는다. (단어 겹침이 아니라 '의미'로 매칭)
    """
    if not DOC_CHUNKS:
        return "No documents loaded. Put .txt/.md files in ./docs"

    # 1) 질문을 벡터로 (정규화)
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]

    # 2) 모든 조각 벡터와 내적 = 코사인 유사도 (정규화했으므로)
    scores = DOC_EMBEDDINGS @ q_vec   # shape: (조각수,)

    # 3) 상위 top_k 인덱스
    top_idx = np.argsort(scores)[::-1][:top_k]

    out = []
    for i in top_idx:
        ch = DOC_CHUNKS[i]
        out.append(f"[출처: {ch['source']} | 유사도: {scores[i]:.3f}]\n{ch['text']}")
    return "\n\n---\n\n".join(out)


# ── 도구 등록 (이 세 줄이 도구를 에이전트에 꽂는 전부) ──
TOOLS.append({
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the user's local document collection for "
                       "relevant passages. Use this when the user asks about "
                       "information that may be in their documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search keywords"},
                "top_k": {"type": "integer", "description": "how many passages, default 3"},
            },
            "required": ["query"],
        },
    },
})
TOOL_FUNCS["search_docs"] = search_docs

# ══════════════════════════════════════════════════════
#  웹 검색 도구 (로컬 SearXNG 인스턴스 사용)
# ══════════════════════════════════════════════════════
import requests

SEARXNG_URL = "http://localhost:8080/search"  # 본인 SearXNG 주소

def web_search(query: str, num_results: int = 5) -> str:
    """
    로컬 SearXNG로 웹을 검색해 상위 결과의 제목·요약·URL을 반환.
    외부 네트워크를 타므로 실패할 수 있어 try/except로 감싼다.
    """
    try:
        resp = requests.get(
            SEARXNG_URL,
            params={"q": query, "format": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        return "Error: search timed out. Try a simpler query."
    except requests.exceptions.RequestException as e:
        return f"Error: search failed ({e})"
    except ValueError:
        return "Error: could not parse search response as JSON."

    results = data.get("results", [])[:num_results]
    if not results:
        return f"No web results found for: {query}"

    out = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "(no title)")
        content = r.get("content", "").strip()
        url = r.get("url", "")
        out.append(f"{i}. {title}\n   {content}\n   ({url})")
    return "\n\n".join(out)


# ── 도구 등록 ─────────────────────────────────────────
TOOLS.append({
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the public web for current, external, or "
                       "real-time information (news, recent events, facts "
                       "not in the user's local documents). Returns titles, "
                       "snippets, and URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "num_results": {"type": "integer", "description": "default 5"},
            },
            "required": ["query"],
        },
    },
})
TOOL_FUNCS["web_search"] = web_search

# ══════════════════════════════════════════════════════
#  하이브리드 메모리 압축
#  구조: [system] + [요약(선택)] + [최근 KEEP_RECENT_TURNS 원본]
# ══════════════════════════════════════════════════════
COMPRESS_THRESHOLD = 16   # 메시지가 이 개수를 넘으면 압축 시도
KEEP_RECENT_MSGS   = 8    # 최근 이 개수의 메시지는 항상 원본 보존

def compress_memory(messages, verbose=True):
    """
    messages가 너무 길면:
      1) 압축 대상(오래된 구간)의 긴 tool 결과를 먼저 비운다.
      2) 그래도 길면 그 구간을 LLM으로 요약해 한 덩어리로 대체한다.
    system 메시지와 최근 KEEP_RECENT_MSGS개는 건드리지 않는다.
    """
    if len(messages) <= COMPRESS_THRESHOLD:
        return messages   # 아직 압축할 필요 없음

    system_msg = messages[0]              # 항상 보존
    recent = messages[-KEEP_RECENT_MSGS:] # 최근 원본 보존
    middle = messages[1:-KEEP_RECENT_MSGS] # 압축 대상 (중간 구간)

    if not middle:
        return messages

    # ── 안전장치: 잘린 구간이 tool 메시지로 시작하면 안 된다 ──
    # (assistant의 tool_calls와 tool 결과가 분리되면 API가 에러를 냄)
    while recent and recent[0].get("role") == "tool":
        # tool 결과가 고아가 되지 않게, 한 칸씩 middle로 되돌린다
        middle.append(recent.pop(0))
    if not recent:
        return messages  # 비정상 상황이면 그냥 둔다

    if verbose:
        print(f"   🗜️  메모리 압축 시작: 중간 {len(middle)}개 메시지 → 요약")

    # ── 1단계: 중간 구간의 긴 tool 결과부터 비운다 ──
    for m in middle:
        if m.get("role") == "tool" and len(str(m.get("content", ""))) > 80:
            m["content"] = "[이전 도구 결과 생략됨]"

    # ── 2단계: 중간 구간을 텍스트로 펼쳐 LLM에 요약 요청 ──
    transcript = []
    for m in middle:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            transcript.append(f"User: {content}")
        elif role == "assistant":
            if m.get("tool_calls"):
                names = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                transcript.append(f"Assistant: (도구 호출: {names}) {content}")
            else:
                transcript.append(f"Assistant: {content}")
        elif role == "tool":
            transcript.append(f"ToolResult: {content}")
    transcript_text = "\n".join(transcript)

    summary_resp = chat([
        {"role": "system", "content":
            "Summarize the following conversation excerpt concisely in Korean. "
            "Preserve key facts, decisions, commands, and any answers the user "
            "received, so the conversation can continue coherently. "
            "Be compact — a few sentences."},
        {"role": "user", "content": transcript_text},
    ])
    summary_text = summary_resp.choices[0].message.content

    # ── 재조립: system + 요약 + 최근 원본 ──
    summary_msg = {
        "role": "system",
        "content": f"[이전 대화 요약]\n{summary_text}",
    }
    new_messages = [system_msg, summary_msg] + recent

    if verbose:
        print(f"   🗜️  압축 완료: {len(messages)}개 → {len(new_messages)}개")
    return new_messages


# ══════════════════════════════════════════════════════
#  구글 캘린더 도구 (읽기 전용)
#  test_gcal.py에서 검증한 get_calendar_service를 그대로 가져온다.
# ══════════════════════════════════════════════════════
import os
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

GCAL_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
_calendar_service = None   # 한 번 만들면 재사용 (매 호출마다 인증 안 하도록)

def get_calendar_service():
    global _calendar_service
    if _calendar_service is not None:
        return _calendar_service

    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", GCAL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", GCAL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())

    _calendar_service = build("calendar", "v3", credentials=creds)
    return _calendar_service


# ── 도구 함수 ─────────────────────────────────────────
def list_calendar_events(days_ahead: int = 7, max_results: int = 10) -> str:
    """
    지금부터 days_ahead일 이내의 일정을 시간순으로 조회한다.
    네트워크/인증 실패에 대비해 try/except로 감싼다 (web_search와 동일 규약).
    """
    try:
        service = get_calendar_service()
        now = datetime.datetime.now(datetime.timezone.utc)
        time_min = now.isoformat()
        time_max = (now + datetime.timedelta(days=days_ahead)).isoformat()

        result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
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
        summary = e.get("summary", "(제목 없음)")
        lines.append(f"- {start} | {summary}")
    return "\n".join(lines)


# ── 도구 등록 ─────────────────────────────────────────
TOOLS.append({
    "type": "function",
    "function": {
        "name": "list_calendar_events",
        "description": "List the user's upcoming Google Calendar events within "
                       "a given number of days from now. Use when the user asks "
                       "about their schedule, appointments, or what's coming up.",
        "parameters": {
            "type": "object",
            "properties": {
                "days_ahead": {"type": "integer", "description": "how many days ahead to look, default 7"},
                "max_results": {"type": "integer", "description": "max events to return, default 10"},
            },
        },
    },
})
TOOL_FUNCS["list_calendar_events"] = list_calendar_events


# ══════════════════════════════════════════════════════
#  Plan-and-Execute 에이전트
#  1) 계획 단계: 도구 없이 단계별 계획을 세운다
#  2) 실행 단계: 계획을 컨텍스트에 넣고 기존 run_agent로 실행
# ══════════════════════════════════════════════════════
def make_plan(user_message, verbose=True):
    """도구를 주지 않고, 요청을 푸는 단계별 계획만 생성한다."""
    tool_names = ", ".join(TOOL_FUNCS.keys())
    planning_messages = [
        {"role": "system", "content":
            "You are a planning module. Given a user request, produce a SHORT "
            "numbered plan (2-5 steps) describing how to fulfill it. "
            f"Available tools you can plan around: {tool_names}. "
            "For each step, note which tool (if any) would be used. "
            "Output ONLY the plan, no preamble, no answer yet. "
            "If the request is trivial (one step, no tools), say 'DIRECT' and nothing else."},
        {"role": "user", "content": user_message},
    ]
    resp = chat(planning_messages)   # 도구 없이 호출 → 계획 텍스트만 나옴
    plan = resp.choices[0].message.content.strip()
    if verbose:
        print(f"\n📋 계획:\n{plan}\n")
    return plan


def run_planning_agent(messages, user_input, verbose=True):
    """계획을 세운 뒤, 그 계획을 길잡이로 기존 run_agent를 돌린다."""
    global _seen_calls_this_turn
    _seen_calls_this_turn = set()

    # 1) 계획 단계
    plan = make_plan(user_input, verbose=verbose)

    # 2) 사용자 메시지 추가
    messages.append({"role": "user", "content": user_input})

    # 'DIRECT'면 계획 주입 없이 바로 실행 (단순 질문은 오버헤드 방지)
    if plan != "DIRECT":
        # 계획을 system 힌트로 끼워넣어 실행을 안내한다
        messages.append({
            "role": "system",
            "content": f"Follow this plan to answer the user "
                       f"(adapt if needed):\n{plan}",
        })

    # 3) 실행 단계 — 기존 루프 그대로
    answer, messages = run_agent(messages, verbose=verbose)

    # 주입했던 계획 힌트는 대화 기록에서 빼둔다 (메모리 오염 방지)
    # → 계획은 일회용이므로 다음 턴까지 들고 갈 필요 없음
    messages = [m for m in messages
                if not (m.get("role") == "system"
                        and str(m.get("content", "")).startswith("Follow this plan"))]

    messages = compress_memory(messages, verbose=verbose)
    return answer, messages

if __name__ == "__main__":
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("\n=== 계획 수립형 에이전트 (종료: 'quit') ===")
    while True:
        user_input = input("\n🙂 You: ").strip()
        if user_input.lower() in ("quit", "exit", "q", ""):
            break
        answer, conversation = run_planning_agent(conversation, user_input)
        print(f"\n🤖 {answer}")
        print(f"   [메모리: {len(conversation)}개 메시지]")