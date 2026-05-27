"""
로컬 llama-server 기반 단일 AI 에이전트 (학습용)

구성 (위에서 아래로):
  1. 설정       — 엔드포인트, 모델명, 각종 임계값을 한 곳에 모음
  2. LLM 호출    — chat(): llama-server를 호출하는 래퍼
  3. 도구        — calculator / get_time / search_docs / web_search / list_calendar_events
  4. 에이전트 루프 — run_agent(): ReAct 루프 + 가드레일(중복차단/step한계/재시도)
  5. 메모리      — compress_memory(): 하이브리드 압축(요약 + 도구결과 비우기)
  6. 계획        — make_plan() / run_planning_agent(): Plan-and-Execute
  7. 진입점      — 멀티턴 대화 루프
"""

# ──────────────────────────────────────────────────────
#  표준 라이브러리
# ──────────────────────────────────────────────────────
import os
import glob
import json
import time
import datetime

# ──────────────────────────────────────────────────────
#  서드파티
# ──────────────────────────────────────────────────────
import requests
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# ══════════════════════════════════════════════════════
#  1. 설정 — 바뀔 수 있는 값은 전부 여기 모음
# ══════════════════════════════════════════════════════
# LLM (원격 DGX Spark의 llama-server, OpenAI 호환 모드)
LLM_BASE_URL = "http://10.1.10.111:8080/v1"
LLM_API_KEY  = "not-needed"          # 로컬/사설이면 아무 값
MODEL        = "local-model"         # llama-server는 보통 모델명 무시
TEMPERATURE  = 0.3

# RAG (로컬 문서 검색)
DOCS_DIR         = "./docs"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DEVICE     = "cuda"

# 웹 검색 (로컬 SearXNG 인스턴스 — llama-server와 호스트가 다름에 주의)
SEARXNG_URL = "http://localhost:8080/search"

# 구글 캘린더
GCAL_SCOPES   = ["https://www.googleapis.com/auth/calendar.readonly"]
GCAL_CREDS    = "credentials.json"
GCAL_TOKEN    = "token.json"

# 에이전트 루프 / 메모리
MAX_STEPS          = 6
COMPRESS_THRESHOLD = 16   # 메시지가 이 개수를 넘으면 압축 시도
KEEP_RECENT_MSGS   = 8    # 최근 이 개수의 메시지는 항상 원본 보존

SYSTEM_PROMPT = (
    "You are a helpful agent with access to several tools. "
    "Choose the right tool for each question:\n"
    "- search_docs: for the USER'S OWN documents, notes, internal/team commands.\n"
    "- web_search: for CURRENT or EXTERNAL info — news, recent events, public facts.\n"
    "- list_calendar_events: for the user's schedule, appointments, upcoming events.\n"
    "- calculator / get_time: for arithmetic and current time.\n\n"
    "Rules:\n"
    "1. Do NOT repeat the same search you already made. If a tool result is "
    "   insufficient, either try a MEANINGFULLY different query or answer with "
    "   what you have.\n"
    "2. Once you have enough information, answer immediately. Don't over-call tools.\n"
    "3. Base answers on tool results and cite the source (document name or URL). "
    "   If you lack enough info, say so honestly."
)


# ══════════════════════════════════════════════════════
#  2. LLM 호출
# ══════════════════════════════════════════════════════
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

def chat(messages, tools=None):
    """llama-server를 한 번 호출하는 래퍼."""
    kwargs = {"model": MODEL, "messages": messages, "temperature": TEMPERATURE}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs)


# ══════════════════════════════════════════════════════
#  3. 도구
#  TOOLS(스키마 목록)와 TOOL_FUNCS(이름→함수)를 아래에서 누적해 채운다.
# ══════════════════════════════════════════════════════
TOOLS = []
TOOL_FUNCS = {}


# ── 3-1. 기본 도구: calculator / get_time ─────────────
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
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

TOOLS += [
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
TOOL_FUNCS["calculator"] = calculator
TOOL_FUNCS["get_time"]   = get_time


# ── 3-2. RAG 도구: search_docs (임베딩 기반) ──────────
def load_and_chunk(docs_dir=DOCS_DIR, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """폴더 안 텍스트 파일을 일정 글자 수 조각으로 자른다(overlap만큼 겹침)."""
    chunks = []
    paths = glob.glob(os.path.join(docs_dir, "*.txt")) + \
            glob.glob(os.path.join(docs_dir, "*.md"))
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size]
            chunks.append({"source": os.path.basename(path), "text": chunk})
            start += chunk_size - overlap
    return chunks

# 시작 시 한 번: 모델 로드 + 문서 적재 + 임베딩 사전 계산
print(f"[RAG] 임베딩 모델 로딩 중: {EMBED_MODEL_NAME} ...")
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
print(f"[RAG] 임베딩 모델 로드 완료 (device: {embedder.device})")

DOC_CHUNKS = load_and_chunk()
print(f"[RAG] {len(DOC_CHUNKS)}개 조각 적재 완료 (from {DOCS_DIR})")

_texts = [c["text"] for c in DOC_CHUNKS]
DOC_EMBEDDINGS = (
    embedder.encode(_texts, normalize_embeddings=True, show_progress_bar=False)
    if _texts else np.zeros((0, 384))
)
print(f"[RAG] 임베딩 사전 계산 완료: shape={DOC_EMBEDDINGS.shape}")


def search_docs(query: str, top_k: int = 3) -> str:
    """질문을 임베딩해 의미적으로 가장 가까운 조각을 코사인 유사도로 찾는다."""
    if not DOC_CHUNKS:
        return "No documents loaded. Put .txt/.md files in ./docs"
    q_vec = embedder.encode([query], normalize_embeddings=True)[0]
    scores = DOC_EMBEDDINGS @ q_vec
    top_idx = np.argsort(scores)[::-1][:top_k]
    out = []
    for i in top_idx:
        ch = DOC_CHUNKS[i]
        out.append(f"[출처: {ch['source']} | 유사도: {scores[i]:.3f}]\n{ch['text']}")
    return "\n\n---\n\n".join(out)

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


# ── 3-3. 웹 검색 도구: web_search (SearXNG) ───────────
def web_search(query: str, num_results: int = 5) -> str:
    """로컬 SearXNG로 웹을 검색해 상위 결과의 제목·요약·URL을 반환."""
    try:
        resp = requests.get(
            SEARXNG_URL, params={"q": query, "format": "json"}, timeout=15
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


# ── 3-4. 구글 캘린더 도구: list_calendar_events ───────
_calendar_service = None   # 첫 호출 시 lazy 생성 후 재사용

def get_calendar_service():
    global _calendar_service
    if _calendar_service is not None:
        return _calendar_service

    creds = None
    if os.path.exists(GCAL_TOKEN):
        creds = Credentials.from_authorized_user_file(GCAL_TOKEN, GCAL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(GCAL_CREDS, GCAL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(GCAL_TOKEN, "w") as f:
            f.write(creds.to_json())

    _calendar_service = build("calendar", "v3", credentials=creds)
    return _calendar_service


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
#  4. 에이전트 루프 (ReAct + 가드레일)
# ══════════════════════════════════════════════════════
# 턴 단위로만 살아야 하는 상태(중복 추적). run_planning_agent 진입 시 초기화.
_seen_calls_this_turn = set()

def _run_tool_with_retry(func, args, name, verbose, retries=2):
    """도구 실행. 결과가 'Error:'로 시작하면 일시 오류로 보고 점증 백오프 재시도."""
    for attempt in range(retries + 1):
        try:
            result = func(**args)
        except Exception as e:
            result = f"Error: tool '{name}' raised {type(e).__name__}: {e}"
        if not str(result).startswith("Error:"):
            return result
        if attempt < retries:
            if verbose:
                print(f"           ↻ 재시도 {attempt + 1}/{retries} ({name})")
            time.sleep(1.0 * (attempt + 1))
    return result


def run_agent(messages, max_steps=MAX_STEPS, verbose=True):
    """
    messages: 이번 user 메시지까지 들어 있는 대화 전체(system 포함).
    반환: (최종답변, 갱신된 messages)
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

        # 도구 호출 없음 → 최종 답변
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

        for tc in msg.tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments
            args = json.loads(raw_args)

            # (A) 중복 호출 차단
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

            # (C) 실행 + 재시도
            func = TOOL_FUNCS.get(name)
            result = (f"Error: unknown tool '{name}'" if func is None
                      else _run_tool_with_retry(func, args, name, verbose))

            if verbose:  # (D) 로그 정리
                preview = str(result)[:120].replace("\n", " ")
                print(f"           → {preview}{'...' if len(str(result)) > 120 else ''}")

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    # (B) step 한계: 위 is_last 분기에서 도구 없이 답을 받았어야 하지만,
    #     혹시 끝까지 도구만 부른 경우의 안전망.
    messages.append({"role": "assistant", "content": "최대 스텝 도달 — 답을 못 냈습니다."})
    return "최대 스텝 도달 — 답을 못 냈습니다.", messages


# ══════════════════════════════════════════════════════
#  5. 메모리 (하이브리드 압축)
# ══════════════════════════════════════════════════════
def compress_memory(messages, verbose=True):
    """
    너무 길면: ① 오래된 구간의 긴 tool 결과를 비우고 ② 그 구간을 LLM으로 요약.
    system 메시지와 최근 KEEP_RECENT_MSGS개는 보존.
    """
    if len(messages) <= COMPRESS_THRESHOLD:
        return messages

    system_msg = messages[0]
    recent = messages[-KEEP_RECENT_MSGS:]
    middle = messages[1:-KEEP_RECENT_MSGS]
    if not middle:
        return messages

    # 안전장치: 최근 구간이 tool 메시지로 시작하면(짝인 assistant가 잘려나가면)
    # API가 에러를 내므로, 그 tool 메시지를 요약 대상(middle)으로 되돌린다.
    while recent and recent[0].get("role") == "tool":
        middle.append(recent.pop(0))
    if not recent:
        return messages

    if verbose:
        print(f"   🗜️  메모리 압축 시작: 중간 {len(middle)}개 메시지 → 요약")

    # ① 긴 tool 결과 비우기
    for m in middle:
        if m.get("role") == "tool" and len(str(m.get("content", ""))) > 80:
            m["content"] = "[이전 도구 결과 생략됨]"

    # ② 요약
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
    summary_msg = {
        "role": "system",
        "content": f"[이전 대화 요약]\n{summary_resp.choices[0].message.content}",
    }
    new_messages = [system_msg, summary_msg] + recent
    if verbose:
        print(f"   🗜️  압축 완료: {len(messages)}개 → {len(new_messages)}개")
    return new_messages


# ══════════════════════════════════════════════════════
#  6. 계획 (Plan-and-Execute)
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
    plan = chat(planning_messages).choices[0].message.content.strip()
    if verbose:
        print(f"\n📋 계획:\n{plan}\n")
    return plan


def run_planning_agent(messages, user_input, verbose=True):
    """계획을 세운 뒤, 그 계획을 길잡이로 run_agent를 실행한다."""
    global _seen_calls_this_turn
    _seen_calls_this_turn = set()

    plan = make_plan(user_input, verbose=verbose)
    messages.append({"role": "user", "content": user_input})

    if plan != "DIRECT":
        messages.append({
            "role": "system",
            "content": f"Follow this plan to answer the user (adapt if needed):\n{plan}",
        })

    answer, messages = run_agent(messages, verbose=verbose)

    # 일회용 계획 힌트는 대화 기록에서 제거(메모리 오염 방지)
    messages = [m for m in messages
                if not (m.get("role") == "system"
                        and str(m.get("content", "")).startswith("Follow this plan"))]

    messages = compress_memory(messages, verbose=verbose)
    return answer, messages


# ══════════════════════════════════════════════════════
#  7. 진입점
# ══════════════════════════════════════════════════════
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