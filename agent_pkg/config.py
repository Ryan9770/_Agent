"""설정 — 바뀔 수 있는 값은 전부 여기. 이 모듈은 아무것도 import하지 않는다(최하층)."""

# ── LLM (원격 DGX Spark의 llama-server, OpenAI 호환) ──
LLM_BASE_URL = "http://{DGX_SERVER}:8080/v1"
LLM_API_KEY  = "not-needed"
MODEL        = "local-model"
TEMPERATURE  = 0.3

# ── RAG (로컬 문서 검색) ──
DOCS_DIR         = "./docs"
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DEVICE     = "cuda"
RAG_MIN_SCORE    = 0.1   # search_docs 최고 유사도가 이보다 낮으면 관련 문서 없음 처리

# ── 웹 검색 (로컬 SearXNG — llama-server와 호스트가 다름에 주의) ──
SEARXNG_URL = "http://localhost:8080/search"

# ── 구글 캘린더 ──
GCAL_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
GCAL_CREDS  = "credentials.json"
GCAL_TOKEN  = "token.json"

# ── 에이전트 루프 / 메모리 ──
MAX_STEPS          = 6
COMPRESS_THRESHOLD = 16
KEEP_RECENT_MSGS   = 8

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
