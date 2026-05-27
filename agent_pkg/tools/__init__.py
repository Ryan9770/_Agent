"""
tools 패키지 — 도구 등록의 트리거.

이 패키지를 import하면 아래 each 모듈이 import되고, 그 안의 @tool
데코레이터들이 실행되어 registry.TOOLS / registry.TOOL_FUNCS에 등록된다.
즉 '새 도구 추가'는 (1) tools/ 안에 모듈 만들고 (2) 아래에 한 줄 import 추가
— 이 두 단계로 끝난다.
"""

from .registry import TOOLS, TOOL_FUNCS  # 다른 모듈이 from .tools import TOOLS 하도록 노출

# ── 도구 모듈 import = 등록 트리거 (순서가 곧 도구 목록 순서) ──
from . import basic       # noqa: F401  calculator, get_time
from . import rag         # noqa: F401  search_docs  (import 시 GPU 로딩!)
from . import web         # noqa: F401  web_search
from . import calendar    # noqa: F401  list_calendar_events
