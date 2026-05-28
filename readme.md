# LLM Agent — From Scratch to LangGraph

로컬 `llama-server` 위에서 단일 에이전트를 **밑바닥부터** 만들고,
리팩터링하고, LangGraph로 포팅한 뒤, 고급 패턴(plan-execute, self-reflection,
multi-agent, checkpointer, eval, 병렬 실행)까지 쌓아 올린 학습 레포.

> "프레임워크가 뭘 추상화하는지"를 알려면 먼저 직접 만들어봐야 한다는 전제로,
> raw Python(`agent_pkg/`)부터 시작해 LangGraph로 점진적으로 옮겨갔다.

상세한 단계별 학습 여정과 거기서 얻은 통찰은 **[JOURNEY.md](./JOURNEY.md)** 참고.

---

## 주요 특징

- **단일 파일 → 모듈 분리**: 한 파일에 쌓아 올린 코드를 `agent_pkg/`로 재구성. `@tool` 데코레이터 기반 자동 등록.
- **5종 도구**: calculator / get_time / search_docs(RAG) / web_search(SearXNG) / Google Calendar.
- **하이브리드 메모리 압축**: 최근 K턴은 원본, 그 이전은 LLM 요약. tool_calls 짝 안전장치 포함.
- **Plan-and-Execute**: 단순 질문은 `DIRECT`, 복합 질문은 계획 후 실행.
- **LangGraph 포팅**: `run_agent`의 `for` 루프를 StateGraph로. 가드레일을 커스텀 ToolNode로 이식.
- **고급 패턴**: 동적 재계획, self-reflection, supervisor 멀티에이전트, SQLite Checkpointer, 도구·라우팅 eval, fan-out/fan-in 병렬.

---

## 요구 사항

- Python 3.10+
- OpenAI 호환 모드로 떠 있는 LLM 서버 (예: `llama-server`)
- (선택) 로컬 SearXNG — 웹 검색 도구용
- (선택) Google Cloud OAuth 클라이언트 — 캘린더 도구용
- (선택) CUDA GPU — 임베딩 모델 로컬 실행용 (CPU도 가능)

---

## 설치

```bash
git clone <this-repo>
cd <this-repo>

# 가상환경 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성
pip install -r requirements.txt

# 환경 변수 (서버 주소 등)
cp .env.example .env
# .env를 열어서 본인 환경에 맞게 채우기
```

`.env`에 채워야 할 값:

```
LLM_BASE_URL=http://your-llama-server:8080/v1
LLM_API_KEY=not-needed
SEARXNG_URL=http://localhost:8080/search
```

Google 캘린더를 쓰려면 별도로 `credentials.json`을 프로젝트 루트에 두세요.
([Google Cloud Console](https://console.cloud.google.com)에서 Desktop app OAuth 클라이언트 생성)

RAG용 문서는 `docs/` 폴더에 `.md` 또는 `.txt`로 두면 자동 로드됩니다.

---

## 실행

### 메인 에이전트 (모듈 분리 본체)

```bash
python -m agent_pkg.main
```

가드레일, 하이브리드 메모리, plan-and-execute가 다 포함된 멀티턴 대화 에이전트.

### LangGraph 실험들

각 파일은 독립 실행 가능한 단일 데모.

| 파일 | 패턴 |
|---|---|
| `lg_agent.py` | 기본 ReAct 그래프 + 커스텀 가드레일 ToolNode |
| `replan_agent.py` | Plan-and-Execute with dynamic replanning |
| `reflect_agent.py` | Self-reflection (research → draft → reflect) |
| `supervisor_agent.py` | Multi-agent supervisor (tool-calling 방식) |
| `checkpoint_demo.py` | SqliteSaver 영속 메모리 (껐다 켜도 대화 유지) |
| `eval_tool_selection.py` | 도구 선택 정확도 eval |
| `eval_routing.py` | Supervisor 위임 정확도 eval |
| `parallel_demo.py` | fan-out/fan-in 병렬 실행 |

```bash
python lg_agent.py
python replan_agent.py
# ...
```

---

## 프로젝트 구조

```
.
├── agent_pkg/                # 모듈 분리된 메인 에이전트
│   ├── config.py             # 모든 설정 (env에서 로드)
│   ├── llm.py                # llama-server 호출 래퍼
│   ├── embeddings.py         # 공용 임베딩 (rag와 agent가 공유)
│   ├── memory.py             # 하이브리드 압축
│   ├── agent.py              # run_agent 루프 + 가드레일
│   ├── planner.py            # Plan-and-Execute
│   ├── main.py               # 진입점
│   └── tools/
│       ├── registry.py       # @tool 데코레이터 + TOOLS/TOOL_FUNCS
│       ├── basic.py          # calculator, get_time
│       ├── rag.py            # 임베딩 기반 문서 검색
│       ├── web.py            # SearXNG 웹 검색
│       └── calendar.py       # Google Calendar (읽기 + 키워드 검색)
│
├── lg_agent.py               # LangGraph 포팅 + 커스텀 ToolNode
├── replan_agent.py           # 동적 재계획
├── reflect_agent.py          # Self-reflection
├── supervisor_agent.py       # 멀티에이전트
├── checkpoint_demo.py        # 영속 메모리
├── eval_tool_selection.py    # 도구 선택 eval
├── eval_routing.py           # 라우팅 eval
├── parallel_demo.py          # 병렬 실행
│
├── docs/                     # RAG용 문서 (.md, .txt)
├── .env.example              # 환경 변수 템플릿
├── requirements.txt
├── JOURNEY.md                # 상세 학습 기록
└── README.md
```

---

## 학습 기록

이 레포의 핵심은 **단계별로 무엇을 배우고 어디서 막혔는지의 기록**입니다.
"되는 코드"뿐 아니라 "안 될 때 왜 안 되는지"까지 다룬 회고가 [JOURNEY.md](./JOURNEY.md)에 있습니다.

특히 다음과 같은 실험적 발견이 담겨 있습니다:

- 임베딩 검색의 한계 (절대 점수 vs 상대 순위)
- self-reflection의 근본 한계 (자기가 모르는 건 자기 비평으로 못 잡는다)
- 로컬 모델의 "도구 건너뛰기" 패턴과 그 해결
- 프롬프트로 풀 수 있는 것과 없는 것의 경계
- 메모리 압축의 무결성 함정 (tool_calls 짝 안 깨기)
- eval 기반 개선의 측정→발견→개선→재측정 사이클

---

## 라이선스

MIT

---

## 참고

- 사용 모델: Gemma 계열 27B급 (네이티브 tool calling 지원)
- 추론 환경: 원격 GPU 서버 (OpenAI 호환 모드)
- 임베딩 모델: `paraphrase-multilingual-MiniLM-L12-v2` (한국어/영어)

다른 모델로 돌리려면 `.env`의 `LLM_BASE_URL`만 바꾸면 됩니다.
Tool calling을 지원하는 모델이어야 합니다.