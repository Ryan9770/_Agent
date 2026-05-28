# 단일 LLM 에이전트 학습

DGX Spark의 llama-server(Gemma 계열 모델)를 기반으로,
**raw Python → 단일 에이전트 → 리팩터링 → LangGraph → 고급 패턴**까지 손으로 쌓아 올린 기록.

목표는 "에이전트 아키텍처/기술 학습". 결과물뿐 아니라 **언제 왜 안 되는지**까지 함께 배운 것이 핵심.

---

## 환경

- **추론 서버**: 원격 DGX Spark의 `llama-server` (OpenAI 호환 모드, `--parallel 4`)
- **로컬 PC**: RTX 5070 Ti, VRAM 16GB (임베딩 모델 GPU 로드용)
- **모델**: Gemma 계열 27B급, 네이티브 tool calling 지원 확인됨
- **부가 서비스**: 로컬 SearXNG(웹 검색), 로컬 ChromaDB 안 씀(파일 기반 RAG)

---

## 전체 여정 한눈에

```
[기초]        bare client → tool calling → ReAct 루프
[단일 에이전트] RAG(키워드→임베딩) → 멀티툴 라우팅 → 가드레일
              → 멀티턴 메모리(하이브리드 압축) → 캘린더 연동 → plan-execute
[리팩터링]    청소(페이즈1) → 모듈 분리(페이즈2, @tool 자동등록)
[개선]        라우팅 / 임계값 / 의미 기반 중복 차단
[프레임워크]  LangGraph 포팅 (raw 루프 → StateGraph)
[깊은 기술]   동적 재계획 → self-reflection → 멀티에이전트(supervisor)
[고급]        Checkpointer → eval → 병렬 실행
```

---

# 1부. 기초 — 챗봇과 에이전트의 경계

## 1.1 Bare client + tool calling 진단

`openai` 라이브러리로 llama-server에 붙이고, 두 가지를 확인.

- 진단 1: 기본 chat 동작 (`pong` 반환)
- 진단 2: 네이티브 tool calling 지원 (`calculator` 호출 자동 결정)

**핵심 통찰**: tool calling 지원 여부를 먼저 확인하지 않고 큰 그래프를 짜면 어디서 깨졌는지 추적이 안 된다. 이 진단 패턴은 이후 structured output, langchain `@tool` 등 새 기능 도입 시마다 재사용됨.

## 1.2 ReAct 루프 (직접 구현)

`for step in range(max_steps): chat → if not msg.tool_calls: return → tool 실행 → messages.append`

**핵심 통찰**: 챗봇과 에이전트를 가르는 한 줄은 `if not msg.tool_calls`. 도구를 부를지 / 답할지를 **모델이 매 step 스스로 판단**하는 자율 루프가 에이전트의 정체.

---

# 2부. 단일 에이전트 키우기

## 2.1 RAG: 키워드 → 임베딩

**4-A 키워드 검색**: 글자 단위 청킹 + 단어 겹침 점수. 단어가 안 겹치면 못 찾는 한계 직접 체감.

**4-B 임베딩 검색**: `paraphrase-multilingual-MiniLM-L12-v2` (로컬 GPU), 코사인 유사도. docker-compose 노이즈가 사라지고 쿠버네티스 결과만 잡힘.

**핵심 통찰**:
- 임베딩 유사도의 **절대값보다 상대 순위가 중요**. 0.2여도 다른 것보다 높으면 정답.
- 청킹이 험하면(표 중간 절단) 검색 품질의 절반이 무너짐. RAG 품질은 청킹에서 갈린다.

## 2.2 멀티툴 라우팅

도구 5종: `calculator`, `get_time`, `search_docs`, `web_search` (SearXNG), `list_calendar_events`.

**핵심 통찰**: 도구가 늘면 system 프롬프트가 **라우팅을 좌우**한다. "문서면 search_docs, 외부 사실이면 web_search" 같은 명시적 기준이 없으면 모델이 헷갈린다. 라우팅 실패는 모델 탓이 아니라 **도구 설명(description) 부족** 탓인 경우가 많음.

## 2.3 가드레일 (네 겹)

- (A) 중복 호출 차단 — `_seen_calls_this_turn`에 (도구명, 인자) 기록
- (B) step 한계 시 도구 없이 강제 마무리
- (C) 도구 실패 시 점증 백오프 재시도
- (D) 로그 정리 (긴 결과는 120자 미리보기)

**핵심 통찰**: 가드레일은 **두 층**으로 작동.
- 예방층(프롬프트): "같은 검색 반복 마라" → 모델이 애초에 안 함
- 차단층(코드): 그래도 뚫고 나오면 차단
- 평소엔 차단층이 발동 안 하는 게 정상. **안전망은 조용한 게 기본.**

## 2.4 멀티턴 메모리 + 하이브리드 압축

`run_agent`를 `messages` 인자로 받고 갱신본 반환하도록 리팩터링. `chat_turn`이 턴 단위 상태 관리.

**하이브리드 압축**: `[system] + [요약된 과거] + [최근 K턴 원본]` 구조.
- 임계값(16) 넘으면 압축 시도
- 중간 구간의 긴 tool 결과 먼저 비우기
- 그래도 길면 LLM으로 요약
- **함정**: 압축 경계가 `assistant tool_calls`와 `tool` 메시지 사이를 가르면 API가 에러. → "고아 tool 메시지를 middle로 되돌리기" 안전장치 필수.

**핵심 통찰**:
- 메모리는 영구 상태(대화)와 일회용 상태(중복 추적)를 구분해야 함.
- 압축 자체는 쉽지만, **무결성 처리(짝 안 깨기)가 코드의 절반**.

## 2.5 Google 캘린더 연동 (OAuth 정공법)

External + 테스트 사용자 + Desktop app + `credentials.json`/`token.json`.

처음엔 `list_calendar_events`(미래 일정)만 만들었으나, "내 생일 언제?" 질문에서 라우팅 실패 발견.

→ `search_calendar_events(query)` 추가. 키워드 기반 검색 + ±365일 범위. **description에 "생일·기념일 포함"을 명시**해 라우팅 고침.

**핵심 통찰**:
- 외부 API 연동은 **인증 만료라는 새로운 실패 양상**. `Error:` 규약으로 가드레일 (C) 재시도와 자연스럽게 연결.
- 도구 이름이 비슷해도 **세부 description**이 라우팅을 바꾼다.

## 2.6 Plan-and-Execute (정적)

`make_plan`이 도구 없이 계획만 생성 → `run_planning_agent`가 계획을 system 힌트로 주입해 `run_agent` 실행.

`DIRECT` 분기: 단순 질문엔 계획 없이 바로 실행 (오버헤드 방지).

**핵심 통찰**:
- 계획은 **일회용**. 다음 턴까지 들고 가면 메모리 오염. → 실행 후 제거.
- 정적 plan의 약점: 계획이 틀리면 끝까지 틀린 길로 감 (생일 질문에서 실제 겪음). → 후일 동적 재계획의 동기.

---

# 3부. 리팩터링 — "성장한 코드" 정리

## 페이즈 1: 한 파일 안에서 청소

- 위험한 import 제거 (`openai.types.beta.chatkit ...`)
- 죽은 코드 제거 (`chat_turn`, 4-A의 중복 적재)
- import 전부 맨 위로
- 설정값(`base_url`, `MODEL`, `SEARXNG_URL` 등) → 상단 한 곳에 통합
- 함수를 논리적 순서로 재배치

**원칙**: **로직은 한 줄도 안 바꿈.** 동작 동일성 보장 후 페이즈 2.

## 페이즈 2: 모듈 분리 + @tool 자동 등록

```
agent_pkg/
├── config.py           설정 전부
├── llm.py              chat() 래퍼
├── embeddings.py       공용 임베딩 (rag와 agent가 둘 다 사용)
├── memory.py           compress_memory
├── agent.py            run_agent 루프 + 가드레일
├── planner.py          make_plan, run_planning_agent
├── main.py             진입점
└── tools/
    ├── registry.py     @tool 데코레이터 + TOOLS/TOOL_FUNCS
    ├── basic.py        calculator, get_time
    ├── rag.py          search_docs
    ├── web.py          web_search
    ├── calendar.py     list_calendar_events, search_calendar_events
    └── __init__.py     도구 모듈 import = 등록 트리거
```

**의존 방향(한 방향만)**:
```
config ← (llm, embeddings, registry) ← tools/* ← (memory, agent) ← planner ← main
```

**@tool 자동 등록의 핵심**: 데코레이터는 **모듈이 import될 때 실행**. 따라서 `tools/__init__.py`에서 모든 도구 모듈을 import하는 한 줄이 등록 트리거.

**핵심 통찰**:
- 한 번에 다 갈아엎지 말고 **2페이즈**로 (청소 → 분리). 청소 안 된 상태에서 쪼개면 디버깅 지옥.
- 도구 추가가 "파일 하나 + @tool" 두 단계로 줄어드는 게 분리의 실질적 보상. `search_calendar_events` 추가가 이걸 즉시 검증.

---

# 4부. 개선 3종 — 데이터로 약점 잡기

| | 약점 | 해법 | 결과 |
|---|---|---|---|
| 라우팅 | 생일 질문이 search_docs로 잘못 갔음 | `search_calendar_events` 추가 + description 명시 | ✅ 근본 해결 |
| 임계값 | search_docs가 0.0 무관함도 통과 | `RAG_MIN_SCORE = 0.1` + 담백한 반환 메시지 | 🟡 부분 — 0.3대 어설픈 매칭은 못 막음 |
| 의미 중복 차단 | "삼성전자 주가" → "삼성전자 현재 주가" 우회 | 임베딩 유사도 ≥ 0.9면 차단 (A2) | 🟡 부분 — 검색어를 크게 바꾸면 우회 가능 |

**핵심 통찰**:
- 단일 가드레일로 모든 걸 막을 수 없음. **여러 보조 장치를 겹쳐 깔되 각각의 한계를 알고 쓴다.**
- 강한 행동 지시("다른 도구 써라")는 **다음 턴 라우팅을 오염**시키는 부작용. → 담백한 사실 통보로 톤 다운.

---

# 5부. 프레임워크 비교 — LangGraph 포팅

## 개념 1:1 대응

| 본인이 만든 것 | LangGraph |
|---|---|
| `@tool` 데코레이터 (registry.py) | LangChain `@tool` (docstring/타입에서 자동) |
| `TOOLS` / `TOOL_FUNCS` | `bind_tools()` |
| `chat()` 래퍼 | `ChatOpenAI` |
| `run_agent` 루프 | `StateGraph` + `tools_condition` |
| `messages` 손수 관리 | `MessagesState` (자동 누적) |
| `if not msg.tool_calls: return` | `tools_condition` 조건부 엣지 |
| `for tc: func(**args)` | `ToolNode(tools)` |

## 저수준 포팅: ToolNode를 커스텀 노드로 교체

`guarded_tool_node`가 `ToolNode(tools)`를 대체. 본인의 `_seen_calls_this_turn` 검사 로직을 노드 안에 이식. **전역 변수가 State의 정식 필드(`seen_calls`)로 승격**.

**핵심 통찰**:
- 본인이 raw로 만들어봤기 때문에 `tools_condition`이 "마법"이 아니라 "내 if문의 다른 표현"으로 보임. LangGraph 부터 배운 사람은 내부가 깜깜.
- 프레임워크 부품(ToolNode)도 결국 **내가 짤 수 있는 함수**. 이 깨달음이 프레임워크에 대한 막연한 두려움을 없앰.

---

# 6부. 깊은 에이전트 기술 (LangGraph 위에서)

## 6.1 동적 재계획

```
START → planner → executor → replan → (executor 또는 END)
```

- `Plan(steps: List[str])`과 `Response(response: str)`를 `Union`으로 묶어 structured output
- `past_steps`에 `Annotated[..., operator.add]` → 누적
- `recursion_limit=20` 회로 차단기

**vs 정적 plan-execute**: 한 스텝 실행할 때마다 replan이 **남은 계획을 수정**. 첫 스텝 결과(예: 오늘 날짜)가 둘째 스텝의 구체 인자를 채움.

## 6.2 self-reflection

```
START → research → draft → reflect → (need_research / need_rewrite / good_enough)
```

`Critique(verdict: Literal[3가지], feedback: str)` structured output. 회로 차단기 3회.

**중대한 발견 — self-reflection의 근본 한계**:
- reflect도 같은 모델. **자기가 모르는 건 자기 비평으로 못 잡는다.**
- 본인 모델이 1.36을 모르니, reflect가 "뭔가 이상" 감지는 해도 "뭐가 맞는지"는 끝까지 모름.
- 시점 라벨 / 우선권 지시 실험: **프롬프트는 모델의 "판단"은 조절해도 "지식 부재"는 못 메운다.**

**또 다른 발견 — 검색 건너뛰기**:
- print 디버깅으로 `create_agent`로 만든 research_agent가 web_search를 **한 번도 안 부르고** 자기 지식으로 답한 걸 발견.
- 해법: 모델한테 검색 여부를 맡기지 말고 **코드로 web_search 직접 호출**. 모델은 "정리만" 시킴.

## 6.3 멀티에이전트 (supervisor, tool-calling 방식)

4단 구조:
```
레벨 1: 진짜 도구 (web_search, calculator)
레벨 2: 전문가 에이전트 (research_agent, math_agent) — 각자 도구 보유
레벨 3: 전문가를 @tool로 감싸기 (ask_research_expert, ask_math_expert)
레벨 4: supervisor — 레벨3을 도구처럼 호출
```

**핵심 통찰**:
- "에이전트를 도구로 감싸기"는 **재귀적 발상**. 본인이 아는 tool calling 그대로인데, 도구가 함수가 아니라 또 다른 에이전트.
- supervisor의 라우팅 정확도는 **전문가 역할 분리의 명확성에 비례**. 리서치 vs 산수처럼 칼같이 갈리면 100%.
- 멀티에이전트가 항상 더 낫다는 환상은 금물. 단일 에이전트로 풀 일을 굳이 멀티로 쪼개면 오버엔지니어링.

---

# 7부. 고급 주제 — 인프라와 측정

## 7.1 Checkpointer (영속 메모리)

```python
with SqliteSaver.from_conn_string("chat_memory.db") as checkpointer:
    app = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "persistent_chat"}}
app.invoke(..., config=config)
```

- `compile(checkpointer=...)` 한 줄로 영속성 ON
- `thread_id`로 대화 식별 + 분리
- 프로그램 종료 → 재시작 후 **이전 대화 복원** 확인 ("5년차 개발자 이직 준비" → 기억)

**핵심 통찰**:
- 본인이 손으로 만든 `conversation` 리스트 관리를 **thread_id 하나**로 대체. 거기에 **영속성까지 공짜**.
- 단, Checkpointer는 **영속성**은 주지만 **압축**은 안 함. 실무는 **본인의 압축 + Checkpointer의 영속성**을 합쳐야 완성.

## 7.2 eval 파이프라인

### 도구 선택 eval (eval_tool_selection.py)

질문 8개 → 실제 호출 도구 추출 → 기대 도구와 비교 → 정확도 집계.

**측정 → 발견 → 개선 → 재측정 사이클**:
1. 측정: 87.5% (7/8). 떨어진 건 `(15+27)/6` — calculator를 안 부르고 암산.
2. 발견: **"계산이 쉬울수록 도구 건너뛰는" 경향** 데이터로 확인.
3. 개선: system_prompt에 "아무리 쉬워 보여도 계산은 무조건 calculator" 명시.
4. 재측정: **100%**. 프롬프트 한 줄이 실제 개선을 만들었다는 증명.

### 라우팅 eval (eval_routing.py)

같은 골격으로 평가 대상만 supervisor로 교체. "기대 전문가"가 실제로 위임받았나 채점.

**결과**: 7/7 (100%). 복합 케이스("애플·MS 직원 수 합산")까지 둘 다 위임됨.

**핵심 통찰**:
- eval 없이는 "잘 됐네/안 됐네"가 감. eval이 있으면 **숫자로 닫힘**.
- 한 번 골격을 만들면 평가 대상만 바꿔 다른 걸 측정 가능. 도구 선택 → 라우팅이 거의 복사.
- 측정의 가장 큰 가치는 **회귀 감지**. 프롬프트를 고쳤을 때 정말 나아졌는지(또는 딴 게 망가졌는지) 알 수 있음.

## 7.3 병렬 실행 (fan-out / fan-in)

```python
def dispatch(state):
    return [Send("worker", {"topic": t}) for t in state["topics"]]

class ParallelState(TypedDict):
    results: Annotated[List[str], operator.add]   # 병렬 결과 안전 취합
```

- `Send("worker", {...})` 리스트 반환 → worker N개 동시 실행 (fan-out)
- `operator.add` reducer → 동시에 같은 필드에 쓸 때 덮어쓰지 않고 합침 (fan-in)
- 본인 서버 `--parallel 4` 덕에 진짜 동시 처리

**실측**: 3개 worker 시작 시각이 **08:37:21**로 동일. 완료는 제각각(11/36/52초). 순차였다면 ~150초, 병렬로 66초.

**핵심 통찰**:
- "병렬 = 동시 출발"이 아니라 **"대기 시간이 겹친다"**. 발사는 마이크로초 시차 순차지만, LLM 응답 대기가 겹쳐서 총 시간이 줄어듦.
- 본인 환경에선 **두 겹의 병렬**이 작동:
  1. 클라이언트(LangGraph 비동기) — 발사 후 안 기다리고 다음 발사
  2. 서버(`--parallel 4`) — 4 슬롯에 나눠 실제 동시 연산
- GPU 공유라 N배는 안 됨. 작업이 많고 대기가 길수록 이득 큼.

---

# 만든 파일 목록

## 단일 파일 학습 (raw + 리팩터링 페이즈 1)
- `agent.py` — 가드레일·메모리·plan-execute까지 모두 들어간 단일 파일 (페이즈 1 청소 후)

## 모듈 분리 (리팩터링 페이즈 2)
- `agent_pkg/` 전체 (config / llm / embeddings / memory / agent / planner / main / tools/*)

## LangGraph 실험
- `lg_agent.py` — 본인 run_agent를 LangGraph StateGraph로 포팅 + `guarded_tool_node`
- `replan_agent.py` — 동적 재계획 (planner / executor / replan)
- `reflect_agent.py` — self-reflection (research / draft / reflect + 세 갈래 루프)
- `supervisor_agent.py` — 멀티에이전트 supervisor (4단 구조)

## 고급 주제
- `checkpoint_demo.py` — SqliteSaver 영속 메모리
- `eval_tool_selection.py` — 도구 선택 eval (87.5% → 100%)
- `eval_routing.py` — 라우팅 eval (100%)
- `parallel_demo.py` — fan-out/fan-in 병렬

## 부속
- `test_structured.py` — structured output 사전 진단
- `test_gcal.py` — Google 캘린더 인증 단독 테스트
- `docs/` — RAG용 .md 문서들 (쿠버네티스/도커 명령어 등)
- `credentials.json` / `token.json` / `chat_memory.db` — 인증·영속 데이터

---

# 가장 중요한 교훈 10가지

1. **챗봇과 에이전트의 경계는 `if not msg.tool_calls` 한 줄.** 도구 부를지 답할지를 모델이 매 step 스스로 결정하는 자율 루프가 에이전트의 정체.

2. **라우팅 실패는 모델 탓이 아니라 도구 description 부족 탓.** "search_docs는 문서, web_search는 외부"를 명시적으로 안 주면 모델이 헷갈리는 게 당연.

3. **가드레일은 두 층으로 작동.** 프롬프트(예방) + 코드(차단). 평소엔 코드 차단이 안 발동하는 게 정상 — 안전망은 조용한 게 기본.

4. **메모리는 영구 상태와 일회용 상태를 구분.** `messages`는 영구, `seen_calls`는 한 턴짜리. 둘을 섞으면 다음 턴 라우팅이 오염됨.

5. **압축 자체는 쉽고 무결성 처리가 어렵다.** assistant `tool_calls`와 `tool` 메시지의 짝을 안 깨는 안전장치가 코드의 절반.

6. **모델이 도구를 "건너뛴다".** "이건 내가 아니까 검색 안 해도 돼" 패턴. 확실히 하려면 **모델 판단에 맡기지 말고 코드로 강제 호출**.

7. **프롬프트는 모델의 "판단"은 조절해도 "지식 부재"는 못 메운다.** 시점 라벨·우선권 지시로 모델 행동은 바꿔도, 없는 지식은 생성 안 됨.

8. **self-reflection의 근본 한계: 자기가 모르는 건 자기 비평으로 못 잡는다.** reflect가 "뭔가 이상"은 알아도 "뭐가 맞는지"는 모름. reflect도 같은 모델이라.

9. **eval이 있고 없고의 차이.** 감으로는 "나아진 것 같다"에서 멈춤. 숫자로 보면 "87.5% → 100%, 7번 케이스 ❌→✅"로 닫힘. **회귀 감지가 eval의 가장 큰 가치.**

10. **프레임워크의 부품도 결국 내가 짤 수 있는 함수.** raw로 먼저 만들어봤기에 LangGraph가 마법이 아니라 "내 코드의 다른 표현"으로 보임. 이 순서가 학습 밀도를 결정.

---

# 다음 갈래 후보

- **미뤄둔 실용 작업**: 캘린더 쓰기 (스코프 변경 + 확인 절차), Gmail, RAG 청킹 개선
- **실사용으로 다듬기**: Gradio 등으로 웹 UI, 매일 쓸 워크플로 정비
- **다른 모델로 실험**: 학습 시점이 더 최신인 모델로 같은 코드 재실행 → 한계가 얼마나 풀리는지
- **더 깊은 LangGraph**: HITL (human-in-the-loop) 인터럽트, time travel(체크포인트로 되돌리기), streaming

이 문서가 다음 진입점 역할도 함. 새로 시작할 때 위 후보 중 하나를 골라 진행.