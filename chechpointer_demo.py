"""
Checkpointer 최소 예제 — 껐다 켜도 대화를 기억하는 영속 메모리

도구도 멀티에이전트도 없는 가장 단순한 챗 그래프에,
SqliteSaver(SQLite 파일에 State 저장)만 붙였다.

실행법:
    python checkpoint_demo.py          ← 대화 (종료: quit)
    (프로그램을 끄고 다시 실행해도, 같은 thread_id면 이전 대화를 기억함)

    python checkpoint_demo.py new      ← 새 thread_id로 시작 (기억 초기화)

핵심 한 줄:  app = builder.compile(checkpointer=checkpointer)
핵심 인자:  config={"configurable": {"thread_id": ...}}

설치 필요 시: pip install langgraph-checkpoint-sqlite
"""

import sys

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import os 
from dotenv import load_dotenv


load_dotenv()

AI_SERVER = os.getenv("AI_SERVER")

# ══════════════════════════════════════════════════════
#  모델
# ══════════════════════════════════════════════════════
model = ChatOpenAI(
    base_url=AI_SERVER,
    api_key="not-needed",
    model="local-model",
    temperature=0,
)


# ══════════════════════════════════════════════════════
#  가장 단순한 챗 그래프 (노드 하나)
#  MessagesState의 messages가 대화 history. Checkpointer가 이걸 저장한다.
# ══════════════════════════════════════════════════════
def chat_node(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)


# ══════════════════════════════════════════════════════
#  Checkpointer: 이 한 줄(+compile 인자)이 영속성의 전부
#  with 블록 안에서 db 연결을 열고 그래프를 컴파일한다.
# ══════════════════════════════════════════════════════
def main():
    # 'new' 인자를 주면 새 thread로 시작(기억 초기화), 아니면 고정 thread 재사용
    thread_id = "fresh_" + str(__import__("time").time()) if (len(sys.argv) > 1 and sys.argv[1] == "new") else "persistent_chat"
    config = {"configurable": {"thread_id": thread_id}}

    # SqliteSaver를 with로 열어야 db 연결이 제대로 관리된다
    with SqliteSaver.from_conn_string("chat_memory.db") as checkpointer:
        app = builder.compile(checkpointer=checkpointer)   # ← 영속성 ON

        print(f"=== Checkpointer 챗 (thread_id={thread_id}) ===")
        print("(종료: quit / 프로그램을 껐다 켜도 같은 thread면 대화를 기억함)\n")

        # 시작 시, 저장된 이전 대화가 있으면 몇 개 보여준다 (영속성 확인용)
        saved = app.get_state(config)
        prev = saved.values.get("messages", []) if saved.values else []
        if prev:
            print(f"💾 저장된 이전 대화 {len(prev)}개 메시지를 복원했습니다.")
            for m in prev[-4:]:
                role = getattr(m, "type", "?")
                content = str(getattr(m, "content", ""))[:60]
                print(f"   [{role}] {content}")
            print()

        while True:
            user_input = input("🙂 You: ").strip()
            if user_input.lower() in ("quit", "exit", "q", ""):
                print("종료합니다. (대화는 chat_memory.db에 저장됨)")
                break

            # thread_id만 주면 LangGraph가 이전 history를 알아서 이어붙인다
            # → 본인이 agent_pkg에서 conversation 리스트를 손으로 들고 다니던 걸 대체
            result = app.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
            )
            print(f"🤖 {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()