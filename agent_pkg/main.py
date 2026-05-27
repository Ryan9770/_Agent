"""진입점 — 멀티턴 대화 루프.

실행: 패키지 폴더의 부모에서  python -m agent_pkg.main
"""

from . import config
from .planner import run_planning_agent


def main():
    conversation = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    print("\n=== 계획 수립형 에이전트 (종료: 'quit') ===")
    while True:
        user_input = input("\n🙂 You: ").strip()
        if user_input.lower() in ("quit", "exit", "q", ""):
            break
        answer, conversation = run_planning_agent(conversation, user_input)
        print(f"\n🤖 {answer}")
        print(f"   [메모리: {len(conversation)}개 메시지]")


if __name__ == "__main__":
    main()
