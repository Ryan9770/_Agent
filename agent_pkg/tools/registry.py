"""
도구 레지스트리 — 자동 등록의 심장.

@tool(...) 데코레이터를 함수에 붙이면, import 시점에 그 함수가
TOOLS(스키마 목록)와 TOOL_FUNCS(이름→함수)에 자동 등록된다.

주의: 데코레이터는 '모듈이 import될 때' 실행된다. 따라서 도구가 정의된
모듈을 어딘가에서 반드시 한 번 import해야 등록이 일어난다.
(그 트리거가 tools/__init__.py다.)
"""

TOOLS = []        # 모델에게 줄 스키마 목록
TOOL_FUNCS = {}   # 이름 → 실제 파이썬 함수


def tool(name, description, parameters):
    """함수를 TOOLS/TOOL_FUNCS에 등록하고, 함수는 그대로 반환한다."""
    def decorator(func):
        TOOLS.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        })
        TOOL_FUNCS[name] = func
        return func
    return decorator
