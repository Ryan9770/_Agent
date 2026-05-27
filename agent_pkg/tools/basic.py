"""기본 도구: calculator, get_time."""

import datetime

from .registry import tool


@tool(
    name="calculator",
    description="Evaluate a basic arithmetic expression.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. '12 * (3 + 4)'"}
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    """안전한 산술 계산만 허용."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: invalid characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="get_time",
    description="Get the current UTC time.",
    parameters={
        "type": "object",
        "properties": {"timezone": {"type": "string"}},
    },
)
def get_time(timezone: str = "UTC") -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
