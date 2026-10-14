import ast
import operator
from typing import Any

from ddgs import DDGS
from langchain_core.tools import tool

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_calculate(expression: str) -> Any:
    """Evaluate basic arithmetic without eval() or arbitrary Python execution."""
    if len(expression) > 200:
        raise ValueError("Expression is too long.")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 10_000:
                raise ValueError("Exponent is too large.")
            result = _ALLOWED_OPERATORS[type(node.op)](left, right)
            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](visit(node.operand))
        raise ValueError("Only basic arithmetic is supported.")

    parsed = ast.parse(expression, mode="eval")
    return visit(parsed)

@tool
def calculator(expression: str) -> str:
    """Calculate a basic arithmetic expression, for example '125 * 48'."""
    try:
        return str(_safe_calculate(expression))
    except Exception as exc:
        return f"Calculator error: {exc}"

@tool
def web_search(query: str) -> str:
    """Search the web for current information. Input should be a concise search query."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No web results were returned."
        formatted = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "Untitled")
            body = item.get("body", "")
            url = item.get("href", "")
            formatted.append(f"{i}. {title}\n   {body}\n   URL: {url}")
        return "\n\n".join(formatted)
    except Exception as exc:
        return f"Web search failed: {exc}"

tools = [calculator, web_search]
