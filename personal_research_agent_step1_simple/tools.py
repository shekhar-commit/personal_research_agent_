
import ast
import operator
from ddgs import DDGS

# Safe arithmetic operators
ALLOWED_OPERATORS = {
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


def calculator(expression: str) -> str:
    """Safely calculate a basic arithmetic expression."""

    if not expression or len(expression) > 200:
        return "Calculator error: invalid or excessively long expression."

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp):
            operation = type(node.op)

            if operation not in ALLOWED_OPERATORS:
                raise ValueError("Unsupported arithmetic operator.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if operation is ast.Pow and abs(right) > 100:
                raise ValueError("Exponent is too large.")

            result = ALLOWED_OPERATORS[operation](left, right)

            if isinstance(result, (int, float)) and abs(result) > 1e100:
                raise ValueError("Result is too large.")

            return result

        if isinstance(node, ast.UnaryOp):
            operation = type(node.op)

            if operation not in ALLOWED_OPERATORS:
                raise ValueError("Unsupported unary operator.")

            return ALLOWED_OPERATORS[operation](
                evaluate(node.operand)
            )

        raise ValueError("Only basic arithmetic is supported.")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(evaluate(tree))

    except Exception as error:
        return f"Calculator error: {error}"


def web_search(query: str) -> str:
    """Search the web using DuckDuckGo."""

    if not query or not query.strip():
        return "Web search error: query is empty."

    try:
        with DDGS() as search_client:
            results = list(
                search_client.text(query, max_results=5)
            )

        if not results:
            return "No web search results were found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            body = result.get("body", "")
            url = result.get("href", "")

            formatted_results.append(
                f"{index}. {title}\n"
                f"   {body}\n"
                f"   URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as error:
        return f"Web search failed: {error}"
