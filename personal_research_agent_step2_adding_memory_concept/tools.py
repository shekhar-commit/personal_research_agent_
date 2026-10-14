def calculator(a: float, b: float, operation: str) -> dict:
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y,
    }
    if operation not in operations:
        return {"error": "Unsupported operation."}
    if operation == "divide" and float(b) == 0:
        return {"error": "Cannot divide by zero."}
    try:
        return {"a": a, "b": b, "operation": operation, "result": operations[operation](float(a), float(b))}
    except (TypeError, ValueError, OverflowError):
        return {"error": "Invalid calculator arguments."}


def web_search(query: str) -> dict:
    if not isinstance(query, str) or not query.strip():
        return {"error": "Search query cannot be empty.", "results": []}
    try:
        from ddgs import DDGS
        with DDGS() as search_client:
            raw = list(search_client.text(query.strip(), max_results=5))
        results = [{"title": x.get("title", ""), "url": x.get("href", x.get("url", "")), "snippet": x.get("body", x.get("snippet", ""))} for x in raw]
        return {"query": query, "results": results}
    except Exception as exc:
        print(f"Web search failed: {type(exc).__name__}: {exc}")
        return {"query": query, "results": [], "error": "Web search is temporarily unavailable."}

AVAILABLE_TOOLS = {"calculator": calculator, "web_search": web_search}
