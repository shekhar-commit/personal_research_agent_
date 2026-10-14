
import json
import os

from dotenv import load_dotenv
from groq import Groq

from tools import calculator, web_search

load_dotenv()

MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)

MAX_AGENT_STEPS = int(
    os.getenv("MAX_AGENT_STEPS", "4")
)

MAX_WEB_SEARCHES = 2


SYSTEM_PROMPT = """
You are a helpful Personal Research AI Agent.

Available tools:
1. calculator(expression): calculate arithmetic.
2. web_search(query): search the web.

Rules:
- Understand the user's goal before selecting a tool.
- Use web_search for current information or explicit search requests.
- Use calculator for arithmetic when useful.
- Never claim you searched unless you actually used web_search.
- Use at most two web searches per user request.
- Do not repeat the same search unnecessarily.
- Use tool results to create a clear final answer.
- If a tool fails, explain the limitation honestly.
- Stop when you have enough information to answer.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Calculate a basic arithmetic expression, "
                "for example 125 * 48."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression"
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    }
]


def execute_tool(name, arguments):
    """Execute the tool selected by the model."""

    if name == "calculator":
        return calculator(
            arguments.get("expression", "")
        )

    if name == "web_search":
        return web_search(
            arguments.get("query", "")
        )

    return f"Unknown tool: {name}"


def run_agent(user_input: str) -> str:
    """Run the agent loop for one user request."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "GROQ_API_KEY is missing. "
            "Add your API key to the .env file."
        )

    if MAX_AGENT_STEPS < 1:
        return "MAX_AGENT_STEPS must be at least 1."

    client = Groq(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_input
        }
    ]

    web_search_count = 0

    for step in range(MAX_AGENT_STEPS):
        print(
            f"[Agent step {step + 1}/{MAX_AGENT_STEPS}]"
        )

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_completion_tokens=2048
            )

        except Exception as error:
            details = str(error)
            lowered = details.lower()

            if (
                "429" in details
                or "rate limit" in lowered
                or "quota" in lowered
            ):
                return (
                    "Groq rate limit or quota reached. "
                    "Check your Groq Console. Details: "
                    + details
                )

            if (
                "401" in details
                or "authentication" in lowered
                or "api key" in lowered
            ):
                return (
                    "Groq authentication failed. "
                    "Check your API key in .env. Details: "
                    + details
                )

            return "Groq API request failed: " + details

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # No tool requested: return the final answer.
        if not tool_calls:
            return (
                message.content
                or "The model returned an empty answer."
            )

        # Add the assistant tool-call message to the history.
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments
                    }
                }
                for call in tool_calls
            ]
        })

        # Execute each requested tool and return its result.
        for call in tool_calls:
            tool_name = call.function.name

            try:
                arguments = json.loads(
                    call.function.arguments or "{}"
                )
            except json.JSONDecodeError:
                arguments = {}

            if tool_name == "web_search":
                if web_search_count >= MAX_WEB_SEARCHES:
                    result = (
                        "Web search limit reached. "
                        "Use the results already collected."
                    )
                else:
                    web_search_count += 1
                    result = execute_tool(
                        tool_name, arguments
                    )
            else:
                result = execute_tool(
                    tool_name, arguments
                )

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result)[:12000]
            })

    # If the model keeps requesting tools, ask it to finish
    # without providing any tools for this final call.
    try:
        messages.append({
            "role": "system",
            "content": (
                "Do not call more tools. Give the best final "
                "answer using the available information."
            )
        })

        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_completion_tokens=2048
        )

        return (
            final_response.choices[0].message.content
            or "Unable to produce a final answer."
        )

    except Exception as error:
        return (
            "Agent stopped after reaching its step limit. "
            "Details: " + str(error)
        )
