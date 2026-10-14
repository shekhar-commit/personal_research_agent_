import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools

load_dotenv()

SYSTEM_PROMPT = """
You are a helpful Personal Research AI Agent.
You can use tools when useful:
- calculator: perform arithmetic
- web_search: search the web for current information

Rules:
1. Understand the user's actual goal before choosing a tool.
2. Use web_search for current facts or when the user explicitly asks to search the web.
3. Do not claim that you searched the web unless you actually called web_search.
4. Use a maximum of 2 web searches per user request. Avoid repeating the same query.
5. Use calculator for arithmetic instead of mental calculation when useful.
6. After tool results arrive, synthesize a direct answer and stop. Do not call tools repeatedly without a clear reason.
7. If a tool fails, explain the limitation and do not invent its result.
8. Use prior messages to resolve follow-up questions.
9. Be clear about uncertainty and include source URLs when web search results provide them.
"""

class AgentState(TypedDict):
    # add_messages appends new messages and updates messages by ID when appropriate.
    messages: Annotated[list[AnyMessage], add_messages]

def build_graph(checkpointer=None):
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    model = ChatGroq(
        model=model_name,
        temperature=0,
        max_retries=1,
    ).bind_tools(tools)

    def assistant_node(state: AgentState):
        # System instruction is added for every model call but is not persisted
        # as a repeated conversation message.
        response = model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        )
        return {"messages": [response]}

    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=checkpointer)
