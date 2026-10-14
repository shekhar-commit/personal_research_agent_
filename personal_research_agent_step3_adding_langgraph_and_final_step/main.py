import os
import uuid
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from agent import build_graph

load_dotenv()

def printable_content(content):
    if isinstance(content, str):
        return content
    return str(content)

def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY is missing. Add it to your .env file.")
        return

    # SQLite checkpoints persist graph state between program runs.
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as checkpointer:
        graph = build_graph(checkpointer)

        # Reuse this thread ID to continue the saved conversation.
        thread_id = "default"
        print("\nPersonal Research AI Agent — Step 3 (LangGraph + Groq)")
        print("Commands: /history, /state, /new, /exit")
        print("Try: Search the web for the latest Python release\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            command = user_input.lower()

            if command in {"/exit", "exit", "quit"}:
                print("Goodbye!")
                break

            config = {"configurable": {"thread_id": thread_id}}

            if command == "/new":
                thread_id = str(uuid.uuid4())
                print("Started a new conversation. Previous checkpoints are preserved.")
                continue

            if command == "/history":
                snapshot = graph.get_state(config)
                messages = snapshot.values.get("messages", []) if snapshot.values else []
                if not messages:
                    print("No messages in this conversation yet.")
                else:
                    for message in messages:
                        role = getattr(message, "type", "message")
                        content = printable_content(getattr(message, "content", ""))
                        # Keep history readable; tool-call payloads are not printed.
                        if role == "human":
                            label = "You"
                        elif role == "ai":
                            label = "Agent"
                        elif role == "tool":
                            label = f"Tool ({getattr(message, 'name', 'unknown')})"
                        else:
                            label = role
                        if content:
                            print(f"{label}: {content}")
                continue

            if command == "/state":
                snapshot = graph.get_state(config)
                if not snapshot.values:
                    print("No saved graph state for this conversation yet.")
                else:
                    print("Current graph state:")
                    print("  messages:", len(snapshot.values.get("messages", [])))
                    print("  next nodes:", snapshot.next or "none (graph is idle)")
                    print("  checkpoint:", snapshot.config.get("configurable", {}).get("checkpoint_id", "available"))
                continue

            try:
                result = graph.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config=config,
                )
                messages = result.get("messages", [])
                final_message = messages[-1] if messages else None
                answer = getattr(final_message, "content", "") if final_message else ""
                print(f"\nAgent: {printable_content(answer)}\n")
            except Exception as exc:
                message = str(exc)
                if "429" in message or "rate_limit" in message.lower() or "rate limit" in message.lower():
                    print("\nAgent: Groq rate limit reached. Check your Groq console or wait for the limit to reset.\n")
                elif "401" in message or "authentication" in message.lower() or "api key" in message.lower():
                    print("\nAgent: Groq authentication failed. Check GROQ_API_KEY in .env.\n")
                else:
                    print(f"\nAgent error: {message}\n")

if __name__ == "__main__":
    main()
