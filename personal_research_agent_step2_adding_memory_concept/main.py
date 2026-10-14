from memory import ConversationMemory
from agent import run_agent


def main():
    print("=" * 58)
    print("          PERSONAL RESEARCH AI AGENT")
    print("=" * 58)
    print("Tools: Calculator, Web Search")
    print("Memory: Persistent JSON history")
    print("Commands: /history, /state, /clear, exit")
    memory = ConversationMemory("default")
    print(f"Loaded {len(memory.get_history())} saved messages.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        command = user_input.lower()
        if command in ("exit", "quit"):
            print("Goodbye!")
            break
        if command == "/history":
            history = memory.get_history()
            if not history:
                print("No conversation history yet.")
            else:
                for item in history:
                    print(f"\n{item['role'].capitalize()}: {item['content']}")
            continue
        if command == "/state":
            print("\nAgent state:")
            print(memory.get_state())
            continue
        if command == "/clear":
            confirm = input("Clear saved history and state? Type yes to confirm: ").strip().lower()
            if confirm == "yes":
                memory.clear()
                print("Conversation memory cleared.")
            else:
                print("Cancelled.")
            continue
        answer = run_agent(user_input, memory)
        print(f"\nAgent: {answer}")
        print("-" * 58)


if __name__ == "__main__":
    main()
