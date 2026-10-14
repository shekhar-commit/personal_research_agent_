
from agent import run_agent


def main():
    print("\n======================================")
    print(" Personal Research AI Agent - Step 1")
    print(" Powered by Groq")
    print("======================================")
    print("Tools: Calculator and Web Search")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        answer = run_agent(user_input)
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
