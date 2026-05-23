"""
Engram — Interactive Chat
Run: python chat.py
"""

from engram.core.claude import EngramOllama


def main() -> None:
    print("=== Engram — AI with Intelligent Memory ===")
    print("Type 'quit' to exit")
    print("Type 'memory' to see current memory state")
    print("Type 'decay' to run a decay cycle")
    print("===========================================")
    print()

    client = EngramOllama()
    print(client.memory_summary())
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "memory":
            print(f"\n{client.memory_summary()}\n")
            continue

        if user_input.lower() == "decay":
            client.run_decay()
            print(f"\ndecay cycle complete\n{client.memory_summary()}\n")
            continue

        print("\nEngram: ", end="", flush=True)
        response = client.chat(user_input)
        print(response)
        print()

    client.close()
    print("goodbye.")


if __name__ == "__main__":
    main()