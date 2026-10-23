#!/usr/bin/env python3
"""
ask.py - Interactive CLI chatbot for Orient BlackSwan History Catalogue.
"""
import sys
import uuid
import rag


def main():
    print("=" * 60)
    print("Orient BlackSwan History Bot - Terminal Chat")
    print("Ask anything about the History Catalogue 2024-25")
    print("Type 'exit' to quit.")
    print("=" * 60)

    session_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("\nAsk a question (type 'exit' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting chat. Goodbye!")
            break

        try:
            history = rag.get_session_history(session_id)
            result = rag.answer_question(user_input, history=history)

            answer = result.get("answer", "Not found")
            sources = result.get("sources", [])

            print("\n--- Answer ---")
            print(answer)

            if sources:
                pages = sorted(list({s["page"] for s in sources}))
                print(f"\n[Source Pages: {', '.join(str(p) for p in pages)}]")

            # Update session history
            rag.add_session_message(session_id, "user", user_input)
            rag.add_session_message(session_id, "assistant", answer)

        except Exception as e:
            print(f"\n[Error processing question]: {e}")


if __name__ == "__main__":
    main()
