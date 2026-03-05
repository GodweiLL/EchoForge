"""EchoForge — multimodal ReAct agent CLI."""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

from agent import build_agent, build_user_message

load_dotenv()

_CONFIG = {"configurable": {"thread_id": "default"}}


def run() -> None:
    agent = build_agent()

    print(f"EchoForge Agent  (model: {os.environ['MODEL']})")
    print("输入文字。输入 'exit' 退出。\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Bye!")
            break

        msg = build_user_message(user_input)

        for chunk in agent.stream({"messages": [msg]}, config=_CONFIG, stream_mode="values"):
            latest = chunk["messages"][-1]
            if isinstance(latest, AIMessage) and latest.tool_calls:
                for tc in latest.tool_calls:
                    print(f"[调用工具] {tc['name']}  参数: {tc['args']}")
            elif isinstance(latest, ToolMessage):
                print(f"[工具结果] {latest.name}: {latest.content}")
            elif isinstance(latest, AIMessage) and latest.content:
                print(f"\nAgent: {latest.content}\n")


if __name__ == "__main__":
    run()
