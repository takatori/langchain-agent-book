import argparse
from agent.workflow import DocumentationAgent

from langchain_anthropic import ChatAnthropic

def main():

    parser = argparse.ArgumentParser(description="ユーザ要求に基づいて要件定義書を生成します。")

    parser.add_argument(
        "--task",
        type=str,
        help="作成したいアプリケーションについて記載してください"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="生成したいペルソナの人数を設定してください。デフォルトは5です。"
    )

    args = parser.parse_args()

    llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.0)
    agent = DocumentationAgent(llm, k=args.k)
    
    final_output = agent.run(user_request=args.task)

    print(final_output)

if __name__ == "__main__":
    main()