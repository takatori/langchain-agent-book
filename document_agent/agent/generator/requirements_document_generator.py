from agent.model import Interview
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RequirementsDocumentGenerator:
    """
    ユーザ要求とインタビュー結果に基づいて最終的な要件定義所を生成する
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたは収集した情報に基づいて要件定義書を作成する専門家です。"),
                (
                    "human",
                    "以下のユーザリクエストと複数のペルソナからのインタビュー結果に基づいて、要件定義書を作成してください。\n\n"
                    "ユーザリクエスト: {user_request}\n\n"
                    "インタビュー結果:\n{interview_results}\n"
                    "要件文書には以下のセクションを含めてください:\n"
                    "1. プロジェクト概要\n"
                    "2. 主要機能\n"
                    "3. 非機能要件\n"
                    "4. 制約条件\n"
                    "5. ターゲットユーザー\n"
                    "6. 優先順位\n"
                    "7. リスクと軽減策\n\n"
                    "出力は必ず日本語でお願いします。\n\n要件文書:",
                ),
            ]
        )
        
        
        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                    f"質問: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )

