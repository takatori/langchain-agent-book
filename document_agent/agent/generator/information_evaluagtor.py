from agent.model import EvaluationResult, Interview
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

class InformationEvaluator:
    """
    情報の十分性を評価するクラス
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm.with_structured_output(EvaluationResult)

    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:        
        """
        ユーザリクエストとインタビュー結果に基づいて情報の十分性を評価する
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたは包括的な要件文書を作成するための情報の十分性を評価する専門家です。"),
                (
                    "human",
                    "以下のユーザリクエストとインタビュー結果に基づいて、包括的な情報要件文書を作成するのに十分な情報が集まったかどうかを判断してください。\n\n"
                    "ユーザリクエスト: {user_request}\n\n"
                    "インタビュー結果:\n{interview_results}",
                ),
            ]
        )
        
        chain = prompt | self.llm


        return chain.invoke(
            {
                "user_request": user_request, 
                "interview_results": "\n".join(
                    f"ペルソナ: {i.persona.name} - {i.persona.background}\n"
                    f"質問: {i.question}\n回答: {i.answer}\n"
                    for i in interviews                    
                )
            }
        )