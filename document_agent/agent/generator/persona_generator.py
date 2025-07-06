from agent.model import Personas
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

class PersonaGenerator:
    def __init__(self, llm: BaseChatModel, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        # プロンプトテンプレートを定義
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "あなたはユーザインタビュー用の多様なペルソナを生成する専門家です。"),
                (
                    "human",
                    f"以下のユーザリクエストに関するインタビュー用に、{self.k}人の多様なペルソナを生成してください。\n\n"
                    "ユーザリクエスト: {user_request}\n\n"
                    "各ペルソナには名前と簡単な背景を含めてください。年齢、性別、職業、技術的専門知識において多様性を確保してください。",
                ),
            ]
        )
        # ペルソナ生成のためのチェーンを生成
        chain = prompt | self.llm
        # ペルソナ生成
        return chain.invoke({"user_request": user_request})
