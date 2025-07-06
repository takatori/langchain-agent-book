from agent.model import Persona, Interview, InterviewResult
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class InterviewConductor:
    """
    生成されたペルソナに対してインタビューを実施する
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # 質問を生成
        questions = self._generate_questions(user_request, personas)
        # 回答を生成
        answers = self._generate_answers(personas, questions)
        # インタビュー結果を生成
        interviews = self._create_interviews(personas, questions, answers)
        return InterviewResult(interviews=interviews)

    def _generate_questions(self, user_request: str, personas: list[Persona]) -> list[str]:
        """
        ユーザリクエストとペルソナに基づいてインタビューの質問を生成する
        """
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system", 
                    "あなたはユーザ要件に基づいて適切な質問を生成する専門家です。"
                ),
                (
                    "human",
                    "以下のペルソナに関連するユーザリクエストについて、1つの質問を生成してください。\n\n"
                    "ユーザリクエスト: {user_request}\n\n"
                    "ペルソナ: {persona_name} - {persona_background}\n\n"
                    "質問は具体的で、このペルソナの視点から重要な情報を引き出すように設計してください。",
                ),
            ]
        )
        # 質問生成のためのチェーンを生成
        question_chain = question_prompt | self.llm | StrOutputParser()

        # 各ペルソナに対する質問クエリを生成
        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]

        # 質問をバッチで処理
        return question_chain.batch(question_queries)
    

    def _generate_answers(self, personas: list[Persona], questions: list[str]) -> list[str]:
        """
        各ペルソナに対して質問に基づいて回答を生成する
        """
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたは以下のペルソナとして回答しています: {persona_name} - {persona_background}",
                ),
                (
                    "human",
                    "質問: {question}"
                ),
            ]
        )
        # 回答生成のためのチェーンを生成
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        # 各ペルソナと質問に対する回答クエリを生成
        answer_queries = [
            {
                "question": question,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona, question in zip(personas, questions)
        ]

        # 回答をバッチで処理
        return answer_chain.batch(answer_queries)

    def _create_interviews(self, personas: list[Persona], questions: list[str], answers: list[str]) -> list[Interview]:
        """
        質問と回答をペルソナごとにまとめてインタビュー結果を生成する
        """
        return [
            Interview(
                persona=persona,
                question=question,
                answer=answer
            )
            for persona, question, answer in zip(personas, questions, answers)
        ]