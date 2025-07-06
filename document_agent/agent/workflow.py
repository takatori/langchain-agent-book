from typing import Optional, Any
from langchain_core.language_models import BaseChatModel
from agent.generator.persona_generator import PersonaGenerator
from agent.generator.interview_conductor import InterviewConductor
from agent.generator.requirements_document_generator import RequirementsDocumentGenerator
from agent.generator.information_evaluagtor import InformationEvaluator
from agent.model import InterviewState, InterviewResult, EvaluationResult
from langgraph.graph import StateGraph, END

class DocumentationAgent:
    def __init__(self, llm: BaseChatModel, k: Optional[int] = None):
        self.persona_generator = PersonaGenerator(llm, k)
        self.interview_conductor = InterviewConductor(llm)
        self.information_evaluator = InformationEvaluator(llm)
        self.requirements_document_generator = RequirementsDocumentGenerator(llm)

        self.graph = self._create_graph()

    def run(self, user_request: str) -> str:
        initial_state = InterviewState(user_request=user_request)
        final_state = self.graph.invoke(initial_state)
        return final_state["requirements_doc"]

    def _create_graph(self) -> StateGraph:
        
        workflow = StateGraph(InterviewState)

        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)
        
        workflow.set_entry_point("generate_personas")
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")

        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"}
        )

        workflow.add_edge("generate_requirements", END)

        return workflow.compile()

    def  _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        new_personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }
    
    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        new_interviews: InterviewResult = self.interview_conductor.run(
            state.user_request, state.personas[-5:]
        )
        return {
            "interviews": new_interviews.interviews,
        }
    
    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request, state.interviews
        )
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason,
        }
    
    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        requirements_doc: str = self.requirements_document_generator.run(
            state.user_request, state.interviews
        )
        return {
            "requirements_doc": requirements_doc,
        }