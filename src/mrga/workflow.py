from pathlib import Path
from .models import CaseInput, CaseOutput, RetrievalDoc
from .rag import KeywordRetriever
from .memory import PersistentMemory
from .agents import (
    ContextAgent,
    RetrievalAgent,
    DiagnosisAgent,
    ProcedureAgent,
    TrustGateAgent,
    FeedbackAgent,
    MemoryAgent,
)


class MRGAPipeline:
    def __init__(self, memory_path: str | Path = "out/persistent_memory.json"):
        corpus = [
            RetrievalDoc("ICD", "전원 28V 공급 불안정 시 FAIL 및 재시도 증가"),
            RetrievalDoc("FMEA", "RF 주파수변환 실패 시 down-converter 경로 우선 점검"),
            RetrievalDoc("절차서", "Ethernet CRC 오류 시 링크/케이블/커넥터 확인"),
            RetrievalDoc("정비이력", "동일 조건 재시험으로 일시적 FAIL 복구 사례 다수"),
        ]
        self.context_agent = ContextAgent()
        self.retrieval_agent = RetrievalAgent(KeywordRetriever(corpus))
        self.diagnosis_agent = DiagnosisAgent()
        self.procedure_agent = ProcedureAgent()
        self.trust_gate_agent = TrustGateAgent()
        self.feedback_agent = FeedbackAgent()
        self.memory_agent = MemoryAgent(PersistentMemory(Path(memory_path)))

    def run(self, query: str) -> CaseOutput:
        case = CaseInput(query=query)
        context = self.context_agent.run(case)
        docs = self.retrieval_agent.run(context)
        cause, risk, conf = self.diagnosis_agent.run(context, docs)
        actions = self.procedure_agent.run(cause)
        approved = self.trust_gate_agent.run(conf, risk)

        out = CaseOutput(
            risk=risk,
            cause=cause,
            recommended_actions=actions,
            evidence=docs,
            confidence=conf,
            approved=approved,
        )
        out.feedback_note = self.feedback_agent.run(out.approved)
        out.memory_updates = self.memory_agent.run(case, out)
        return out
