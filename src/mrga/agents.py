from .models import CaseInput, CaseOutput
from .rag import KeywordRetriever
from .memory import PersistentMemory


class ContextAgent:
    def run(self, case: CaseInput) -> str:
        return case.query.strip()


class RetrievalAgent:
    def __init__(self, retriever: KeywordRetriever):
        self.retriever = retriever

    def run(self, context: str):
        return self.retriever.search(context, top_k=5)


class DiagnosisAgent:
    def run(self, context: str, docs):
        text = (context + " " + " ".join(d.content for d in docs)).lower()
        if any(k in text for k in ["주파수", "rf", "frequency"]):
            return "rf_path", "HIGH", 0.86
        if any(k in text for k in ["전원", "power", "28v"]):
            return "power_path", "HIGH", 0.84
        if any(k in text for k in ["ethernet", "통신", "crc", "modem"]):
            return "communication_path", "MEDIUM", 0.78
        if "fail" in text or "불량" in text:
            return "test_failure_pattern", "MEDIUM", 0.72
        return "normal", "LOW", 0.65


class ProcedureAgent:
    ACTIONS = {
        "rf_path": ["RF 경로 점검", "주파수 변환부 점검", "RF 케이블 점검"],
        "power_path": ["전원 경로 점검", "입력전원/출력전압 확인", "전원 케이블 재체결"],
        "communication_path": ["통신 라인 점검", "링크 재협상", "커넥터 재체결"],
        "test_failure_pattern": ["동일 조건 재시험", "시험치구 점검", "시스템제어기조립체 점검"],
        "normal": ["정상 범위, 주기 모니터링"],
    }

    def run(self, cause: str):
        return self.ACTIONS.get(cause, self.ACTIONS["test_failure_pattern"])


class TrustGateAgent:
    def run(self, confidence: float, risk: str) -> bool:
        threshold = 0.75 if risk in {"HIGH", "MEDIUM"} else 0.6
        return confidence >= threshold


class FeedbackAgent:
    def run(self, approved: bool) -> str:
        return "승인됨: 권고 순서를 유지합니다." if approved else "보류됨: 현장 재확인 필요"


class MemoryAgent:
    def __init__(self, memory: PersistentMemory):
        self.memory = memory

    def run(self, case: CaseInput, out: CaseOutput):
        rec = {
            "query": case.query,
            "cause": out.cause,
            "risk": out.risk,
            "approved": out.approved,
            "feedback": out.feedback_note,
            "actions": out.recommended_actions,
        }
        self.memory.append(rec)
        return {"saved": "true", "cause": out.cause}
