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
    KEYWORDS = {
        "rf_path": ["주파수", "rf", "frequency", "down-converter", "up-converter", "증폭"],
        "power_path": ["전원", "power", "28v", "전압", "전류", "bus voltage"],
        "communication_path": ["ethernet", "통신", "crc", "modem", "rs-422", "rs-232", "can"],
        "test_failure_pattern": ["fail", "failed", "불량", "고장", "재시험", "retry"],
    }

    # Source reliability/intent weights
    SOURCE_WEIGHTS = {
        "fmea": 1.4,
        "절차서": 1.2,
        "icd": 1.1,
        "정비이력": 1.0,
    }

    def _source_weight(self, source: str) -> float:
        s = (source or "").lower()
        for key, w in self.SOURCE_WEIGHTS.items():
            if key in s:
                return w
        return 1.0

    def _count_hits(self, text: str, tokens: list[str]) -> int:
        return sum(1 for t in tokens if t in text)

    def _token_hit_count(self, text: str, tokens: list[str]) -> int:
        return sum(text.count(t) for t in tokens)

    @staticmethod
    def _has_any(text: str, tokens: list[str]) -> bool:
        return any(t in text for t in tokens)

    def run(self, context: str, docs):
        ctx = (context or "").lower()

        comm_tokens = ["통신", "crc", "ethernet", "modem", "rs-422", "rs-232", "can", "packet", "frame", "ifcc", "uart", "serial", "timeout", "handshake", "ack", "nack"]
        power_tokens = ["전원", "power", "28v", "전압", "전류", "voltage", "current"]
        rf_tokens = ["주파수", "rf", "frequency", "증폭"]
        fail_tokens = ["fail", "failed", "불량", "고장", "retry", "재시험", "오류", "error"]
        pass_tokens = ["pass", "정상"]

        has_comm_intent = self._has_any(ctx, comm_tokens)
        has_power_intent = self._has_any(ctx, power_tokens)
        has_rf_intent = self._has_any(ctx, rf_tokens)
        has_fail_signal = self._has_any(ctx, fail_tokens)
        has_pass_signal = self._has_any(ctx, pass_tokens)

        comm_ctx_hits = self._token_hit_count(ctx, comm_tokens)
        power_ctx_hits = self._token_hit_count(ctx, power_tokens)
        rf_ctx_hits = self._token_hit_count(ctx, rf_tokens)

        if has_pass_signal and not has_fail_signal:
            return "normal", "LOW", 0.72

        scores = {
            "rf_path": 0.0,
            "power_path": 0.0,
            "communication_path": 0.0,
            "test_failure_pattern": 0.0,
        }

        # 1) User-query/context signal (strongest)
        for cause, tokens in self.KEYWORDS.items():
            base_weight = 1.2 if cause == "test_failure_pattern" else 2.5
            scores[cause] += base_weight * self._count_hits(ctx, tokens)

        # Explicit query-intent boost to avoid evidence drift
        if has_comm_intent:
            scores["communication_path"] += 4.0
        if has_power_intent:
            scores["power_path"] += 3.0
        if has_rf_intent:
            scores["rf_path"] += 3.0

        # Platform/line-name clues often carry communication intent in JAN logs.
        if any(k in ctx for k in ["ifcc", "uart", "serial", "rs-422", "rs-232", "can"]):
            scores["communication_path"] += 2.0

        # Fail-coupled communication signals should override routine power/rf boilerplate lines.
        if has_fail_signal and any(k in ctx for k in ["crc", "packet", "timeout", "handshake", "ack", "nack", "통신", "retry", "재시험"]):
            scores["communication_path"] += 4.0
            scores["power_path"] *= 0.75
            scores["rf_path"] *= 0.70

        # If hint says communication and fail signal exists, trust comm path more strongly.
        if "hint_cause=communication_path" in ctx and has_fail_signal:
            scores["communication_path"] += 2.5

        # Context ratio calibration: emphasize dominant intent in the user/query text.
        if comm_ctx_hits >= power_ctx_hits + 2 and comm_ctx_hits >= rf_ctx_hits + 2:
            scores["communication_path"] += 2.5
            scores["power_path"] *= 0.75
            scores["rf_path"] *= 0.75
        elif power_ctx_hits >= comm_ctx_hits + 2 and power_ctx_hits >= rf_ctx_hits + 2:
            scores["power_path"] += 2.0
        elif rf_ctx_hits >= comm_ctx_hits + 2 and rf_ctx_hits >= power_ctx_hits + 2:
            scores["rf_path"] += 2.0

        # 2) Retrieved evidence signal (weighted by source)
        for d in docs or []:
            text = (d.content or "").lower()
            sw = self._source_weight(getattr(d, "source", ""))
            for cause, tokens in self.KEYWORDS.items():
                scores[cause] += sw * self._count_hits(text, tokens)

        if has_comm_intent and not has_power_intent:
            scores["power_path"] *= 0.7
        if has_comm_intent and not has_rf_intent:
            scores["rf_path"] *= 0.7

        best_cause = max(scores, key=scores.get)

        # Calibration: prevent generic over-prediction when technical causes are close.
        if best_cause == "test_failure_pattern":
            technical_causes = ["communication_path", "power_path", "rf_path"]
            tech_best = max(technical_causes, key=lambda c: scores[c])
            tech_score = scores[tech_best]
            generic_score = scores["test_failure_pattern"]
            if tech_score >= 2.0 and tech_score >= generic_score * 0.7:
                best_cause = tech_best

        best_score = scores[best_cause]

        if best_score <= 0:
            return "normal", "LOW", 0.65

        # Confidence normalization for trust gate
        conf = min(0.95, 0.68 + min(best_score, 8.0) * 0.035)

        if best_cause in {"rf_path", "power_path"}:
            risk = "HIGH"
        elif best_cause in {"communication_path", "test_failure_pattern"}:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return best_cause, risk, round(conf, 4)


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
        episode_rec = {
            "query": case.query,
            "cause": out.cause,
            "risk": out.risk,
            "approved": out.approved,
            "feedback": out.feedback_note,
            "actions": out.recommended_actions,
            "confidence": out.confidence,
            "evidence_sources": [d.source for d in out.evidence],
        }
        verification_rec = {
            "query": case.query,
            "approval_status": "approved" if out.approved else "hold",
            "risk": out.risk,
            "confidence": out.confidence,
            "feedback": out.feedback_note,
        }

        self.memory.append_episode(episode_rec)
        self.memory.append_verification(verification_rec)
        return {"saved": "true", "cause": out.cause, "schema": "1.1"}
