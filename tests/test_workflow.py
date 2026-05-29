import json

from mrga.workflow import MRGAPipeline
from mrga.models import RetrievalDoc
from mrga.rag import HybridRetriever, KeywordRetriever


def test_rf_query_returns_rf_path(tmp_path):
    mem = tmp_path / "mem.json"
    p = MRGAPipeline(memory_path=mem)
    out = p.run("GPRA 주파수 변환 fail 발생")
    assert out.cause == "rf_path"
    assert out.risk in {"HIGH", "MEDIUM"}
    assert len(out.recommended_actions) >= 1


def test_memory_written(tmp_path):
    mem = tmp_path / "mem.json"
    p = MRGAPipeline(memory_path=mem)
    p.run("전원 28V fail")
    data = json.loads(mem.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.1"
    assert data["episode_memory"]["history"]
    assert data["episode_memory"]["history"][-1]["cause"] == "power_path"
    assert data["verification_memory"]["history"]
    assert "approval_status" in data["verification_memory"]["history"][-1]


def test_ifcc_crc_query_prefers_communication_path(tmp_path):
    mem = tmp_path / "mem.json"
    p = MRGAPipeline(memory_path=mem)
    out = p.run("IFCC 통신 CRC fail")
    assert out.cause == "communication_path"


def test_hybrid_retriever_deduplicates_and_limits_top_k():
    fallback = KeywordRetriever([])
    h = HybridRetriever(vector_db_dir="C:/not-used", fallback=fallback)

    docs = [
        RetrievalDoc(source="절차서", content="IFCC CRC 통신 오류 점검 절차"),
        RetrievalDoc(source="절차서", content="IFCC CRC 통신 오류 점검 절차"),  # duplicate
        RetrievalDoc(source="FMEA", content="통신 CRC 오류 시 커넥터 점검"),
        RetrievalDoc(source="ICD", content="CAN 통신 프레임 규격 확인"),
    ]

    out = h._rerank_and_dedupe("IFCC 통신 CRC fail", docs, top_k=3)
    assert len(out) == 3
    assert len({d.content for d in out}) == 3


def test_hybrid_retriever_source_priority_prefers_fmea_on_tie():
    fallback = KeywordRetriever([])
    h = HybridRetriever(vector_db_dir="C:/not-used", fallback=fallback)

    docs = [
        RetrievalDoc(source="기타문서", content="통신 crc 점검"),
        RetrievalDoc(source="FMEA", content="통신 crc 점검"),
    ]
    out = h._rerank_and_dedupe("통신 crc", docs, top_k=2)
    assert out[0].source == "FMEA"


def test_vector_db_connection_state_available_on_this_host(tmp_path):
    mem = tmp_path / "mem.json"
    p = MRGAPipeline(memory_path=mem)
    # user-confirmed vector DB path should be usable on this host
    assert p.hybrid_retriever.uses_vector_db is True


def test_legacy_memory_records_are_migrated(tmp_path):
    mem = tmp_path / "legacy_mem.json"
    mem.write_text(
        json.dumps({"records": [{"query": "old", "cause": "power_path"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    p = MRGAPipeline(memory_path=mem)
    p.run("IFCC 통신 CRC fail")

    data = json.loads(mem.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.1"
    assert data["episode_memory"]["history"][0]["cause"] == "power_path"
    assert len(data["episode_memory"]["history"]) >= 2
