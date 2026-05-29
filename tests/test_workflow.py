from mrga.workflow import MRGAPipeline


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
    txt = mem.read_text(encoding="utf-8")
    assert "power_path" in txt


def test_vector_db_connection_state_available_on_this_host(tmp_path):
    mem = tmp_path / "mem.json"
    p = MRGAPipeline(memory_path=mem)
    # user-confirmed vector DB path should be usable on this host
    assert p.hybrid_retriever.uses_vector_db is True
