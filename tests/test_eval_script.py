import csv
import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "evaluate_agent_only_with_ground_truth.py"
spec = importlib.util.spec_from_file_location("eval_gt", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)
evaluate = mod.evaluate


def test_evaluate_computes_metrics(tmp_path):
    gt = tmp_path / "ground_truth.csv"
    runs = tmp_path / "runs"
    runs.mkdir()

    with gt.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "log_id",
            "log_name",
            "gt_status",
            "gt_cause_top1",
            "gt_cause_top3",
            "gt_action_top1",
            "gt_action_top3",
        ])
        w.writerow(["x", "a.TXT", "FAIL", "communication_path", "communication_path", "통신 라인 점검", "-"])
        w.writerow(["y", "b.TXT", "PASS", "normal", "normal", "정상 범위, 주기 모니터링", "-"])

    pred_a = {
        "generated_at": "2026-05-29T00:00:00",
        "focus": {
            "file": "a.TXT",
            "cause": "communication_path",
            "recommended_actions": ["통신 라인 점검"],
        },
    }
    pred_b = {
        "generated_at": "2026-05-29T00:00:01",
        "focus": {
            "file": "b.TXT",
            "cause": "normal",
            "recommended_actions": ["정상 범위, 주기 모니터링"],
        },
    }
    (runs / "a.json").write_text(json.dumps(pred_a, ensure_ascii=False), encoding="utf-8")
    (runs / "b.json").write_text(json.dumps(pred_b, ensure_ascii=False), encoding="utf-8")

    rows, summary = evaluate(gt, runs)

    assert len(rows) == 2
    assert summary["coverage"] == 1.0
    assert summary["status_accuracy"] == 1.0
    assert summary["cause_top1_accuracy"] == 1.0
    assert summary["action_top1_contains_accuracy"] == 1.0
