import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class EvalRow:
    log_name: str
    gt_status: str
    gt_cause_top1: str
    gt_action_top1: str
    pred_status: str
    pred_cause_top1: str
    pred_action_top1: str
    status_match: int
    cause_top1_match: int
    action_top1_contains: int


def normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def infer_status_from_cause(cause: str) -> str:
    return "PASS" if normalize_text(cause) == "normal" else "FAIL"


def parse_time(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.min


def load_predictions(runs_dir: Path) -> dict[str, dict[str, Any]]:
    by_focus: dict[str, dict[str, Any]] = {}
    for p in runs_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        focus = data.get("focus") or {}
        focus_file = focus.get("file")
        if not focus_file:
            continue
        prev = by_focus.get(focus_file)
        if prev is None:
            by_focus[focus_file] = data
            continue
        if parse_time((data.get("generated_at") or "")) >= parse_time((prev.get("generated_at") or "")):
            by_focus[focus_file] = data
    return by_focus


def evaluate(ground_truth_csv: Path, runs_dir: Path) -> tuple[list[EvalRow], dict[str, Any]]:
    preds = load_predictions(runs_dir)
    rows: list[EvalRow] = []

    with ground_truth_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for gt in reader:
            log_name = (gt.get("log_name") or "").strip()
            pred = preds.get(log_name)

            if pred:
                focus = pred.get("focus") or {}
                pred_cause = str(focus.get("cause") or "")
                pred_status = infer_status_from_cause(pred_cause)
                rec_actions = focus.get("recommended_actions") or []
                pred_action = str(rec_actions[0]) if rec_actions else ""
            else:
                pred_cause = ""
                pred_status = ""
                pred_action = ""

            gt_status = str(gt.get("gt_status") or "")
            gt_cause = str(gt.get("gt_cause_top1") or "")
            gt_action = str(gt.get("gt_action_top1") or "")

            status_match = int(normalize_text(pred_status) == normalize_text(gt_status)) if pred else 0
            cause_match = int(normalize_text(pred_cause) == normalize_text(gt_cause)) if pred else 0
            action_contains = int(normalize_text(gt_action) in normalize_text(pred_action)) if pred else 0

            rows.append(
                EvalRow(
                    log_name=log_name,
                    gt_status=gt_status,
                    gt_cause_top1=gt_cause,
                    gt_action_top1=gt_action,
                    pred_status=pred_status,
                    pred_cause_top1=pred_cause,
                    pred_action_top1=pred_action,
                    status_match=status_match,
                    cause_top1_match=cause_match,
                    action_top1_contains=action_contains,
                )
            )

    n = len(rows)
    covered = sum(1 for r in rows if r.pred_cause_top1)
    summary = {
        "total": n,
        "covered": covered,
        "coverage": round(covered / n, 4) if n else 0.0,
        "status_accuracy": round(sum(r.status_match for r in rows) / n, 4) if n else 0.0,
        "cause_top1_accuracy": round(sum(r.cause_top1_match for r in rows) / n, 4) if n else 0.0,
        "action_top1_contains_accuracy": round(sum(r.action_top1_contains for r in rows) / n, 4) if n else 0.0,
        "cause_confusion": dict(
            sorted(
                Counter(f"{r.gt_cause_top1} -> {r.pred_cause_top1 or 'MISSING'}" for r in rows).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }
    return rows, summary


def write_outputs(rows: list[EvalRow], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "agent_only_eval_vs_ground_truth.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "log_name",
                "gt_status",
                "gt_cause_top1",
                "gt_action_top1",
                "pred_status",
                "pred_cause_top1",
                "pred_action_top1",
                "status_match",
                "cause_top1_match",
                "action_top1_contains",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.log_name,
                    r.gt_status,
                    r.gt_cause_top1,
                    r.gt_action_top1,
                    r.pred_status,
                    r.pred_cause_top1,
                    r.pred_action_top1,
                    r.status_match,
                    r.cause_top1_match,
                    r.action_top1_contains,
                ]
            )

    json_path = out_dir / "agent_only_eval_vs_ground_truth_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / "agent_only_eval_vs_ground_truth.md"
    lines = [
        "# Agent Only vs Ground Truth",
        "",
        f"- total: {summary['total']}",
        f"- covered: {summary['covered']} ({summary['coverage']:.2%})",
        f"- status_accuracy: {summary['status_accuracy']:.2%}",
        f"- cause_top1_accuracy: {summary['cause_top1_accuracy']:.2%}",
        f"- action_top1_contains_accuracy: {summary['action_top1_contains_accuracy']:.2%}",
        "",
        "## Top confusion",
    ]
    for k, v in list(summary["cause_confusion"].items())[:15]:
        lines.append(f"- {k}: {v}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default="C:/Users/yjs/Desktop/JAN/comparison_eval/ground_truth.csv")
    ap.add_argument("--runs-dir", default="C:/Users/yjs/Desktop/JAN/comparison_eval/runs/agent_only")
    ap.add_argument("--out-dir", default="out/eval")
    args = ap.parse_args()

    rows, summary = evaluate(Path(args.ground_truth), Path(args.runs_dir))
    write_outputs(rows, summary, Path(args.out_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
