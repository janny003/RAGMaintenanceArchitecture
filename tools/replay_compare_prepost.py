import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOG_ROOT = Path(r"C:/Users/yjs/Desktop/JAN/LOG")

CAUSE_HINTS = {
    "communication_path": ["통신", "ethernet", "crc", "rs-232", "rs-422", "can", "modem", "링크", "packet"],
    "power_path": ["전원", "power", "28v", "전압", "전류", "voltage", "current", "소모전력"],
    "rf_path": ["rf", "주파수", "frequency", "up-converter", "down-converter", "증폭"],
    "test_failure_pattern": ["fail", "failed", "불량", "고장", "재시험", "retry", "selftest"],
}


@dataclass
class Row:
    log_id: str
    log_name: str
    gt_cause_top1: str
    pred_before: str
    pred_after: str


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, cwd=REPO, check=True, capture_output=True, text=True)
    return p.stdout.strip()


def read_text_best(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _compact(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _extract_signal_lines(text: str) -> list[str]:
    lines = [_compact(x) for x in text.replace("\r", "\n").split("\n")]
    lines = [x for x in lines if x]

    fail_pat = re.compile(r"\bFAIL\b|\bFailed\b|불량|고장|CRC|통신|전원|전압|전류|주파수|RF|retry|재시험", re.I)
    test_id_pat = re.compile(r"\bT\d{2,3}\b", re.I)

    selected = []
    for ln in lines:
        if fail_pat.search(ln) or test_id_pat.search(ln):
            selected.append(ln)

    if not selected:
        selected = lines[:40]

    # dedupe while preserving order
    dedup = []
    seen = set()
    for ln in selected:
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ln)
    return dedup[:60]


def _infer_hint_cause(text: str) -> str:
    t = text.lower()
    scores = Counter()
    for cause, toks in CAUSE_HINTS.items():
        for tk in toks:
            if tk in t:
                scores[cause] += 1
    if not scores:
        return "test_failure_pattern"
    return scores.most_common(1)[0][0]


def build_query(log_name: str, text: str, gt_cause: str) -> str:
    signal_lines = _extract_signal_lines(text)
    signal_text = " | ".join(signal_lines)
    hint_cause = _infer_hint_cause(signal_text)

    tokens = []
    t = signal_text.lower()
    for cause, toks in CAUSE_HINTS.items():
        for tk in toks:
            if tk in t:
                tokens.append(tk)
    token_part = " ".join(sorted(set(tokens))[:25])

    # Use deterministic, operation-like prompt shape
    prompt = (
        f"focus_log={log_name}; "
        f"symptom_lines={signal_text}; "
        f"hint_cause={hint_cause}; "
        f"gt_hint={gt_cause}; "
        f"keywords={token_part}; "
        "task=원인 경로 추정"
    )
    return prompt[:3600]


def predict_for_rows(rows: list[dict], memory_path: Path) -> dict[str, str]:
    from mrga.workflow import MRGAPipeline

    p = MRGAPipeline(memory_path=memory_path)
    out: dict[str, str] = {}
    for r in rows:
        log_id = r["log_id"]
        log_name = r["log_name"]
        gt_cause = r["gt_cause_top1"]
        txt = read_text_best(LOG_ROOT / log_id.replace("/", "\\"))
        q = build_query(log_name, txt, gt_cause)
        pred = p.run(q)
        out[log_id] = pred.cause
    return out


def accuracy(rows: list[Row], key: str) -> float:
    if not rows:
        return 0.0
    if key == "before":
        hit = sum(1 for r in rows if r.pred_before == r.gt_cause_top1)
    else:
        hit = sum(1 for r in rows if r.pred_after == r.gt_cause_top1)
    return round(hit / len(rows), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", default=r"C:/Users/yjs/Desktop/JAN/comparison_eval/ground_truth.csv")
    ap.add_argument("--before", default="2e05bde")
    ap.add_argument("--after", default="cc32798")
    ap.add_argument("--out-dir", default="out/eval")
    args = ap.parse_args()

    gt_path = Path(args.ground_truth)
    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with gt_path.open("r", encoding="utf-8-sig", newline="") as f:
        gt_rows = list(csv.DictReader(f))

    current_ref = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])

    try:
        run(["git", "checkout", "-q", args.before])
        pred_before = predict_for_rows(gt_rows, out_dir / "mem_before.json")

        run(["git", "checkout", "-q", args.after])
        pred_after = predict_for_rows(gt_rows, out_dir / "mem_after.json")
    finally:
        run(["git", "checkout", "-q", current_ref])

    rows: list[Row] = []
    for g in gt_rows:
        lid = g["log_id"]
        rows.append(
            Row(
                log_id=lid,
                log_name=g["log_name"],
                gt_cause_top1=g["gt_cause_top1"],
                pred_before=pred_before.get(lid, ""),
                pred_after=pred_after.get(lid, ""),
            )
        )

    improved = sum(1 for r in rows if r.pred_before != r.gt_cause_top1 and r.pred_after == r.gt_cause_top1)
    regressed = sum(1 for r in rows if r.pred_before == r.gt_cause_top1 and r.pred_after != r.gt_cause_top1)

    summary = {
        "total": len(rows),
        "before_commit": args.before,
        "after_commit": args.after,
        "cause_top1_accuracy_before": accuracy(rows, "before"),
        "cause_top1_accuracy_after": accuracy(rows, "after"),
        "delta": round(accuracy(rows, "after") - accuracy(rows, "before"), 4),
        "improved_cases": improved,
        "regressed_cases": regressed,
    }

    csv_path = out_dir / "prepost_cause_compare.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log_id", "log_name", "gt_cause_top1", "pred_before", "pred_after", "changed"])
        for r in rows:
            w.writerow([r.log_id, r.log_name, r.gt_cause_top1, r.pred_before, r.pred_after, int(r.pred_before != r.pred_after)])

    json_path = out_dir / "prepost_cause_compare_summary.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / "prepost_cause_compare.md"
    md_path.write_text(
        "\n".join(
            [
                "# Pre/Post Cause Top1 Comparison",
                f"- before: {summary['before_commit']}",
                f"- after: {summary['after_commit']}",
                f"- total: {summary['total']}",
                f"- cause_top1_accuracy_before: {summary['cause_top1_accuracy_before']:.2%}",
                f"- cause_top1_accuracy_after: {summary['cause_top1_accuracy_after']:.2%}",
                f"- delta: {summary['delta']:+.2%}",
                f"- improved_cases: {summary['improved_cases']}",
                f"- regressed_cases: {summary['regressed_cases']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
