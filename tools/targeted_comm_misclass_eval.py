import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare-csv", default="out/eval/prepost_cause_compare.csv")
    ap.add_argument("--out-dir", default="out/eval")
    ap.add_argument("--target-gt", default="communication_path")
    args = ap.parse_args()

    compare_csv = Path(args.compare_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(compare_csv.open("r", encoding="utf-8-sig", newline="")))
    target = [r for r in rows if r.get("gt_cause_top1") == args.target_gt]

    total = len(target)
    before_hit = sum(1 for r in target if r.get("pred_before") == args.target_gt)
    after_hit = sum(1 for r in target if r.get("pred_after") == args.target_gt)

    improved = [r for r in target if r.get("pred_before") != args.target_gt and r.get("pred_after") == args.target_gt]
    regressed = [r for r in target if r.get("pred_before") == args.target_gt and r.get("pred_after") != args.target_gt]

    summary = {
        "target_gt": args.target_gt,
        "total_target_cases": total,
        "before_recall": round(before_hit / total, 4) if total else 0.0,
        "after_recall": round(after_hit / total, 4) if total else 0.0,
        "delta": round((after_hit - before_hit) / total, 4) if total else 0.0,
        "improved_cases": len(improved),
        "regressed_cases": len(regressed),
    }

    (out_dir / "targeted_comm_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / "targeted_comm_eval_changed.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "log_id", "log_name", "gt", "pred_before", "pred_after"])
        for r in improved:
            w.writerow(["improved", r.get("log_id"), r.get("log_name"), r.get("gt_cause_top1"), r.get("pred_before"), r.get("pred_after")])
        for r in regressed:
            w.writerow(["regressed", r.get("log_id"), r.get("log_name"), r.get("gt_cause_top1"), r.get("pred_before"), r.get("pred_after")])

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
