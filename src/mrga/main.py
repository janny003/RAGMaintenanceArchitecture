import argparse
import json
from dataclasses import asdict
from .workflow import MRGAPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--memory-path", default="out/persistent_memory.json")
    args = ap.parse_args()

    p = MRGAPipeline(memory_path=args.memory_path)
    out = p.run(args.query)
    payload = asdict(out)
    payload["evidence"] = [asdict(d) for d in out.evidence]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
