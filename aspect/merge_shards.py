import argparse
import glob
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="aspect")
    args = ap.parse_args()
    want = [json.loads(l)["item_id"] for l in open(args.bench)]
    files = sorted(glob.glob(f"{args.out}/{args.tag}.s*_raw.jsonl")) or [f"{args.out}/{args.tag}_raw.jsonl"]
    rows = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["item_id"] in rows:
                raise SystemExit(f"duplicate item_id {r['item_id']} in {f}")
            rows[r["item_id"]] = r
    missing = [k for k in want if k not in rows]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(want)} questions have no prediction, e.g. {missing[:5]}")
    path = Path(args.out) / f"{args.tag}_raw.jsonl"
    with open(path, "w") as f:
        for k in want:
            f.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    print(f"{len(want)} predictions -> {path}")


if __name__ == "__main__":
    main()
