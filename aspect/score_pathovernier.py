import argparse
import json

import numpy as np

from .scoring.metrics import decompose
from .scoring.task_rules import consistency_mode


def option_map(gq, opts):
    return {k: o for k in gq for o in (opts or []) if k == o or k.endswith("_" + o)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--min-coverage", type=float, default=0.30)
    args = ap.parse_args()
    gold = {}
    for line in open(args.bench):
        r = json.loads(line)
        gold[r["item_id"]] = r
    pred = {}
    for line in open(args.predictions):
        r = json.loads(line)
        if r.get("item_id") in gold:
            pred[r["item_id"]] = r
    n = len(gold)
    acc, S, ca, con, rawr = [], [], [], [], []
    n_complete = 0
    for iid, g in gold.items():
        r = pred.get(iid) or {}
        gq = {k: float(v) for k, v in g["quantities"].items() if not str(k).startswith("_")}
        d = decompose(r.get("raw") or "", gq, g["skill"], g["answer"], options=g["options"],
                      opt_map=option_map(gq, g["options"]), cons_mode=consistency_mode(g),
                      question=g["question"], pred_override=r.get("pred"))
        correct = bool(r.get("correct"))
        acc.append(correct)
        S.append(d["S"])
        n_complete += d["matched"]
        if d["B"] is not None:
            ca.append(d["B"])
        if d["D"] is not None:
            con.append(d["D"])
        if correct and d["matched"]:
            rawr.append(d["S"])
    coverage = len(ca) / n
    res = {
        "n": n, "n_missing_predictions": n - len(pred),
        "Acc": float(np.mean(acc)),
        "CA": float(np.mean(ca)) if ca else None, "CA_coverage": f"{len(ca)}/{n}",
        "RAWR": 1.0 - float(np.mean(rawr)) if rawr else None, "N_RAWR": len(rawr),
        "complete_count_coverage": f"{n_complete}/{n}",
        "Count_Acc": float(np.mean(S)),
        "Consistency": float(np.mean(con)) if con else None,
        "CA_RAWR_reported": coverage >= args.min_coverage,
    }
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
