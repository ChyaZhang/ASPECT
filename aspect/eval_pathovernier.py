import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image

from .inference import generate, load
from .scoring.answer_parser import parse_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Mikezcy/ASPECT-8B")
    ap.add_argument("--bench", required=True, help="pathovernier.jsonl")
    ap.add_argument("--images", required=True, help="directory produced by reconstruct_images.py")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--tag", default="aspect")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    torch.manual_seed(42)

    rows = [json.loads(l) for l in open(args.bench)]
    rows.sort(key=lambda x: (x["skill"], x["item_id"]))
    rows = rows[args.shard::args.num_shards]
    model, processor = load(args.model)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f".s{args.shard}" if args.num_shards > 1 else ""
    path = out_dir / f"{args.tag}{sfx}_raw.jsonl"
    n_correct = 0
    with open(path, "w") as f:
        for i, g in enumerate(rows, 1):
            image = Image.open(Path(args.images) / g["dataset"] / f"{g['patch_id']}.png")
            question = re.split(r"\n?In <observe>", g["question"])[0].rstrip()
            raw = generate(model, processor, image, question, args.max_new_tokens)
            pred = parse_pred(raw, g["options"])
            n_correct += pred == g["answer"]
            f.write(json.dumps(dict(item_id=g["item_id"], skill=g["skill"], dataset=g["dataset"],
                                    gold=g["answer"], pred=pred, correct=pred == g["answer"],
                                    raw=raw), ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(rows)}  acc={n_correct / i:.3f}", flush=True)
    print(f"n={len(rows)}  acc={n_correct / max(len(rows), 1):.4f}  -> {path}")


if __name__ == "__main__":
    main()
