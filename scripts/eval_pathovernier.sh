#!/usr/bin/env bash
set -euo pipefail
MODEL=${MODEL:-Mikezcy/ASPECT-8B}
BENCH=${BENCH:?path to pathovernier.jsonl}
IMAGES=${IMAGES:?directory written by reconstruct_images.py}
NS=${NUM_SHARDS:-1}
OUT=${OUT:-outputs}
pids=()
for i in $(seq 0 $((NS - 1))); do
  CUDA_VISIBLE_DEVICES=$i python -m aspect.eval_pathovernier --model "$MODEL" --bench "$BENCH" \
      --images "$IMAGES" --out "$OUT" --num-shards "$NS" --shard "$i" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
python -m aspect.merge_shards --bench "$BENCH" --out "$OUT"
python -m aspect.score_pathovernier "$OUT/aspect_raw.jsonl" --bench "$BENCH"
