#!/usr/bin/env bash
# Evaluate source-view segmentation across GPU shards.
#   bash scripts/run_source_val_4gpu.sh
#   DEVICES=4,5,6,7 bash scripts/run_source_val_4gpu.sh configs/scannet_panst3r.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/scannet_panst3r.yaml}"; [ $# -gt 0 ] && shift
PYTHON="${PYTHON:-python}"
IFS=',' read -ra DEVS <<< "${DEVICES:-0,1,2,3}"
NS=${#DEVS[@]}
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}":"$("$PYTHON" -c 'import torch,os;print(os.path.dirname(torch.__file__)+"/lib")' 2>/dev/null)"

OUT="runs/source_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
echo "config=$CONFIG  gpus=${DEVS[*]}  out=$OUT"

pids=()
for i in $(seq 0 $((NS-1))); do
    CUDA_VISIBLE_DEVICES=${DEVS[$i]} PYTHONUNBUFFERED=1 \
    "$PYTHON" -u experiments/eval_source_only.py --config "$CONFIG" \
        --shard-idx "$i" --num-shards "$NS" "$@" > "$OUT/shard_$i.log" 2>&1 &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

for i in $(seq 0 $((NS-1))); do
    echo "== shard $i =="; grep "^METRIC" "$OUT/shard_$i.log" || true
done
