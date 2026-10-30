#!/bin/bash
# The full experiment on one 24 GB GPU (about 7 hours): GRPO and collision RL, seeds 0 and 1, then MMStar and MMMU-Pro
# evaluation of the base model and the four adapters, then results/report.md. Finished runs and evaluations are skipped.
set -e; cd "$(dirname "$0")"
[ -f data/pool.jsonl ] || python data.py
./serve.sh train
for arm in grpo collision; do
  for seed in 0 1; do
    tag=$arm; [ "$seed" = 0 ] || tag=${arm}_s$seed
    [ -f "runs/$tag/final/adapter_config.json" ] || python train.py --arm "$arm" --seed "$seed" --out "runs/$tag"
  done
done
./serve.sh eval
for s in mmstar mmmu; do
  python evaluate.py run --tag base --set "$s"
  for tag in grpo grpo_s1 collision collision_s1; do python evaluate.py run --tag "$tag" --set "$s" --adapter "runs/$tag/final"; done
done
./serve.sh stop
python evaluate.py report
