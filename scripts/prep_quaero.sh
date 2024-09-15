#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

set -e

echo "START TIME: $(date)"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"

python scripts/prep_quaero.py --dataset_name DrBenchmark/QUAERO --dataset_config medline --output_dir data/quaero
python scripts/prep_quaero.py --dataset_name DrBenchmark/QUAERO --dataset_config emea --output_dir data/quaero

echo "END TIME: $(date)"
