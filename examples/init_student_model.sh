#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

set -e

echo "START TIME: $(date)"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# cuda
export CUDA_VISIBLE_DEVICES=""

# model_name_or_path=outputs/quaero/drbert_7gb_ft_ep100_bs32_lr5e5_cosine_ntr-fl
model_name_or_path=outputs/synthetic/drbert_7gb_ft_ep50_bs64_lr5e5_cosine_ntr-fl

layer_selection_strategy=max_spaced
# layer_selection_strategy=first
# layer_selection_strategy=last
encoder_layers=2

python init_student_model.py \
    --model_name_or_path $model_name_or_path \
    --output_dir $model_name_or_path-${layer_selection_strategy}${encoder_layers}-init \
    --encoder_layers $encoder_layers \
    --layer_selection_strategy $layer_selection_strategy

echo "END TIME: $(date)"
