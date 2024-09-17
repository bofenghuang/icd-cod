#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

set -e

echo "START TIME: $(date)"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# cuda
# export CUDA_VISIBLE_DEVICES="3"
export CUDA_VISIBLE_DEVICES=""

input_file=data/samples_by_length.json
output_file=outputs/eval/inference_time.jsonl

num_threads=1
signature=cpu_${num_threads}

# num_threads=-1
# signature=gpu_h100

cmd=""
# cmd="taskset -c 0-15"

model_paths=(
    outputs/synthetic/drbert_7gb_ft_ep50_bs64_lr5e5_cosine_ntr-fl
    outputs/synthetic/drbert_7gb_large_ft_ep50_bs64_lr5e5_cosine_ntr-fl
    outputs/synthetic/drbert_7gb_ft_ep50_bs64_lr5e5_cosine_ntr-fl-max_spaced6-init-distil-drbert_7gb_large-ep80-bs64-lr5e5-cosine-ntr-temp05
    outputs/synthetic/drbert_7gb_ft_ep50_bs64_lr5e5_cosine_ntr-fl-max_spaced4-init-distil-drbert_7gb_large-ep80-bs64-lr5e5-cosine-ntr-temp05
    outputs/synthetic/drbert_7gb_ft_ep50_bs64_lr5e5_cosine_ntr-fl-max_spaced2-init-distil-drbert_7gb_large-ep80-bs64-lr5e5-cosine-ntr-temp05
)

for model_path in "${model_paths[@]}"; do
    $cmd python bench_infer_time.py \
        --model_type hf \
        --model_name_or_path $model_path \
        --input_file $input_file \
        --output_file $output_file \
        --num_threads $num_threads \
        --signature $signature
done

# model_name_or_path=outputs/tfidf_lr/synthetic/model_tfidf_lr.pkl

# $cmd python bench_infer_time.py \
#     --model_type sklearn \
#     --model_name_or_path $model_name_or_path \
#     --input_file $input_file \
#     --output_file $output_file \
#     --num_threads $num_threads \
#     --signature $signature

# model_name_or_path=outputs/fasttext/synthetic/model_fasttext.bin

# $cmd python bench_infer_time.py \
#     --model_type fasttext \
#     --model_name_or_path $model_name_or_path \
#     --input_file $input_file \
#     --output_file $output_file \
#     --num_threads $num_threads \
#     --signature $signature


echo "END TIME: $(date)"

