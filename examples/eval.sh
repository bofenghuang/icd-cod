#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

set -e

echo "START TIME: $(date)"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# cuda
export CUDA_VISIBLE_DEVICES="7"

model_name_or_path=$1

outdir=outputs/eval/quaero

output_dir=${outdir}/${model_name_or_path##*/}
mkdir -p $output_dir

valid_file=(
    data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-medline-validation-cls-mistral_large_instruct_2407-processed.jsonl
    data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-emea-validation-cls-mistral_large_instruct_2407-processed.jsonl
)
valid_pretty_name=quaero-valid

test_files=(
    data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-medline-test-cls-mistral_large_instruct_2407-processed.jsonl
    data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-emea-test-cls-mistral_large_instruct_2407-processed.jsonl
)

valid_file=$(IFS=+; echo "${valid_file[*]}")

echo -e "\nRunning inference..."
python infer.py \
    --model_name_or_path $model_name_or_path \
    --input_file $valid_file \
    --output_file ${output_dir}/${valid_pretty_name}-pred.jsonl \
    --batch_size 64

echo -e "\nRunning evaluation..."
python eval.py \
    --model_name_or_path $model_name_or_path \
    --input_file ${output_dir}/${valid_pretty_name}-pred.jsonl \
    --output_file ${output_dir}/${valid_pretty_name}-pred-eval.json \
    --threshold 0.5 \
    --grid_search_threshold True

# get threshold
threshold=$(grep -m 1 '"threshold":' ${output_dir}/${valid_pretty_name}-pred-eval.json | cut -d ':' -f2 | tr -d ',' | xargs)
# echo "Use threshold $threshold for further processing"

# eval test set individually
for test_file in "${test_files[@]}"; do
    pretty_filename=${test_file##*/}
    pretty_filename=${pretty_filename%.*}

    echo -e "\nRunning inference..."
    python infer.py \
        --model_name_or_path $model_name_or_path \
        --input_file $test_file \
        --output_file ${output_dir}/${pretty_filename}-pred.jsonl \
        --batch_size 64

    echo -e "\nRunning evaluation..."
    python eval.py \
        --model_name_or_path $model_name_or_path \
        --input_file ${output_dir}/${pretty_filename}-pred.jsonl \
        --output_file ${output_dir}/${pretty_filename}-pred-eval.json \
        --threshold $threshold

done


echo "END TIME: $(date)"
