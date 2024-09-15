#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

set -e

echo "START TIME: $(date)"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
# tp
# tp_size=4
tp_size=8

# meta prompt for icd-10 label evaluation
prompt_file=data/prompts/evaluate_icd10.txt

# icd-10 file to retrieve code descripiton
# context_file=data/icd10/icd10_by_block.jsonl
context_file=data/icd10/icd10_by_category.jsonl

model_path=/projects/bhuang/models/llm/pretrained/mistralai/Mistral-Large-Instruct-2407
# model_path=/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8
# model_path=/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
# model_path=aaditya/Llama3-OpenBioLLM-70B

eval_icd10_labels () {
    input_file=$1
    # model_path=$2

    pretty_model_path=${model_path##*/}
    pretty_model_path=${pretty_model_path//-/_}
    pretty_model_path=${pretty_model_path,,}

    output_file=${input_file%.*}-eval-${pretty_model_path}.jsonl

        # --max_samples 128 \
        # --max_num_seqs 32 \
        # --max_model_len 32000 \

    python scripts/generate_vllm.py eval_label \
        --input_file $input_file \
        --prompt_file $prompt_file \
        --context_file $context_file \
        --output_file $output_file \
        --text_column_name text \
        --model_name_or_path $model_path \
        --dtype bfloat16 \
        --tensor_parallel_size $tp_size \
        --gpu_memory_utilization 0.95 \
        --max_model_len 8192 \
        --max_tokens 2048 \
        --temperature 0 \
        --top_p 1.0 \
        --checkpoint_every 1024
}

# eval_icd10_labels data/quaero_icd10_by_block/drbenchmark_quaero-medline-validation-cls-mistral_large_instruct_2407-processed.jsonl
# eval_icd10_labels data/quaero_icd10_by_block/drbenchmark_quaero-medline-validation-cls-meta_llama_3.1_405b_instruct_fp8-processed.jsonl
# eval_icd10_labels data/quaero_icd10_by_block/drbenchmark_quaero-medline-validation-cls-llama3_openbiollm_70b-processed.jsonl

echo "END TIME: $(date)"
