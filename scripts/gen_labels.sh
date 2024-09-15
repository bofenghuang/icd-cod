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

# meta prompt for generate icd-10 labels
# prompt_file=data/prompts/classify_icd10_block.txt
prompt_file=data/prompts/classify_icd10_category.txt

# icd-10 file to provide icd-10 codes and descriptions
# context_file=data/icd10/icd10_by_block.jsonl
context_file=data/icd10/icd10_by_category.jsonl

suffix=${context_file##*/}
suffix=_${suffix%.*}

model_path=/projects/bhuang/models/llm/pretrained/mistralai/Mistral-Large-Instruct-2407
# model_path=/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8
# model_path=/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
# model_path=aaditya/Llama3-OpenBioLLM-70B

gen_icd10_labels () {
    input_file=$1
    # model_path=$2

    pretty_model_path=${model_path##*/}
    pretty_model_path=${pretty_model_path//-/_}
    pretty_model_path=${pretty_model_path,,}

    pretty_input_file=${input_file%.*}
    pretty_input_file=${pretty_input_file/\/quaero\//\/quaero${suffix}\/}

    output_file=${pretty_input_file}-cls-${pretty_model_path}.jsonl

        # --max_samples 128 \
        # --max_num_seqs 32 \
        # --max_model_len 32000 \
        # --max_num_seqs 256 \
        # --max_model_len 8192 \

    # generate labels using LLMs
    python scripts/generate_vllm.py gen_label \
        --input_file $input_file \
        --prompt_file $prompt_file \
        --context_file $context_file \
        --output_file $output_file \
        --text_column_name text \
        --model_name_or_path $model_path \
        --dtype bfloat16 \
        --tensor_parallel_size $tp_size \
        --gpu_memory_utilization 0.95 \
        --max_num_seqs 32 \
        --max_model_len 32000 \
        --max_tokens 2048 \
        --temperature 0 \
        --top_p 1.0 \
        --checkpoint_every 1024

    # postprocess LLM generation
    python scripts/postprocess_labels.py \
        --input_file $output_file \
        --context_file $context_file \
        --output_file ${output_file%.*}-processed.jsonl \
        --max_labels 20

}

gen_icd10_labels data/quaero/drbenchmark_quaero-medline-validation.jsonl
# gen_icd10_labels data/quaero/drbenchmark_quaero-medline-train.jsonl
# gen_icd10_labels data/quaero/drbenchmark_quaero-medline-test.jsonl

# gen_icd10_labels data/quaero/drbenchmark_quaero-emea-validation.jsonl
# gen_icd10_labels data/quaero/drbenchmark_quaero-emea-train.jsonl
# gen_icd10_labels data/quaero/drbenchmark_quaero-emea-test.jsonl

echo "END TIME: $(date)"
