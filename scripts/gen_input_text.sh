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

# meta prompt
prompt_file=data/prompts/gen_input_text.txt

# icd-10 file to provide icd-10 codes and descriptions
# context_file=data/icd10/icd10_by_block.jsonl
context_file=data/icd10/icd10_by_category.jsonl

# icd-10 code frequencies to sample
code_freq_file=data/quaero_icd10_by_category/code_freqs.jsonl

outdir=data/synthetic

model_path=/projects/bhuang/models/llm/pretrained/mistralai/Mistral-Large-Instruct-2407
# model_path=/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8
# model_path=/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
# model_path=aaditya/Llama3-OpenBioLLM-70B

pretty_model_path=${model_path##*/}
pretty_model_path=${pretty_model_path//-/_}
pretty_model_path=${pretty_model_path,,}

timestamp=$(date +%y%m%d)

output_file=${outdir}/synthetic-${pretty_model_path}-${timestamp}.jsonl

# generate using LLMs
python scripts/generate_vllm.py gen_input_text \
    --prompt_file $prompt_file \
    --context_file $context_file \
    --code_freq_file $code_freq_file \
    --output_file $output_file \
    --output_column_name text \
    --model_name_or_path $model_path \
    --dtype bfloat16 \
    --tensor_parallel_size $tp_size \
    --gpu_memory_utilization 0.95 \
    --max_num_seqs 256 \
    --max_model_len 4096 \
    --max_tokens 1024 \
    --temperature 0.7 \
    --top_p 1.0 \
    --num_examples 30000 \
    --checkpoint_every 1024

# postprocess generations
python scripts/postprocess_input_text.py \
    --input_file $output_file \
    --valid_size 1000 \
    --test_size 1000

echo "END TIME: $(date)"
