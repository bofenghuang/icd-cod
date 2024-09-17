#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

# Run classification

set -e

echo "START TIME: $(date)"

# Debugging flags (optional)
# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1
# export PYTHONFAULTHANDLER=1

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export CUDA_VISIBLE_DEVICES="7"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

# wandb
# export WANDB_MODE=offline
# export WANDB_DISABLED=true
# export WANDB_API_KEY=YOUR_WANDB_API_KEY
# export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_PROJECT=medical_v0.2

# Set your number of GPUs here
num_gpus=1

# cmd
# cmd="accelerate launch --multi_gpu --num_processes=$num_gpus"
# cmd="accelerate launch --num_processes=$num_gpus"
# cmd="accelerate launch --num_processes=$num_gpus --mixed_precision bf16"
cmd="accelerate launch --num_processes=$num_gpus --mixed_precision fp16"
# cmd="python"

# model_name_or_path=camembert-base
model_name_or_path=almanach/camembert-large
# model_name_or_path=almanach/camemberta-base
# model_name_or_path=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
# model_name_or_path=Dr-BERT/DrBERT-4GB
# model_name_or_path=Dr-BERT/DrBERT-7GB
# model_name_or_path=Dr-BERT/DrBERT-7GB-Large
# model_name_or_path=Dr-BERT/DrBERT-4GB-CP-PubMedBERT
# model_name_or_path=almanach/camembert-bio-base
# model_name_or_path=OrdalieTech/Solon-embeddings-base-0.1
# model_name_or_path=OrdalieTech/Solon-embeddings-large-0.1

# model_name_or_path=outputs/quaero/drbert_7gb_ft_ep100_bs32_lr5e5_cosine_ntr-fl-max_spaced2-init
# model_name_or_path=outputs/quaero/drbert_7gb_ft_ep100_bs32_lr5e5_cosine_ntr-fl-max_spaced4-init
# model_name_or_path=outputs/quaero/drbert_7gb_ft_ep100_bs32_lr5e5_cosine_ntr-fl-max_spaced6-init

pretty_model_name=${model_name_or_path##*/}
pretty_model_name=${pretty_model_name//-/_}
pretty_model_name=${pretty_model_name,,}

# train_files=data/tyqiangz-multilingual_sentiments/tyqiangz-multilingual_sentiments-train.jsonl
# validation_files=data/tyqiangz-multilingual_sentiments/tyqiangz-multilingual_sentiments-validation.jsonl

train_files=(
data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-medline-train-cls-mistral_large_instruct_2407-processed.jsonl
data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-emea-train-cls-mistral_large_instruct_2407-processed.jsonl
# data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-emea_medline-train-upsampled5.jsonl
# data/synthetic/synthetic-mistral_large_instruct_2407-240909-processed-train.jsonl
# data/synthetic/synthetic-mistral_large_instruct_2407-240909-processed-train-10k.jsonl
# data/synthetic/synthetic-mistral_large_instruct_2407-240913-processed-train.jsonl
)

validation_files=(
data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-medline-validation-cls-mistral_large_instruct_2407-processed.jsonl
data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-emea-validation-cls-mistral_large_instruct_2407-processed.jsonl
# data/synthetic/synthetic-mistral_large_instruct_2407-240909-processed-validation.jsonl
# data/synthetic_test/synthetic-head-processed-validation.jsonl
# data/synthetic_test/synthetic-medium-processed-validation.jsonl
# data/synthetic_test/synthetic-tail-processed-validation.jsonl
# data/synthetic/synthetic-mistral_large_instruct_2407-240913-processed-validation.jsonl
)

test_files=(
data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-medline-test-cls-mistral_large_instruct_2407-processed.jsonl
data/quaero_icd10_by_category_resplitted/drbenchmark_quaero-emea-test-cls-mistral_large_instruct_2407-processed.jsonl
# data/synthetic/synthetic-mistral_large_instruct_2407-240909-processed-test.jsonl
# data/synthetic/synthetic-mistral_large_instruct_2407-240913-processed-test.jsonl
)

# Use array expansion and join elements with a "+"
train_files=$(IFS=+; echo "${train_files[*]}")
validation_files=$(IFS=+; echo "${validation_files[*]}")
test_files=$(IFS=+; echo "${test_files[*]}")

# criterion_name=${1:-bce}
# criterion_name=bce
# criterion_name=fl
# criterion_name=ntr
criterion_name=ntr-fl

output_dir=outputs/quaero/${pretty_model_name}_ft_ep100_bs32_lr5e5_cosine_${criterion_name}

run_name=${output_dir##*/}

# --max_train_samples 32 \
# --max_eval_samples 32 \
# fp16
# --weight_decay 0.1 \
# --gradient_checkpointing \

$cmd train_cls.py \
    --model_name_or_path $model_name_or_path \
    --train_files $train_files \
    --validation_files $validation_files \
    --test_files $test_files \
    --text_column_names text \
    --label_column_name labels \
    --max_seq_length 512 \
    --pad_to_multiple_of 64 \
    --preprocessing_num_workers 16 \
    --shuffle_train_dataset \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --checkpointing_steps epoch \
    --save_total_limit 3 \
    --criterion_name $criterion_name \
    --num_train_epochs 100 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $run_name \
    --do_train \
    --do_eval

echo "END TIME: $(date)"
