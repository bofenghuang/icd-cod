# coding=utf-8
# Copyright 2024  Bofeng Huang

import os
from typing import Optional

import fire
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(
    model_name_or_path: str,
    input_file: str,
    output_file: str,
    text_column_name: str = "text",
    output_column_name: str = "pred_probs",
    torch_dtype: str = "float32",
    batch_size: int = 32,
    max_samples: Optional[int] = None,
):
    # 1. load model and tokenizer
    # print("Loading model and tokenizer...")

    if torch_dtype == "float16":
        torch_dtype = torch.float16
    elif torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        # low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        # attn_implementation=attn_implementation,
    )

    # eval mode
    model.eval()
    # move to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 2. load data
    # print("Loading data...")
    input_file = input_file.split("+")
    dataset = load_dataset("json", data_files=input_file, split="train")
    # print(f"Loaded {dataset.num_rows:,d} examples")

    # debug
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    # 3. infer
    # print("Inferring...")

    def process_and_infer(examples):
        # tokenize with padding to max length in batch
        inputs = tokenizer(examples[text_column_name], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)

        return {output_column_name: probabilities.cpu().numpy().tolist()}

    processed_dataset = dataset.map(
        process_and_infer,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
        # remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Inferring..."
    )

    # 4. export
    # print("Saving results...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    # print(f"The processed data is saved into {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
