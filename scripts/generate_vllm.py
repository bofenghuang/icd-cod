#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

import json
import os
import random
import re
import time
from typing import Optional

import fire
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def write_dataset_to_json(dataset, output_file, mode="w", encoding="utf-8", default=str, ensure_ascii=False):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, mode, encoding=encoding) as fo:
        for sample in dataset:
            fo.write(f"{json.dumps(sample, default=default, ensure_ascii=ensure_ascii)}\n")


def load_data(
    input_file: str,
    output_file: str,
    text_column_name: str,
    num_processing_workers: int,
    max_samples: Optional[int] = None,
):
    # load dataset
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    # take max samples
    if max_samples is not None:
        # data = data[:max_samples]
        dataset = dataset.select(range(max_samples))
        print(f"Sampled the first {dataset.num_rows:,d} examples")

    # remove examples already existing in output file
    if os.path.exists(output_file):
        existing_dataset = load_dataset("json", data_files=output_file, split="train")
        existing_values = existing_dataset.unique(text_column_name)
        existing_values = set(existing_values)
        print(f"Found {len(existing_values):,d} existing examples in {output_file}")

        dataset = dataset.filter(
            lambda x: x not in existing_values, input_columns=text_column_name, num_proc=num_processing_workers
        )
        print(f"Filtered to {dataset.num_rows:,d} examples")

    return dataset


def load_llm(
    model_name_or_path: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_model_len: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    # load llm
    llm = LLM(
        model=model_name_or_path,
        dtype=dtype,
        trust_remote_code=True,
        max_model_len=max_model_len,  # limited by kv-cache
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,  # activate Automatic Prefix Caching (APC) to reuse the KV cache if it shares the same prefix with one of the existing queries
    )
    # decoding params
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        # stop_token_ids=stop_token_ids,
    )
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    return llm, params, tokenizer


def process_batch(batch, llm, params, tokenizer, output_column_name="output"):
    user_instructions = [item.pop("pre_query_str") for item in batch]

    prompts = []
    for instruction in user_instructions:
        chat = [{"role": "user", "content": instruction}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        item[output_column_name] = outputs[i].outputs[0].text.strip()
        item["gen_configs"] = {
            "prompt": prompts[i],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "repetition_penalty": params.repetition_penalty,
            "max_tokens": params.max_tokens,
            # "stop_tokens": params.stop_tokens,
            "model": llm.llm_engine.model_config.model,
        }

    return batch


def generate_and_update(
    dataset: Dataset,
    output_file: str,
    # llm
    model_name_or_path: str,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    max_model_len: int,
    # llm decoding params
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    # others
    output_column_name: str,
    checkpoint_every: int,
):
    # load llm
    llm, params, tokenizer = load_llm(
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    start_time = time.perf_counter()

    num_batches = (len(dataset) + checkpoint_every - 1) // checkpoint_every

    for i in tqdm(range(num_batches)):
        start_idx = i * checkpoint_every
        end_idx = min((i + 1) * checkpoint_every, len(dataset))
        batch = dataset.select(range(start_idx, end_idx)).to_list()

        batch = process_batch(
            batch=batch,
            llm=llm,
            params=params,
            tokenizer=tokenizer,
            output_column_name=output_column_name,
        )

        # append batch result to output file
        write_dataset_to_json(batch, output_file, mode="a")

    print(
        f"Generation completed in {time.strftime('%Hh%Mm%Ss', time.gmtime(time.perf_counter() - start_time))}.\n"
        f"The generated data is saved into {output_file}"
    )


def gen_label(
    # data
    input_file: str,
    output_file: str,
    prompt_file: str,
    context_file: str,
    # llm
    model_name_or_path: str,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    max_num_seqs: int = 256,
    max_model_len: int = 4096,
    # llm decoding params
    max_tokens: int = 4096,
    temperature: float = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    # data column
    text_column_name: str = "text",
    output_column_name: str = "output",
    # others
    checkpoint_every: int = 1024,
    num_processing_workers: int = 16,
    max_samples: Optional[int] = None,
):
    """Predict icd-10 labels using LLM."""

    # load data
    dataset = load_data(
        input_file=input_file,
        output_file=output_file,
        text_column_name=text_column_name,
        num_processing_workers=num_processing_workers,
        max_samples=max_samples,
    )

    # wrap instruction by higher-level prompts
    if prompt_file is not None:
        with open(prompt_file, encoding="utf-8") as f:
            prompt_template = f.read()

    # get icd-10 codes and descriptions
    context_ds = load_dataset("json", data_files=context_file, split="train")
    context = "\n".join([example["code"] + ": " + example["description"] for example in context_ds])

    # build prompt
    dataset = dataset.map(
        lambda x: {"pre_query_str": prompt_template.format(context=context, input_text=x[text_column_name])},
        num_proc=num_processing_workers,
    )

    # main run
    generate_and_update(
        dataset=dataset,
        output_file=output_file,
        # llm
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        # llm decoding params
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        # others
        output_column_name=output_column_name,
        checkpoint_every=checkpoint_every,
    )


def eval_label(
    # data
    input_file: str,
    output_file: str,
    prompt_file: str,
    context_file: str,
    # llm
    model_name_or_path: str,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    max_num_seqs: int = 256,
    max_model_len: int = 4096,
    # llm decoding params
    max_tokens: int = 4096,
    temperature: float = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    # data column
    text_column_name: str = "text",
    output_column_name: str = "output",
    # others
    checkpoint_every: int = 1024,
    num_processing_workers: int = 16,
    max_samples: Optional[int] = None,
):
    """Evaluate predicted icd10 labels using LLM."""

    # load data
    dataset = load_data(
        input_file=input_file,
        output_file=output_file,
        text_column_name=text_column_name,
        num_processing_workers=num_processing_workers,
        max_samples=max_samples,
    )

    # wrap instruction by higher-level prompts
    if prompt_file is not None:
        with open(prompt_file, encoding="utf-8") as f:
            prompt_template = f.read()

    # dict to retrieve code descripiton
    context_ds = load_dataset("json", data_files=context_file, split="train")
    code_2_description = dict(zip(context_ds["code"], context_ds["description"]))

    # build prompt
    dataset = dataset.map(
        lambda x: {
            "pre_query_str": prompt_template.format(
                input_text=x[text_column_name], labels="\n".join(f"{l}: {code_2_description[l]}" for l in x["labels"])
            )
        },
        num_proc=num_processing_workers,
    )

    # main run
    generate_and_update(
        dataset=dataset,
        output_file=output_file,
        # llm
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        # llm decoding params
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        # others
        output_column_name=output_column_name,
        checkpoint_every=checkpoint_every,
    )


def gen_input_text(
    # data
    prompt_file: str,
    context_file: str,
    code_freq_file: str,
    output_file: str,
    # llm
    model_name_or_path: str,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.95,
    max_num_seqs: int = 256,
    max_model_len: int = 4096,
    # llm decoding params
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    # data column
    output_column_name: str = "output",
    # others
    num_examples: int = 16,
    checkpoint_every: int = 1024,
    num_processing_workers: int = 16,
):
    """Evaluate predicted icd10 labels using LLM."""

    # wrap instruction by higher-level prompts
    if prompt_file is not None:
        with open(prompt_file, encoding="utf-8") as f:
            prompt_template = f.read()

    # dict to retrieve code descripiton
    context_ds = load_dataset("json", data_files=context_file, split="train")
    code_2_description = dict(zip(context_ds["code"], context_ds["description"]))

    # code freq to guide random selection
    code_freq_ds = load_dataset("json", data_files=code_freq_file, split="train")
    codes, freqs = code_freq_ds["code"], code_freq_ds["freq"]
    # re-normalize after writing
    freqs = np.asarray(freqs)
    freqs = freqs / freqs.sum()

    dataset = Dataset.from_list([{"pre_query_str": prompt_template}] * num_examples)

    def process_function(example, num_criteria=3, max_labels=5):
        pre_query_str = example["pre_query_str"]

        # sample types
        m = re.search(r"<types>\n(.+)\n</types>", pre_query_str, flags=re.DOTALL)
        matched_str = m.group()
        criteria_list = m.groups()[0].split("\n")
        final_criteria = "\n".join(random.sample(criteria_list, num_criteria))
        pre_query_str = re.sub(re.escape(matched_str), final_criteria, pre_query_str)

        # sample styles
        m = re.search(r"<styles>\n(.+)\n</styles>", pre_query_str, flags=re.DOTALL)
        matched_str = m.group()
        criteria_list = m.groups()[0].split("\n")
        final_criteria = "\n".join(random.sample(criteria_list, num_criteria))
        pre_query_str = re.sub(re.escape(matched_str), final_criteria, pre_query_str)

        # sample 1-5 labels
        num_labels = random.choice(range(1, max_labels + 1))
        # selected_labels = np.random.choice(codes, size=num_labels, replace=False, p=freqs)
        # todo: uniform sampling
        selected_labels = np.random.choice(codes, size=num_labels, replace=False)
        example["labels"] = selected_labels
        example["pre_query_str"] = pre_query_str.format(
            labels="\n".join(f"{c}: {code_2_description[c]}" for c in selected_labels)
        )
        return example

    # build prompt
    dataset = dataset.map(process_function, num_proc=num_processing_workers)

    # main run
    generate_and_update(
        dataset=dataset,
        output_file=output_file,
        # llm
        model_name_or_path=model_name_or_path,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        # llm decoding params
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        # others
        output_column_name=output_column_name,
        checkpoint_every=checkpoint_every,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "gen_label": gen_label,
            "eval_label": eval_label,
            "gen_input_text": gen_input_text,
        }
    )
