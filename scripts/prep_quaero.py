#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""
Prepare quaero dataset.

Usage: python scripts/prep_quaero.py --dataset_name DrBenchmark/QUAERO --dataset_config medline --output_dir data/quaero
"""

import os
import re

import fire
from datasets import load_dataset


def normalize_text(s):
    s = re.sub(r"[´′’ʼ‘ʻ`]", "'", s)  # standardize quotes and apostrophes
    s = re.sub(r"[−‐–—]", "-", s)  # standardize hyphens and dashes
    # standarize special characters (for french)
    s = re.sub(r"æ", "ae", s)
    s = re.sub(r"œ", "oe", s)
    s = re.sub(r"\s*'\s*", "'", s)  # remove space before/after apostrophe
    s = re.sub(r"\s+([,.])", r"\1", s)  # remove space before comma/period
    # s = re.sub(r"\(\s*(.+?)\s*\)", r"(\1)", s)  # remove spaces inside parentheses, non-greedy
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s*/\s*", "/", s)  # remove spaces around slash
    return s


def main(
    dataset_name: str = "DrBenchmark/QUAERO",
    dataset_config: str = "emea",
    # dataset_split: str = "train",
    output_dir: str = "data/quaero",
    num_processing_workers: int = 8,
):
    # load dataset
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        # split=dataset_split,
        trust_remote_code=True,
    )
    # print(f"Loaded {dataset.num_rows:,d} examples")
    print(dataset)

    # concatenate input text tokens and normalize
    # check if DISO entity exists
    def id_2_label(idx):
        # return dataset.features["ner_tags"].feature.int2str(idx)
        return dataset["train"].features["ner_tags"].feature.int2str(idx)

    def process_function(example):
        example["raw_text"] = " ".join(example["tokens"])
        example["text"] = normalize_text(example["raw_text"])
        example["has_diso"] = any("DISO" in id_2_label(x) for x in example["ner_tags"])
        return example

    processed_dataset = dataset.map(
        process_function,
        # remove_columns=dataset.column_names,
        remove_columns=dataset["train"].column_names,
        num_proc=num_processing_workers,
    )

    # filter out empty input text
    filtered_dataset = processed_dataset.filter(lambda x: x["text"], num_proc=num_processing_workers)
    print(filtered_dataset)

    # deduplicate input text
    # deduplicate validation against test
    filtered_dataset["validation"] = filtered_dataset["validation"].filter(
        lambda x: x["text"] not in set(filtered_dataset["test"]["text"]),
        num_proc=num_processing_workers,
    )
    # deduplicate train against validation+test
    filtered_dataset["train"] = filtered_dataset["train"].filter(
        lambda x: x["text"] not in set(filtered_dataset["validation"]["text"] + filtered_dataset["test"]["text"]),
        num_proc=num_processing_workers,
    )
    print(filtered_dataset)

    # export
    os.makedirs(output_dir, exist_ok=True)
    pretty_dataset_name = dataset_name.replace("/", "_").lower()
    pretty_dataset_config = dataset_config.lower()
    for split_name, dataset_split in filtered_dataset.items():
        output_file = os.path.join(output_dir, f"{pretty_dataset_name}-{pretty_dataset_config}-{split_name}.jsonl")
        dataset_split.to_json(output_file, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
