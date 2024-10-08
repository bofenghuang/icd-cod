#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang


from pathlib import Path

import fire
import numpy as np
from datasets import DatasetDict, load_dataset

""""Postprocess input text generated by LLM."""


def get_hash(example, column_name):
    """Get hash of content field."""
    return {"hash": hash(example[column_name])}


def check_uniques(example, uniques):
    """Check if current hash is still in set of unique hashes and remove if true."""
    if example["hash"] in uniques:
        uniques.remove(example["hash"])
        return True
    else:
        return False


def main(
    input_file: str,
    text_column_name: str = "text",
    valid_size: int = 1_000,
    test_size: int = 1_000,
    num_processing_workers: int = 16,
):
    # load dataset
    dataset = load_dataset("json", data_files=input_file, split="train")
    print(f"Loaded {dataset.num_rows:,d} examples from {input_file}")

    # fake has_diso to be consistent w/ others
    processed_dataset = dataset.map(
        lambda _: {"has_diso": True},
        remove_columns={"gen_configs"} & set(dataset.column_names),
        num_proc=num_processing_workers,
    )

    # exact dedup
    # todo: fuzzy dedup
    processed_dataset = processed_dataset.map(get_hash, fn_kwargs={"column_name": text_column_name})
    uniques = set(processed_dataset.unique("hash"))
    filtered_dataset = processed_dataset.filter(check_uniques, fn_kwargs={"uniques": uniques})
    filtered_dataset = filtered_dataset.remove_columns("hash")
    print(f"Deduped to {filtered_dataset.num_rows:,d} examples")

    # stat
    dataset_df = filtered_dataset.to_pandas()
    # dataset_df["num_labels"] = dataset_df["labels"].map(lambda x: len(x))
    all_labels = np.concatenate(dataset_df["labels"].to_list())
    unique, counts = np.unique(all_labels, return_counts=True)
    count_sort_indices = np.argsort(-counts)
    unique, counts = unique[count_sort_indices], counts[count_sort_indices]
    print(f"Num of unique labels: {len(unique)}")
    print("Top 5 most frequent labels")
    for i in range(5):
        print(f"{unique[i]:10}: {counts[i]}")

    p = Path(input_file)

    # save processed data
    output_file = f'{p.with_suffix("")}-processed{p.suffix}'
    filtered_dataset.to_json(output_file, orient="records", lines=True, force_ascii=False)
    print(f"The processed data is saved into {output_file}")

    # train/valid/test split
    # train_testvalid = filtered_dataset.train_test_split(test_size=valid_size + test_size)  # , stratify_by_column="labels")
    # test_valid = train_testvalid["test"].train_test_split(test_size=test_size)
    # splitted_dataset = DatasetDict(
    #     {
    #         "train": train_testvalid["train"],
    #         "validation": test_valid["train"],
    #         "test": test_valid["test"],
    #     }
    # )
    # print(splitted_dataset)

    splitted_dataset = filtered_dataset.train_test_split(test_size=1000)  # , stratify_by_column="labels")
    splitted_dataset["validation"] = splitted_dataset.pop("train")

    for name, ds in splitted_dataset.items():
        output_file = f'{p.with_suffix("")}-processed-{name}{p.suffix}'
        ds.to_json(output_file, orient="records", lines=True, force_ascii=False)
        print(f"The {name} split is saved into {output_file}")

    # ds = splitted_dataset["train"].train_test_split(test_size=10_000)["test"]
    # output_file = f'{p.with_suffix("")}-processed-train-10k{p.suffix}'
    # ds.to_json(output_file, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
