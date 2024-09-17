# coding=utf-8
# Copyright 2024  Bofeng Huang

import json
from typing import Optional

import fire
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from transformers import AutoConfig

disable_progress_bar()


def evaluate(y, preds, average="micro", verbose=False):
    """evaluate on all metrics"""
    precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average=average, zero_division=1)
    auc_score = roc_auc_score(y, preds, average=average)

    if verbose:
        print(f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}, auc_score: {auc_score:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1, "auc_score": auc_score}


def main(
    input_file: str,
    output_file: str,
    model_name_or_path: str,
    reference_column_name: str = "labels",
    hypothesis_column_name: str = "pred_probs",
    threshold: float = 0.5,
    grid_search_threshold: bool = False,
    num_processing_workers: int = 16,
    max_samples: Optional[int] = None,
):
    # 1. load data
    input_file = input_file.split("+")
    dataset = load_dataset("json", data_files=input_file, split="train")
    # print(f"Loaded {dataset.num_rows:,d} examples")

    # debug
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    # 2. load label&label_id
    config = AutoConfig.from_pretrained(model_name_or_path)
    label_2_id = config.label2id
    num_classes = len(label_2_id)

    encoded_reference_column_name = f"encoded_{reference_column_name}"
    binarized_hypothesis_column_name = f"binarized_{hypothesis_column_name}"

    # 3. encode labels
    def _encode(label):
        encoded_label = np.zeros(num_classes, dtype=np.int64)
        encoded_label[[label_2_id[l] for l in label]] = 1
        return encoded_label

    dataset = dataset.map(
        lambda example: {encoded_reference_column_name: _encode(example[reference_column_name])},
        num_proc=num_processing_workers,
    )

    # 4. binarize output probs using threshold
    def _binarize(example, threshold):
        pred = np.asarray(example[hypothesis_column_name])
        pred = np.where(pred >= threshold, 1, 0)
        example[binarized_hypothesis_column_name] = pred
        return example

    # 5. eval
    def binarize_and_evaluate(ds: Dataset, thr: float):
        processed_ds = ds.map(_binarize, fn_kwargs={"threshold": thr}, num_proc=num_processing_workers)
        res = evaluate(processed_ds[encoded_reference_column_name], processed_ds[binarized_hypothesis_column_name])
        return res

    if grid_search_threshold:
        res_by_thr = []
        for thr in tqdm(np.arange(0.01, 1, 0.05), desc="Grid searching binarization threshold..."):
            r = binarize_and_evaluate(dataset, thr)
            res_by_thr.append({"threshold": thr, **r})

        df_res_by_thr = pd.DataFrame(res_by_thr)
        # sort by f1
        df_res_by_thr = df_res_by_thr.sort_values("f1", ascending=False)
        print(df_res_by_thr.head())
        print(f'\nBest threshold: {df_res_by_thr.iloc[0]["threshold"]:.4f}\n')

        result = df_res_by_thr.iloc[0].to_dict()
    else:
        result = binarize_and_evaluate(dataset, threshold)
        result["threshold"] = threshold

    # save
    with open(output_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)
