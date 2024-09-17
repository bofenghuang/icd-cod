#!/usr/bin/env python
# Copyright 2024  Bofeng Huang

"""
Distilling models for multi-label text classification.

Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_classification.py
"""

import json
import logging
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import datasets
# import evaluate
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
from datasets import Value, load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (  # Trainer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    # EvalPrediction,
    HfArgumentParser,
    # SchedulerType,
    TrainingArguments,
    default_data_collator,
    get_scheduler,
)
# from transformers.trainer_utils import get_last_checkpoint

# from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.45.0.dev0")

from loss import configure_loss_function

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = get_logger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "sentence" column for single/multi-label classification task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "The delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    metric_names: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_files: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_files: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_files: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_files is None or self.validation_files is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_files.split(".")[-1]
            assert train_extension in ["csv", "json", "jsonl"], "`train_files` should be a csv or a json file."
            validation_extension = self.validation_files.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_files` should have the same extension (csv or json) as `train_files`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    teacher_model_name_or_path: str = field(
        metadata={"help": "Path to pretrained teacher model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


@dataclass
class MyTrainingArguments(TrainingArguments):
    dtype: str = field(
        default="float32",
        metadata={"help": "Dtype of model."},
    )
    criterion_name: str = field(
        default="bce",
        metadata={"help": "Type of loss."},
    )
    temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Temperature to anneal the logits when computing the softmax."}
    )
    kl_weight: Optional[float] = field(
        default=0.5,
        metadata={
            "help": (
                "Weighting assigned to the MSE loss in the KD formulation. MSE loss is "
                "computed between the teacher-student hidden states and attentions."
            )
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Activate gradient recompuration."},
    )
    pad_to_multiple_of: int = field(
        default=8,
        metadata={"help": "64 for A100."},
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={"help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."},
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of checkpoints to save"},
    )
    with_tracking: bool = field(
        default=True,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )
    report_to: str = field(
        default="all",
        metadata={
            "help": (
                'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
                ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
                "Only applicable when `--with_tracking` is passed."
            ),
        },
    )
    wandb_project: str = field(
        default="text-classification",
        metadata={"help": "The name of the wandb project."},
    )
    wandb_run_name: str = field(
        default=None,
        metadata={"help": "The name of the wandb run."},
    )


def sort_checkpoints(output_dir: str, checkpoint_prefix: str = "checkpoint_") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest.."""

    def extract_number(path: Path) -> Optional[int]:
        match = re.match(f".*{checkpoint_prefix}([0-9]+)", str(path))
        return int(match.group(1)) if match else None

    valid_checkpoints = [
        (extract_number(path), str(path))
        for path in Path(output_dir).glob(f"{checkpoint_prefix}*")
        if path.is_dir() and extract_number(path) is not None
    ]

    return [path for _, path in sorted(valid_checkpoints)]


def rotate_checkpoints(
    output_dir: Optional[str] = None,
    save_total_limit: Optional[int] = None,
    checkpoint_prefix: str = "checkpoint_"
) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        # logger.info("No checkpoint rotation performed (save_total_limit is None or 0)")
        return

    sorted_checkpoints = sort_checkpoints(output_dir=output_dir, checkpoint_prefix=checkpoint_prefix)
    if len(sorted_checkpoints) <= save_total_limit:
        # logger.info(f"No checkpoint rotation needed (found {len(sorted_checkpoints)} checkpoints, limit is {save_total_limit})")
        return

    num_checkpoints_to_delete = len(sorted_checkpoints) - save_total_limit
    checkpoints_to_be_deleted = sorted_checkpoints[:num_checkpoints_to_delete]

    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint {checkpoint} due to save_total_limit {save_total_limit}")
        shutil.rmtree(checkpoint, ignore_errors=True)
        # try:
        #     shutil.rmtree(checkpoint)
        # except OSError as e:
        #     logger.warning(f"Error deleting checkpoint {checkpoint}: {e}")

    # logger.info(f"Checkpoint rotation complete. Kept {save_total_limit} most recent checkpoints.")


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_classification", model_args, data_args)

    if training_args.dtype == "float16":
        mixed_precision = "fp16"
        teacher_dtype = torch.float16
    elif training_args.dtype == "bfloat16":
        mixed_precision = "bf16"
        teacher_dtype = torch.bfloat16
    else:
        mixed_precision = "no"
        teacher_dtype = torch.float32

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if training_args.with_tracking:
        accelerator_log_kwargs["log_with"] = training_args.report_to
        accelerator_log_kwargs["project_dir"] = training_args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        **accelerator_log_kwargs,
    )
    accelerator.init_trackers(
        project_name=training_args.wandb_project,
        init_kwargs={
            "wandb": {
                "name": training_args.wandb_run_name,
                # "dir": training_args.wandb_dir,
            }
        }
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        # transformers.utils.logging.set_verbosity_info()
        # todo
        # transformers.utils.logging.enable_default_handler()
        # transformers.utils.logging.enable_explicit_format()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if training_args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id

            with open(os.path.join(training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    #     + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    # )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # todo
    # Detecting last checkpoint.
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined together
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        # Try print some info about the dataset
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        # data_files = {"train": train_files, "validation": data_args.validation_files}
        train_files = data_args.train_files.split("+")
        data_files = {"train": train_files}

        all_eval_splits = []
        if data_args.validation_files is not None:
            validation_files = data_args.validation_files.split("+")
            if len(validation_files) == 1:
                all_eval_splits.append("validation")
                data_files["validation"] = validation_files[0]
            else:
                for validation_file in validation_files:
                    pretty_name = Path(validation_file).stem
                    # Split name should match '^\w+(\.\w+)*$'
                    pretty_name = re.sub(r"[-_]", ".", pretty_name)
                    pretty_name = "validation_" + pretty_name
                    all_eval_splits.append(pretty_name)
                    data_files[pretty_name] = validation_file

        # test_files will be only used to "cheat" to retrieve all labels
        # in case many labels in test set don't exist in training set
        all_test_splits = []
        if data_args.test_files is not None:
            test_files = data_args.test_files.split("+")
            if len(test_files) == 1:
                all_test_splits.append("test")
                data_files["test"] = test_files[0]
            else:
                for test_file in test_files:
                    pretty_name = Path(test_file).stem
                    # Split name should match '^\w+(\.\w+)*$'
                    pretty_name = re.sub(r"[-_]", ".", pretty_name)
                    pretty_name = "test_" + pretty_name
                    all_test_splits.append(pretty_name)
                    data_files[pretty_name] = test_file

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_files[0].endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # if data_args.remove_splits is not None:
    #     for split in data_args.remove_splits.split(","):
    #         logger.info(f"removing split {split}")
    #         raw_datasets.pop(split)

    # if data_args.train_split_name is not None:
    #     logger.info(f"using {data_args.train_split_name} as train set")
    #     raw_datasets["train"] = raw_datasets[data_args.train_split_name]
    #     raw_datasets.pop(data_args.train_split_name)

    # if data_args.validation_split_name is not None:
    #     logger.info(f"using {data_args.validation_split_name} as validation set")
    #     raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
    #     raw_datasets.pop(data_args.validation_split_name)

    # if data_args.remove_columns is not None:
    #     for split in raw_datasets.keys():
    #         for column in data_args.remove_columns.split(","):
    #             logger.info(f"removing column {column} from split {split}")
    #             raw_datasets[split] = raw_datasets[split].remove_columns(column)

    # standarize label column
    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    # Trying to have good defaults here, don't hesitate to tweak to your needs.

    is_regression = (
        raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if data_args.do_regression is None
        else data_args.do_regression
    )

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        # regession requires float as label type, let's cast it if needed
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(
                    f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                )
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(
                        f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                    )
                    raise error

    else:  # classification
        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        # for split in ["validation", "test"]:
        # for split in all_eval_splits:
        for split in all_eval_splits + all_test_splits:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    teacher_model = AutoModelForSequenceClassification.from_pretrained(
        model_args.teacher_model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=teacher_dtype,
    )

    # set bias of last layer to stablize multi-label training
    # see https://arxiv.org/abs/1901.05555
    def initialize_classifier_bias(model, num_labels):
        # Calculate initial probability
        pi = 1.0 / num_labels
        # Calculate bias using log odds
        bias = -np.log((1.0 - pi) / pi)
        # Set the classifier bias
        with torch.no_grad():
            if hasattr(model.classifier, "out_proj") and isinstance(model.classifier.out_proj, nn.Linear):
                # reberta
                model.classifier.out_proj.bias.fill_(bias)
            elif isinstance(model.classifier, nn.Linear):
                # bert, deberta_v2
                model.classifier.bias.fill_(bias)
            else:
                logger.warning("Unable to find a suitable Linear layer to apply bias.")

    if is_multi_label:
        initialize_classifier_bias(model, num_labels)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # for training ,we will update the config with label infos,
    # if do_train is not set, we will use the label infos in the config
    if training_args.do_train and not is_regression:  # classification, training
        label_to_id = {v: i for i, v in enumerate(label_list)}
        # update config with label infos
        if model.config.label2id != label_to_id:
            logger.warning(
                "The label2id key in the model config.json is not equal to the label2id key of this "
                "run. You can ignore this if you are doing finetuning."
            )
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
        logger.info(f"Num of labels: {len(model.config.label2id)}")
    elif not is_regression:  # classification, but not training
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(model.config.label2id))
        label_to_id = model.config.label2id
    else:  # regression
        label_to_id = None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            # join together text columns into "sentence" column
            sentence = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    sentence[i] += data_args.text_column_delimiter + examples[column][i]
        # Tokenize the texts
        result = tokenizer(sentence, padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets[next(iter(raw_datasets.keys()))].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        # if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
        if all("validation" not in split for split in raw_datasets):
            raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
        else:
            # eval_dataset = raw_datasets["validation"]
            eval_datasets = []
            for eval_split in all_eval_splits:
                eval_dataset = raw_datasets[eval_split]

                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))

                eval_datasets.append(eval_dataset)

    # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # if data_args.metric_names is not None:
    #     metric_names = data_args.metric_names.split("+")
    #     metrics = {}
    #     for metric_name in metric_names:
    #         metrics[metric_name] = (
    #             evaluate.load(metric_name, config_name="multilabel", cache_dir=model_args.cache_dir)
    #             if is_multi_label
    #             else evaluate.load(metric_name, cache_dir=model_args.cache_dir)
    #         )
    #     logger.info(f"Using metric {data_args.metric_names} for evaluation.")
    # else:
    #     if is_regression:
    #         metrics = {"mse": evaluate.load("mse", cache_dir=model_args.cache_dir)}
    #         data_args.metric_names = "mse"
    #         logger.info("Using mean squared error (mse) as regression score, you can use --metric_names to overwrite.")
    #     else:
    #         if is_multi_label:
    #             metrics = {"f1": evaluate.load("f1", config_name="multilabel", cache_dir=model_args.cache_dir)}
    #             data_args.metric_names = "f1"
    #             logger.info(
    #                 "Using multilabel F1 for multi-label classification task, you can use --metric_names to overwrite."
    #             )
    #         else:
    #             metrics = {"accuracy": evaluate.load("accuracy", cache_dir=model_args.cache_dir)}
    #             data_args.metric_names = "accuracy"
    #             logger.info("Using accuracy as classification score, you can use --metric_names to overwrite.")

    def compute_metrics(predictions, references):
        # all_results = {}
        # for metric in metrics.values():
        #     if is_regression:
        #         preds = np.squeeze(predictions)
        #         result = metric.compute(predictions=preds, references=references)
        #     elif is_multi_label:
        #         # preds = np.array([np.where(p > 0, 1, 0) for p in predictions])  # convert logits to multi-hot encoding
        #         preds = np.where(predictions > 0, 1, 0)
        #         # Micro F1 is commonly used in multi-label classification
        #         result = metric.compute(predictions=preds, references=references, average="micro")
        #     else:
        #         preds = np.argmax(predictions, axis=1)
        #         result = metric.compute(predictions=preds, references=references)
        #     # if len(result) > 1:
        #     #     result["combined_score"] = np.mean(list(result.values())).item()
        #     all_results.update(result)

        if is_regression:
            preds = np.squeeze(predictions)
            # todo
            # result = metric.compute(predictions=preds, references=references)
        elif is_multi_label:
            # preds = np.array([np.where(p > 0, 1, 0) for p in predictions])  # convert logits to multi-hot encoding
            # use 0 to binarize logits (before sigmoid)
            # print(predictions)
            preds = np.where(predictions > 0, 1, 0)
            # print(preds.sum(1))
            # Micro F1 is commonly used in multi-label classification
            precision, recall, f1, _ = precision_recall_fscore_support(references, preds, average="micro", zero_division=1)
            result = {"precision": precision, "recall": recall, "f1": f1}
        else:
            preds = np.argmax(predictions, axis=1)
            # todo
            # result = metric.compute(predictions=preds, references=references)

        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=training_args.pad_to_multiple_of)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_device_train_batch_size,
    )
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)
    eval_dataloaders = [DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size) for eval_dataset in eval_datasets]

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps < 0:
        # float to int
        training_args.num_train_epochs = int(training_args.num_train_epochs)
        training_args.max_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)
        overrode_max_train_steps = True
    elif training_args.max_steps > 0:
        logger.info("max_steps is given, it will override any value given in num_train_epochs")
        training_args.num_train_epochs = int(np.ceil(training_args.max_steps / num_update_steps_per_epoch))

    # warmup ratio
    if training_args.warmup_steps == 0:
        training_args.warmup_steps = math.ceil(training_args.max_steps * training_args.warmup_ratio)

    # set num_training_steps to training_args.max_steps if overrode_max_train_steps (when setting num_train_epochs instead of max_steps)
    # since training_args.max_steps doesn't account for num_processes yet
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=training_args.max_steps if overrode_max_train_steps else training_args.max_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    #     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    # )
    model, teacher_model, optimizer, train_dataloader, lr_scheduler, *eval_dataloaders = accelerator.prepare(
        model, teacher_model, optimizer, train_dataloader, lr_scheduler, *eval_dataloaders
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    # len(train_dataloader) become len(train_dataloader) / num_processes now, recompute max_steps or num_train_epochs
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = training_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    if is_multi_label:
        # N x n_classes -> n_classes
        num_examples_per_cls = np.array(train_dataset["label"]).sum(0)
        num_training_examples = train_dataset.num_rows
        loss_fct = configure_loss_function(training_args.criterion_name, num_examples_per_cls, num_training_examples)
        logger.info(f"Using {training_args.criterion_name} loss")

    def kl_divergence(target_distribution, log_predicted_distribution):
        # kl_loss = nn.KLDivLoss(reduction="none")
        # divergence = kl_loss(log_predicted_distribution, target_distribution)
        # # ignore padded tokens from divergence, i.e. where labels are not set to -100
        # padding_mask = labels >= 0
        # padding_mask = padding_mask.unsqueeze(-1)
        # divergence = divergence * padding_mask
        # # take the average over the mini-batch
        # divergence = divergence.sum() / padding_mask.sum()

        # todo:
        # very big num_classes here, loss will be huge if only reduced by batch
        # kl_loss = nn.KLDivLoss(reduction="batchmean")
        # normalized by num of classifiers
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        divergence = kl_loss(log_predicted_distribution, target_distribution) / target_distribution.size(-1)
        return divergence

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            checkpoint_path = training_args.resume_from_checkpoint
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * training_args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // training_args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, training_args.num_train_epochs):
        model.train()
        teacher_model.eval()

        if training_args.with_tracking:
            total_loss = 0
            total_hard_loss = 0
            total_kl_loss = 0

        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            # automatically perform the gradient accumulation for yo
            with accelerator.accumulate(model):
                student_outputs = model(**batch)
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)

                hard_loss = student_outputs.loss
                # todo: other tasks
                if is_multi_label:
                    hard_loss = loss_fct(student_outputs.logits, batch["labels"])

                # rescale distribution by temperature to ensure gradients scale correctly
                teacher_distribution = nn.functional.sigmoid(teacher_outputs.logits / training_args.temperature)
                # log softmax of student predictions for numerical stability
                log_student_distribution = nn.functional.logsigmoid(student_outputs.logits / training_args.temperature)
                # KL-divergence loss (scaled by temperature)
                kl_loss = kl_divergence(teacher_distribution, log_student_distribution) * (training_args.temperature**2)

                loss = (1 - training_args.kl_weight) * hard_loss + training_args.kl_weight * kl_loss

                # We keep track of the loss at each epoch
                if training_args.with_tracking:
                    total_loss += loss.detach().float()
                    total_hard_loss += hard_loss.detach().float()
                    total_kl_loss += kl_loss.detach().float()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(training_args.checkpointing_steps, int):
                if completed_steps % training_args.checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    # todo: tokenizer

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        rotate_checkpoints(
                            os.path.dirname(os.path.abspath(output_dir)),
                            save_total_limit=training_args.save_total_limit,
                            checkpoint_prefix="step_",
                        )

            if completed_steps >= training_args.max_steps:
                break

        model.eval()
        all_eval_metrics = {}
        for eval_split, eval_dataloader in zip(all_eval_splits, eval_dataloaders):
            samples_seen = 0
            all_predictions, all_references = [], []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)
                # predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
                predictions, references = accelerator.gather((outputs.logits, batch["labels"]))
                # If we are in a multiprocess environment, the last batch has duplicates
                if accelerator.num_processes > 1:
                    if step == len(eval_dataloader) - 1:
                        predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                        references = references[: len(eval_dataloader.dataset) - samples_seen]
                    else:
                        samples_seen += references.shape[0]
                all_predictions.append(predictions.cpu().numpy())
                all_references.append(references.cpu().numpy())

            all_predictions = np.concatenate(all_predictions)
            all_references = np.concatenate(all_references)

            eval_metric = compute_metrics(predictions=all_predictions, references=all_references)
            all_eval_metrics.update({f"{eval_split}/{k}": v for k, v in eval_metric.items()})

        # get grad_norm
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            grad_norm = model.get_global_grad_norm()
            # In some cases the grad norm may not return a float
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()
        else:
            grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        metrics_to_log = {
            "train/epoch": epoch,
            "train/step": completed_steps,
            "train/learning_rate": lr_scheduler.get_last_lr()[0],
            "train/grad_norm": grad_norm,
        }

        if training_args.with_tracking:
            metrics_to_log["train/loss"] = total_loss.item() / len(train_dataloader)
            metrics_to_log["train/hard_loss"] = total_hard_loss.item() / len(train_dataloader)
            metrics_to_log["train/kl_loss"] = total_kl_loss.item() / len(train_dataloader)

        metrics_to_log.update(all_eval_metrics)

        logger.info(json.dumps(metrics_to_log, indent=4))

        if training_args.with_tracking:
            accelerator.log(metrics_to_log, step=completed_steps)

        if training_args.push_to_hub and epoch < training_args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(training_args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=training_args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=training_args.hub_token,
                )

        if training_args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if training_args.output_dir is not None:
                output_dir = os.path.join(training_args.output_dir, output_dir)
            accelerator.save_state(output_dir)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                rotate_checkpoints(
                    os.path.dirname(os.path.abspath(output_dir)),
                    save_total_limit=training_args.save_total_limit,
                    checkpoint_prefix="epoch_",
                )

    if training_args.with_tracking:
        accelerator.end_training()

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(training_args.output_dir)
            if training_args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=training_args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=training_args.hub_token,
                )

    if training_args.output_dir is not None:
        with open(os.path.join(training_args.output_dir, "all_results.json"), "w") as f:
            json.dump(metrics_to_log, f)


if __name__ == "__main__":
    main()
