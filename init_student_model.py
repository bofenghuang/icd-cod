#!/usr/bin/env python
# coding=utf-8
# Copyright 2024  Bofeng Huang

"""Initialize model with different layer selection strategies."""

import copy

import fire
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main(
    model_name_or_path: str,
    output_dir: str,
    encoder_layers: int = 2,
    layer_selection_strategy: str = "max_spaced",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    teacher_config = teacher_model.config
    teacher_encoder_layers = teacher_config.num_hidden_layers

    student_config = copy.deepcopy(teacher_config)
    student_config.update({"num_hidden_layers": encoder_layers})

    if layer_selection_strategy == "max_spaced":
        encoder_mapping = np.linspace(0, teacher_encoder_layers - 1, encoder_layers, dtype=int)
        encoder_mapping[-1] = teacher_encoder_layers - 1
        print(f"Initialized max_spaced encoder mapping: {encoder_mapping}")
    elif layer_selection_strategy == "first":
        encoder_mapping = np.arange(encoder_layers)
        print(f"Initialized first {encoder_layers} layers mapping: {encoder_mapping}")
    elif layer_selection_strategy == "last":
        encoder_mapping = np.arange(teacher_encoder_layers - encoder_layers, teacher_encoder_layers)
        print(f"Initialized last {encoder_layers} layers mapping: {encoder_mapping}")
    else:
        raise ValueError(f"Unsupported layer selection strategy: {layer_selection_strategy}")

    encoder_map = {}
    for student_layer, teacher_layer in enumerate(encoder_mapping):
        encoder_map[teacher_layer] = student_layer

    # init the student params from the teacher model
    student_model = AutoModelForSequenceClassification.from_config(student_config)
    missing_keys, unexpected_keys = student_model.load_state_dict(teacher_model.state_dict(), strict=False)
    if len(missing_keys) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for AutoModelForSequenceClassification. \n"
            f"Missing key(s) in state_dict: {missing_keys}"
        )
    if encoder_layers == teacher_encoder_layers:
        encoder_keys = [key for key in unexpected_keys if "roberta.encoder.layer" in key]
        if len(encoder_keys) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for AutoModelForSequenceClassification. \n"
                f"Unexpected key(s) in state_dict: {encoder_keys}"
            )

    # todo: only for roberta here, depending on model implementation
    if encoder_layers is not None:
        for layer in range(teacher_encoder_layers):
            if layer in encoder_map:
                # re-introduce pre-defined layers from the teacher
                student_model.roberta.encoder.layer[encoder_map[layer]].load_state_dict(
                    teacher_model.roberta.encoder.layer[layer].state_dict()
                )

    # remove the teacher params and model
    del teacher_model

    # save the converted weights and model
    if output_dir is not None:
        student_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    # check we can do a forward pass with the saved model - first load the weights and processor
    print("Checking we can load the saved model...")
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    student_model = AutoModelForSequenceClassification.from_pretrained(output_dir)

    # do a forward pass - outputs will be gibberish for the initialised model so we can't check them
    # but we make can sure the model runs as expected
    print("Checking we can run the converted model forward...")
    dummy_text = "This is a sample sentence for testing."
    encoded_input = tokenizer(dummy_text, return_tensors="pt")
    _ = student_model(**encoded_input)
    print("Conversion successful!")


if __name__ == "__main__":
    fire.Fire(main)
