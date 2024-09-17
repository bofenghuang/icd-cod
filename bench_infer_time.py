#!/usr/bin/env python
# Copyright 2024  Bofeng Huang

import json
import re
import unicodedata
from time import perf_counter

import fire
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

word_token_pattern = re.compile(r"(?u)\b\w\w+\b")
# stopwords_list = stopwords.words("english") + stopwords.words("french")
stopwords = set(stopwords.words("french"))
# init SnowballStemmer
stemmer = SnowballStemmer("french")

TMPDIR = "tmp"


def jload(input_file):
    with open(input_file) as f:
        return json.load(f)


def sigmoid(x: np.ndarray) -> np.ndarray:
    r"""A numerically stable version of the logistic sigmoid function.
    Inspired from https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
    """

    def _positive_sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(x: np.ndarray) -> np.ndarray:
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)
        return exp / (exp + 1)

    positive = x >= 0
    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result


# def convert_graph_to_onnx(model, tokenizer, output_model_file, input_names, output_names):
#     """Convert model graph to ONNX format."""
#     dummy_input = tokenizer("This is a test", return_tensors="pt")

#     model.cpu()
#     model.eval()

#     axes = {}
#     axes.update({ax: {0: "batch_size", 1: "seq_len"} for ax in input_names})
#     axes.update({ax: {0: "batch_size"} for ax in output_names})

#     torch.onnx.export(
#         model,
#         dummy_input,
#         output_model_file,
#         export_params=True,
#         opset_version=11,
#         do_constant_folding=True,
#         input_names=input_names,
#         output_names=output_names,
#         dynamic_axes=axes,
#         # verbose=True,
#     )


def preprocess_and_tokenize(s):

    # adapted to optionally keep selected symbols
    def remove_symbols(s: str, keep: str = ""):
        """
        Replace any other markers, symbols, punctuations with a space, keeping diacritics
        """
        # fmt: off
        return "".join(
            c
            if c in keep
            else " "
            if unicodedata.category(c)[0] in "MSP"
            else c
            for c in unicodedata.normalize("NFKC", s)
        )
        # fmt: on

    def normalize_text(s):
        s = s.lower()  # lowercase

        # normalize punkt
        s = unicodedata.normalize("NFKD", s)  # normalize unicode chars
        s = re.sub(r"[´′’ʼ‘ʻ`]", "'", s)  # standardize quotes and apostrophes
        s = re.sub(r"[−‐–—]", "-", s)  # standardize hyphens and dashes
        s = re.sub(r"\s*'\s*", "' ", s)  # add space after apostrophe
        s = re.sub(r"\s*([,.:;!?])", r" \1", s)  # add space before comma/period
        s = re.sub(r"\s*([-/])\s*", r" \1 ", s)  # add spaces around slash/dash
        s = re.sub(r"\(\s*", "( ", s)  # add space after parentheses
        s = re.sub(r"\s*\)", " )", s)  # add space before parentheses

        # remove punkt except "'"
        """
        s = remove_symbols(s, keep="'")
        s = re.sub(r"\s*'\s*", "' ", s)  # add space after apostrophe
        """

        s = re.sub(r"æ", "ae", s)  # standarize french chars
        s = re.sub(r"œ", "oe", s)  # standarize french chars
        s = re.sub(r"\s+", " ", s).strip()  # remove extra whitespace
        return s

    def word_tokenize(s):
        # return nltk.word_tokenize(s)
        # more efficient
        return word_token_pattern.findall(s)

    def stem_word(word):
        return stemmer.stem(word)

    # normalize the text
    s = normalize_text(s)
    # tokenize
    # return word_tokenize(s)
    # tokenize, remove stopwords
    # return [w for w in word_tokenize(s) if w not in stopwords]
    # tokenize, remove stopwords, and stem
    return [stem_word(w) for w in word_tokenize(s) if w not in stopwords]


def load_sklearn_infer_func(model_path: str):
    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    def _infer(text):
        logits = model.predict_proba([text])[0]
        probabilities = sigmoid(logits)
        return probabilities

    return _infer


def load_fasttext_infer_func(model_path: str):
    import re

    import fasttext

    model = fasttext.load_model(model_path)
    pattern_newline = re.compile("\n")

    def _infer(text):
        # tmp fix
        text = pattern_newline.sub("", text)
        _, pred_probas = model.predict(text)  # , k=3, threshold=0.5)
        return pred_probas

    return _infer


def load_hf_model_infer_func(
    model_name_or_path: str,
    torch_dtype: str,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    # use_onnx: bool = False,
    num_threads: int = -1,
):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if num_threads > 0:
        torch.set_num_threads(num_threads)
    print(f"Using {torch.get_num_threads()} CPU threads")

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

    if use_compile:
        model = torch.compile(model, mode=compile_mode)

    # if use_onnx:
    #     print("Exporting to onnx...")
    #     dummy_input = tokenizer("This is a test", return_tensors="pt")
    #     os.makedirs(TMPDIR, exist_ok=True)
    #     torch.onnx.export(model, (dummy_input.input_ids, dummy_input.attention_mask), f"{TMPDIR}/model.onnx", opset_version=12)
    #     import onnxruntime as ort

    #     model = ort.InferenceSession(f"{TMPDIR}/model.onnx")

    def process_and_infer(text):
        inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")

        # if use_onnx:
        #     ort_inputs = {"input_ids": inputs["input_ids"].numpy(), "attention_mask": inputs["attention_mask"].numpy()}
        #     ort_outputs = model.run(None, ort_inputs)
        #     logits = ort_outputs[0]
        #     probabilities = sigmoid(logits)
        # else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)

        return probabilities

    return process_and_infer


def benchmark_inference_time(infer_func, sentences, iterations):
    latencies_per_seq = []

    # warm up
    # initialize the GPU and prevent it from going into power-saving mode
    for _ in range(10):
        infer_func(sentences[0])

    # time runs
    for i in range(iterations):
        start_time = perf_counter()
        infer_func(sentences[i % len(sentences)])
        latency = 1000 * (perf_counter() - start_time)  # Unit: ms
        latencies_per_seq.append(latency)

    # statistics
    return {
        "num_iter": iterations,
        # "model_id": model_id,
        # "framework": framework,
        # "device": device,
        # "time_ms_per_seq": latencies_per_seq,
        "time_avg_ms_per_seq": np.mean(latencies_per_seq),
        "time_median_ms_per_seq": np.median(latencies_per_seq),
        "time_std_ms_per_seq": np.std(latencies_per_seq),
        "time_p95_ms_per_seq": np.percentile(latencies_per_seq, 95),
    }


def main(
    model_type: str,
    model_name_or_path: str,
    input_file: str,
    output_file: str,
    torch_dtype: str = "float32",
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    # use_onnx: bool = False,
    signature: str = "",
    iterations: int = 100,
    num_threads: int = -1,
):
    # print("Loading data...")
    input_data_dict = jload(input_file)

    if model_type == "hf":
        infer_func = load_hf_model_infer_func(
            model_name_or_path,
            torch_dtype,
            use_compile=use_compile,
            compile_mode=compile_mode,
            num_threads=num_threads,
        )  # , use_onnx=use_onnx)
    elif model_type == "sklearn":
        infer_func = load_sklearn_infer_func(model_name_or_path)
    elif model_type == "fasttext":
        infer_func = load_fasttext_infer_func(model_name_or_path)
    else:
        raise ValueError(f"Invalid model type {model_type}")

    results = []
    for length, sentences in tqdm(input_data_dict.items()):
        r = benchmark_inference_time(infer_func, sentences, iterations=iterations)
        r["length"] = length
        r["model_name"] = model_name_or_path.rsplit("/", 1)[-1]
        r["signature"] = signature
        results.append(r)

    df = pd.DataFrame(results)
    print(df.head())

    # export
    # df.to_json(output_file, orient="records", lines=True, force_ascii=False)
    # append
    json_records = df.to_json(orient="records", lines=True, force_ascii=False)
    # Open the file in append mode and write each record
    with open(output_file, "a", encoding="utf-8") as f:
        for record in json_records.splitlines():
            f.write(record + "\n")


if __name__ == "__main__":
    fire.Fire(main)
