# 🏥 ICD-10 Codification

This project focuses on the automatic codification of medical documents using ICD-10 (International Classification of Diseases, 10th revision) codes. It aims to train multi-label text classification models for efficient and accurate ICD-10 code assignment.

Key features:

- Data preparation and weak labeling using LLMs
- Traditional ML methods (top-k, tf-idf, logistic regression, FastText)
- Fine-tuning and distillation of pre-trained BERT models
- Multiple re-weighting loss functions to mitigate class imbalance in multi-label classification
- Evaluation and benchmarking tools

## Installation

Follow these steps to set up the development environment:

```bash
git clone https://github.com/bofenghuang/icd-cod.git
cd icd-cod
pip install -r requirements.txt
```

## Usage

### Data Preparation

Prepare QUAERO dataset:

```bash
scripts/prep_quaero.sh
```

Generate weak ICD-10 codes using vLLM:

```bash
scripts/gen_labels.sh
```

Evaluate precision of generated ICD-10 codes:

```bash
scripts/eval_labels.sh
```

Generate medical documents using sampled ICD-10 codes:

```bash
scripts/gen_labels.sh
```

### Model Training

For traditional ML methods (top-k, tf-idf lr, FastText), refer to the notebooks in the `notebooks` folder.

Fine-tune multi-label text classification model:

```bash
examples/train_cls.sh
```

Distill multi-label text classification model from a larger teacher model:

```bash
examples/distil_cls.sh
```

### Evaluation

Run inference on validation set, determine best threshold for binarization, and evaluate performance on all test sets:

```bash
examples/eval.sh
```

Benchmark inference time of different models on various input lengths and devices:

```bash
examples/bench_infer_time.sh
```

## Acknowledgements

- Hugging Face, scikit-learn, and FastText for providing model implementations and training facilities (HF Accelerate)
- The open-source community for developing and sharing pre-trained models
