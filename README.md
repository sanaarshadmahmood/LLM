#  Sentiment Analysis on TweetEval using RoBERTa + Domain Adaptation

This project demonstrates how to perform **sentiment classification** on tweets using the [TweetEval Sentiment Dataset](https://huggingface.co/datasets/tweet_eval) and fine-tune a pre-trained Transformer model (`distilbert-base-uncased`). The workflow includes domain adaptation using masked language modeling (MLM), followed by supervised classification and hyperparameter tuning.

---

##  Objectives

- Fine-tune a masked language model (MLM) using tweet data.
- Adapt the model for sentiment classification with 3 labels:
  - `0`: Negative
  - `1`: Neutral
  - `2`: Positive
- Tune learning rates and identify the best-performing model.
- Evaluate model performance on a held-out test set.
- Use the final model for real-time inference.

---

##  Dataset

We use the `sentiment` subset of the [TweetEval](https://huggingface.co/datasets/tweet_eval) benchmark dataset.

| Split        | Description              |
|--------------|--------------------------|
| `train`      | Training data for tweets |
| `validation` | Used for hyperparameter tuning |
| `test`       | Held-out test set for evaluation |

---

##  Model Architecture

- **Base Model**: `distilbert-base-uncased`
- **Stage 1**: Domain adaptation using Masked Language Modeling (MLM)
- **Stage 2**: Sequence classification fine-tuned using labeled data

---

##  Workflow Overview

### 1. Environment Setup
Install required libraries:
- `transformers`
- `datasets`
- `evaluate`

### 2. Dataset Loading and Exploration
- Load `TweetEval` dataset using Hugging Face Datasets library
- Visualize class distribution of sentiment labels

### 3. Domain Adaptation (Unsupervised)
- Fine-tune the base `distilbert` model using **Masked Language Modeling (MLM)** on raw tweets
- Save the adapted model for downstream use

### 4. Fine-tuning for Sentiment Classification
- Fine-tune the MLM-adapted model for sentiment classification using:
  - CrossEntropyLoss
  - Trainer API
  - Learning rate tuning with values: `5e-5`, `3e-5`, `2e-5`
- Save the best-performing model and tokenizer

### 5. Evaluation
- Evaluate best model on the test set
- Compute and display:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix (printed + plotted)

### 6. Inference
- Define a `predict()` function
- Use the trained model to predict sentiment for new tweet samples

---

##  Results

Example performance on the test set (may vary slightly):
- **Accuracy**: ~70%
- **Precision**: ~70%
- **Recall**: ~70%
- **F1 Score**: ~70%

Sample Predictions:
```python
samples = [
  "I love how @Delta handled my flight issue—superb service!",
  "Worst airline ever. My bag was lost and no apology."
]
predict(samples)
# Output: ['positive', 'negative']
