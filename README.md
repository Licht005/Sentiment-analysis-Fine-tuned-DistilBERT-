# Fine-Tuning DistilBERT for Sentiment Analysis

This project demonstrates how to fine-tune **DistilBERT** using the Hugging Face `transformers` library on a sentiment analysis dataset.

---

## 📂 Project Structure

```
HuggingFace Sentiment analysis/
│
├── outputs/                  # Saved model checkpoints and final fine-tuned model
│   ├── checkpoint-500/       # Intermediate checkpoint (example)
│   └── fine_tuned_distilbert/ # Final fine-tuned model (ready to load)
│
├── src/
│   ├── train.py              # Script for training & fine-tuning
│   └── Load_test.py          # Script to load the fine-tuned model and run inference
│
└── requirements.txt          # Python dependencies
```

---

## ⚙️ Installation

1. Clone this repository or download the folder.
2. Create and activate a virtual environment (recommended). Example with conda:

```bash
conda create -n pytorch_env python=3.10 -y
conda activate pytorch_env
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

To fine-tune DistilBERT, run:

```bash
python src/train.py
```

This will:
- Load the dataset (`imdb` sentiment dataset).
- Fine-tune `distilbert-base-uncased`.
- Save the final model in `outputs/fine_tuned_distilbert/`.

You can adjust training parameters like `epochs`, `batch_size`, and `learning_rate` inside `train.py`.

---

## 🔍 Testing / Inference

After training, run inference with:

```bash
python src/Load_test.py
```

Example output:

```python
[{'label': 'LABEL_1', 'score': 0.9487}]  # Positive review
[{'label': 'LABEL_0', 'score': 0.9471}]  # Negative review
```

> **Note:** By default, labels are `"LABEL_0"` (Negative) and `"LABEL_1"` (Positive).

---

## 📊 Results

- Model: `distilbert-base-uncased`
- Dataset: IMDB Reviews
- Accuracy: ~87%
- F1 Score: ~0.88

---

## 💾 Saving & Loading Model

The fine-tuned model is saved in `outputs/fine_tuned_distilbert`. You can load it directly:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "outputs/fine_tuned_distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(classifier("I really enjoyed this movie!"))
```

---

## 📌 Notes

- If you want to extend this to **3-class classification** (Positive, Neutral, Negative), you’ll need:
  1. A dataset with 3 labels.
  2. Change `num_labels=3` in the model config.
  3. Adjust label mappings accordingly.

- Training with GPU (CUDA) is much faster. On CPU, expect longer training times.

---

👨‍💻 Author: *Lucas W. K.*  
📅 Date: 2025

