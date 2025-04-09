# 🎤 Louis CK Sarcasm & Irony Detection

This NLP project analyzes Louis CK's stand-up comedy performances to detect and classify instances of **sarcasm & irony**, and **normal speech**. The goal is to fine-tune a transformer-based model that can distinguish nuanced humor constructs within spoken language.

---

## 📁 Project Structure(soon)

```
NLP-PROJECT/
├── data/                            # All dataset versions and splits
│   ├── louis_ck_clean_sentences.csv
│   ├── louis_ck_context_windows.csv
│   ├── louis_ck_context_windows_tok.csv
│   ├── louis_ck_sentences.csv
│   ├── louis_ck_train.csv
│   ├── louis_ck_train_auto_labeled.csv
│   ├── louis_ck_val.csv
│   ├── louis_ck_val_auto_labeled.csv
│   ├── louis_ck_test.csv
│   ├── louis_ck_test_auto_labeled.csv
│
├── models/                          # Saved ML and transformer models
│   ├── roberta_louisck/             # Fine-tuned RoBERTa model
│   ├── svm_model_baseline.pkl       # Baseline SVM model
│   ├── svm_model_tuned.pkl          # Tuned SVM model
│   └── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│
├── notebooks/                       # Jupyter notebooks for modeling and analysis
│   ├── RoBERTa_vis.ipynb            # RoBERTa fine-tuning & results visualization
│   └── SVM_base_hyper.ipynb         # Classical ML baseline (SVM)
│
├── plots/                           # Finalized visualizations for reporting/presentation
│   ├── baseline_cm_yellow.png
│   ├── classification_report_heatmap.png
│   ├── confusion_matrix.png
│   ├── overall_metrics_dark_green.png
│   ├── roc_curve.png
│   └── tuned_cm_green.png
│
├── scripts/                         # Python scripts for full data/model pipeline
│   ├── data_clean.py
│   ├── extract_transcripts.py
│   ├── generate_context_windows.py
│   ├── label_with_transformers.py
│   ├── split_dataset.py
│   └── train_model.py
│
├── .gitignore
└── README.md
```

---

## 🔍 Project Goals

- Extract transcript data from Louis CK performances
- Clean and standardize text data (e.g. handle censored terms like `[ __ ]` → `[CENSORED]`)
- Window sentences for context-preserving NLP
- Use a pretrained irony detection model to auto-label data
- Fine-tune a transformer (e.g., RoBERTa) to classify sarcasm/irony
- Evaluate model on unseen segments

---

## 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/louis-ck-irony.git
cd louis-ck-irony
```

2. **Create and activate virtual environment**

```bash
python -m venv nlpenv
source nlpenv/bin/activate        # On Windows: nlpenv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Pipeline Overview

### 1. Extract Transcripts

```bash
python scripts/extract_transcripts.py
```

### 2. Clean Transcript Text

```bash
python scripts/data_clean.py
```

### 3. Generate Context Windows

```bash
python scripts/generate_context_windows.py
```

### 4. Split Dataset

```bash
python scripts/split_dataset.py
```

### 5. Auto-label Train/Val/Test

```bash
python scripts/label_with_transformers.py
```

### 6. Train Transformer Model

```bash
python scripts/train_model.py
```

### 7. Create SVM classifier baseline and tuned versions

```bash
SVM_base_hyper.ipynb
```

### 8. Compare results of SVM vs Roberta and visualize results

---

## 🧠 Model Info

- **Pretrained Base Model**: `roberta-base`
- **Auto-labeler**: `cardiffnlp/twitter-roberta-base-irony`
- **Task**: Binary Classification
  - `Irony` → 1
  - `Non_irony` → 0

---

## 📈 Future Work

- Add multi-label classification (e.g., irony vs sarcasm)
- Include confidence-weighted sampling for manual annotation
- Deploy as a Streamlit web app

---

## 👤 Author

**Vanuhi** – _Data science and machine learning bootcamp project with a focus on NLP & humor analysis_
