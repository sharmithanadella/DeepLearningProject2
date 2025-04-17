Here’s a suggested **README.md** for your `dl-project2.ipynb` notebook. You can drop this file alongside your notebook in the same repo or folder:

```markdown
# DeepMochi – DL Project 2: Parameter‑Efficient Text Classification with LoRA

This repository contains **DL Project 2** for Spring 2025: a parameter‑efficient fine‑tuning of RoBERTa‑base on the AGNEWS dataset using **LoRA** (Low‑Rank Adaptation).

---

## 📋 Table of Contents

- [Project Overview](#project-overview)  
- [Notebook Structure](#notebook-structure)  
- [Getting Started](#getting-started)  
  - [Requirements](#requirements)  
  - [Installation](#installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Citations](#citations)  

---

## 🚀 Project Overview

In this project, we demonstrate how to fine‑tune a large pretrained transformer (**RoBERTa‑base**, 125M parameters) on the AGNEWS news‐classification task while updating **fewer than one million parameters** by injecting low‑rank adapters (LoRA) into the attention layers.

Key highlights:

- **Dataset:** AGNEWS (4‐way classification of news articles)  
- **Base Model:** RoBERTa‑base (all original weights frozen)  
- **Parameter‐Efficient Fine‐Tuning:** LoRA adapters (rank _r_ = 8, α = 16) applied to query/value projections  
- **Performance:** 94.22 % test accuracy with only 888,580 trainable parameters  

---

## 📑 Notebook Structure

The Jupyter notebook `dl-project2.ipynb` is organized into the following sections:

1. **Environment Setup**  
   - Import libraries (PyTorch, Transformers, PEFT, Datasets, etc.)  
   - GPU detection and seed setting  

2. **Data Loading & Preprocessing**  
   - Download and load AGNEWS via `datasets`  
   - Tokenization with `AutoTokenizer` (max length = 128)  
   - Formatting for PyTorch  

3. **Model Definition**  
   - Load pretrained `roberta-base` model for sequence classification  
   - Configure LoRA adapters (`LoraConfig`) on query/value modules  
   - Wrap model with `get_peft_model`  

4. **Training Setup**  
   - Define `TrainingArguments` (learning rate, batch sizes, epochs, etc.)  
   - Implement `compute_metrics` for accuracy  
   - Initialize `Trainer`  

5. **Training & Evaluation**  
   - Run `trainer.train()`  
   - Evaluate on test set (`trainer.evaluate()`)  
   - Print final accuracy  

6. **Inference & Submission**  
   - Load and preprocess unlabeled test data from pickle  
   - Batch‑wise prediction loop  
   - Save predictions to `submission.csv`  

---

## 🛠️ Getting Started

### Requirements

- Python **3.8+**  
- PyTorch **1.13+**  
- Hugging Face **transformers**, **datasets**, **peft**  
- **scikit-learn** (for accuracy)  

You can install the core dependencies with:

```bash
pip install torch transformers datasets peft scikit-learn
```

### Installation

1. Clone this repo (or download the notebook):

   ```bash
   git clone https://github.com/yourusername/deepmochi-dl-project2.git
   cd deepmochi-dl-project2
   ```

2. (Optional) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

> **Note:** A `requirements.txt` with pinned versions can be added:

```
torch>=1.13
transformers>=4.30
datasets>=2.10
peft>=0.4
scikit-learn>=1.2
```

---

## ▶️ Usage

1. Open the notebook:

   ```bash
   jupyter lab dl-project2.ipynb
   ```

2. Run cells in order.  
   - Make sure you have access to a GPU for faster training.  
   - The Kaggle data loader cells at the top can be removed if running locally—just ensure the AGNEWS dataset downloads via the hub.

3. After training, `submission.csv` will be created in the working directory.

---

## 📊 Results

- **Test Accuracy:** 94.22 %  
- **Trainable Parameters:** 888,580 (< 1 % of RoBERTa‑base)  
- **LoRA Rank (r):** 8  
- **Scaling Factor (α):** 16  

See the notebook’s **Results** and **Model Architecture** sections for detailed tables and diagrams.

---

## 📚 Citations

If you use this code, please cite:

- Edward J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” *arXiv:2106.09685*, 2021.  
- Tim Dettmers et al., “PEFT: State-of-the-Art Parameter-Efficient Fine-Tuning Methods,” *arXiv:2402.12354*, 2024.  

---

**DeepMochi** | Spring 2025 | Sharmitha Vijayan (sv2877), Tse Heng Hsueh (th3448), Sanjana Kosuru (sk11528)  
```

Feel free to adjust paths or add any project‑specific notes!
