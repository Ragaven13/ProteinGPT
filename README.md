# 🧬 ProteinGPT: Function-Conditioned Protein Sequence Generation with Transformer Language Models

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red?style=for-the-badge&logo=pytorch)
![Transformers](https://img.shields.io/badge/Transformer-GPT-green?style=for-the-badge)
![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Protein_AI-purple?style=for-the-badge)
![Research](https://img.shields.io/badge/Research-Generative_Biology-orange?style=for-the-badge)

---

# 📌 Overview

ProteinGPT is a Generative AI framework for **protein sequence design** using Transformer language models.

Inspired by Large Language Models (LLMs), ProteinGPT treats amino acids as tokens and learns biological sequence patterns through next-token prediction.

The project investigates whether **biological function conditioning** can improve protein generation quality and controllability compared to sequence-only models.

### Key Capabilities

- 🧬 Protein Sequence Generation
- 🎯 Function-Conditioned Generation
- 🤖 GPT Transformer Architecture
- 🔬 Bioinformatics Sequence Modeling
- 📊 Novelty & Diversity Evaluation
- ⚖️ LSTM vs GPT Benchmarking

---

# 🎯 Problem Statement

Can biological function labels improve protein generation quality and controllability?

Specifically:

- Generate realistic protein sequences
- Control generated protein function
- Increase sequence diversity
- Maintain novelty without memorization
- Compare Transformer-based models against traditional sequence models

without relying solely on next-token prediction accuracy.

---

# 🏗️ System Architecture

```text
Swiss-Prot Dataset
        │
        ▼
Data Cleaning
        │
        ▼
Function Label Assignment
        │
        ▼
Tokenization
        │
        ▼
 ┌────────────────────┐
 │ Conditional GPT    │
 └────────────────────┘
        │
        ▼
Protein Generation
        │
        ▼
Evaluation Pipeline
        │
        ▼
Model Comparison
```

---

# 📂 Dataset

## UniProt Swiss-Prot

High-quality manually curated protein database used for training.

### Dataset Statistics

| Metric | Value |
|----------|---------:|
| Raw Proteins | 574,627 |
| Clean Proteins | 460,903 |
| Function-Labeled Proteins | 95,121 |
| Max Sequence Length | 512 |
| Amino Acids | 20 |
| Vocabulary Size | 29 |

---

# 🧪 Functional Protein Classes

| Class | Samples |
|---------|---------:|
| Enzyme | 68,431 |
| Transporter | 9,937 |
| Membrane Protein | 6,900 |
| Receptor | 6,036 |
| DNA-Binding | 3,501 |
| Immune | 316 |

---

# ⚙️ Tech Stack

![Python](https://img.shields.io/badge/Python-Programming-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-purple)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green)
![Transformers](https://img.shields.io/badge/Transformers-GPT-orange)
![Bioinformatics](https://img.shields.io/badge/Bioinformatics-Protein_Modeling-teal)
![Git](https://img.shields.io/badge/Git-Version_Control-red)

---

# 🤖 Models Evaluated

## 1️⃣ LSTM Baseline

Traditional sequence modeling approach.

```text
Embedding
    ↓
LSTM
    ↓
Linear Layer
```

---

## 2️⃣ Vanilla GPT

Sequence-only Transformer model.

```text
Protein Sequence
       ↓
 GPT Transformer
       ↓
Next Amino Acid Prediction
```

---

## 3️⃣ ProteinGPT (Proposed)

Function-conditioned Transformer architecture.

```text
<enzyme>
     +
Protein Sequence
        ↓
 GPT Transformer
        ↓
Controlled Protein Generation
```

---

# 🏋️ Training Configuration

| Parameter | Value |
|------------|---------:|
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Sequence Length | 512 |
| Batch Size | 16 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Epochs | 5 |

---

# 📊 Evaluation Metrics

## Novelty Score

Measures how different generated proteins are from proteins seen during training.

```text
Higher = Better
```

---

## Diversity Score

Measures variability among generated protein sequences.

```text
Higher = Better
```

---

## Average Sequence Length

Measures biological realism relative to Swiss-Prot proteins.

---

# 🏆 Results

| Model | Novelty ↑ | Diversity ↑ | Avg Length | Function Control |
|---------|---------:|---------:|---------:|---------|
| LSTM Baseline | 0.7229 | 0.9336 | 225.9 | ❌ |
| Vanilla GPT | 0.7343 | 0.9508 | 252.7 | ❌ |
| ProteinGPT | **0.7356** | **0.9574** | **264.2** | ✅ |

---

# 📈 Experimental Findings

### ProteinGPT achieved:

✅ Highest Novelty Score

✅ Highest Diversity Score

✅ Function-Controlled Generation

✅ Better Sequence Quality than Baselines

✅ Controllable Protein Design

---

# 🧬 Example Generation

### Input

```text
<enzyme>
```

### Generated Sequence

```text
MKKLVDSDNVKTRLPYTIPMVEIGKDEAVVRTINIKERLVFIRETPV...
```

---

# 📁 Project Structure

```text
ProteinGPT/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
│   ├── figures/
│   └── evaluations/
│
├── src/
│   ├── preprocess.py
│   ├── tokenizer.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate.py
│   └── compare_models.py
│
└── README.md
```

---

# 🚀 Future Improvements

- ESM-2 Embedding Evaluation
- BLAST Similarity Analysis
- AlphaFold Structure Validation
- Protein Diffusion Models
- Reinforcement Learning Optimization
- Multi-Function Protein Design
- Retrieval-Augmented Protein Generation

---

# 📚 Citation

```bibtex
@article{proteingpt2026,
  title={ProteinGPT: Function-Conditioned Protein Sequence Generation with Transformer Language Models},
  author={Anandha Ragaven R},
  year={2026}
}
```

---

# 👨‍💻 Author

### Anandha Ragaven R

M.S. Artificial Intelligence  
Stevens Institute of Technology

LinkedIn: [Your LinkedIn]
GitHub: [Your GitHub]
