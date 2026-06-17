🧬 ProteinGPT: Function-Conditioned Protein Sequence Generation with Transformer Language Models










📌 Overview

ProteinGPT is a generative AI framework for protein sequence design using Transformer language models.

The project explores whether biological function annotations can improve protein generation quality compared to traditional sequence-only models.

Inspired by Large Language Models (LLMs), ProteinGPT treats amino acids as tokens and learns protein grammar through next-token prediction.

The framework supports:

🧬 Protein sequence generation
🎯 Function-conditioned generation
📊 Novelty & diversity evaluation
🤖 Transformer vs LSTM comparison
🔬 Bioinformatics sequence modeling
🎯 Research Question

Can biological function labels improve protein generation quality and controllability compared to sequence-only models?

ProteinGPT investigates whether conditioning generation on protein functions such as:

Enzyme
Transporter
Membrane Protein
Receptor
DNA-Binding Protein

can improve generated sequence diversity while enabling controllable protein design.

🏗️ System Architecture
Swiss-Prot Dataset
        │
        ▼
Protein Preprocessing
        │
        ▼
Function Label Assignment
        │
        ▼
Tokenization
        │
        ▼
 ┌──────────────────┐
 │ Conditional GPT  │
 └──────────────────┘
        │
        ▼
Protein Generation
        │
        ▼
Novelty Evaluation
Diversity Evaluation
Model Comparison
📂 Dataset
UniProt Swiss-Prot

Source:

UniProt Consortium

Dataset Statistics:

Metric	Value
Raw Proteins	574,627
Clean Proteins	460,903
Function-Labeled Proteins	95,121
Max Sequence Length	512
Amino Acids	20
Vocabulary Size	29
🧪 Protein Function Classes
Function	Count
Enzyme	68,431
Transporter	9,937
Membrane	6,900
Receptor	6,036
DNA Binding	3,501
Immune	316
🤖 Models
1. LSTM Baseline
Embedding
    ↓
LSTM
    ↓
Linear Layer

Purpose:

Traditional sequence model baseline
2. Vanilla GPT
Protein Sequence
        ↓
GPT Transformer
        ↓
Next Amino Acid Prediction

Purpose:

Transformer without function information
3. ProteinGPT (Proposed Model)
<enzyme>
     +
Protein Sequence
        ↓
GPT Transformer
        ↓
Function-Conditioned Generation

Purpose:

Controllable protein generation
⚙️ Training Configuration
Parameter	Value
Embedding Dimension	128
Attention Heads	4
Transformer Layers	4
Sequence Length	512
Batch Size	16
Optimizer	AdamW
Learning Rate	3e-4
Epochs	5
📊 Evaluation Metrics

ProteinGPT was evaluated using:

Novelty Score

Measures how different generated proteins are from training proteins.

Higher = Better
Diversity Score

Measures variability among generated proteins.

Higher = Better
Average Sequence Length

Measures biological realism compared to Swiss-Prot proteins.

🏆 Results
Model	Novelty ↑	Diversity ↑	Avg Length	Function Control
LSTM Baseline	0.7229	0.9336	225.9	❌
Vanilla GPT	0.7343	0.9508	252.7	❌
ProteinGPT	0.7356	0.9574	264.2	✅
Key Findings

✅ Highest novelty score

✅ Highest diversity score

✅ Function-conditioned generation

✅ Outperformed sequence-only GPT baseline

✅ Outperformed LSTM baseline

📈 Visualizations
results/figures/
├── novelty_comparison.png
├── diversity_comparison.png
└── length_comparison.png
🚀 Example Generation
Input
<enzyme>
Generated Protein
MKKLVDSDNVKTRLPYTIPMVEIGKDEAVVRTINIKERLVFIRETPV...
🛠️ Tech Stack
Python
PyTorch
NumPy
Pandas
Matplotlib
Bioinformatics
Transformer Architecture
LSTM
Git/GitHub
🔮 Future Work
ESM-2 Protein Foundation Model Evaluation
BLAST Similarity Analysis
Structural Validation
AlphaFold Integration
Diffusion-Based Protein Generation
Reinforcement Learning for Protein Optimization
📚 Citation
@article{proteingpt2026,
  title={ProteinGPT: Function-Conditioned Protein Sequence Generation with Transformer Language Models},
  author={Anandha Ragaven R},
  year={2026}
}
👨‍💻 Author

Anandha Ragaven R

M.S. Artificial Intelligence
Stevens Institute of Technology
