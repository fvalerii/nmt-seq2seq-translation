# 🌐 Neural-Translation-Seq2Seq: English-to-German Study
### *Encoder-Decoder Architectures for Neural Machine Translation*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fvalerii/nmt-seq2seq-translation/blob/main/notebooks/nmt_english_german_seq2seq.ipynb)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Imperial College London](https://img.shields.io/badge/Academic_Partner-Imperial_College_London-blue.svg)
![BLEU SCore](https://img.shields.io/badge/BLEU_Score-Pending-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📋 Research Overview
This project investigates the implementation of a Neural Machine Translation (NMT) system using a Sequence-to-Sequence (Seq2Seq) framework. Developed as a Capstone Research Study for the Imperial College London TensorFlow 2 Professional Certification, the study focuses on mapping latent semantic representations between English and German using deep recurrent neural networks.

The core objective was to move beyond simple classification and master the complexities of variable-length sequence modeling, latent-space bottlenecks, and autoregressive inference.

---

## 🎯 Technical Architecture
The system utilizes a dual-recurrent framework optimized for high-dimensional semantic mapping.

### **1. The Encoder (Feature Extraction)**
- **Semantic Projection:** Utilizes a pre-trained **NNLM (Neural-Net Language Model)** embedding from TensorFlow Hub to project English tokens into a 128-dimensional latent space.
- **Latent Bottleneck:** Employs a **512-unit LSTM layer** to compress the entire source sequence into a final hidden ($h$) and cell ($c$) state (the "Context Vector").
- **Learned Boundaries:** Includes a custom layer with a trainable terminal embedding to signal sequence boundaries.

### **2. The Decoder (Generative Subnetwork)**
- **State Seeding:** Initialized using the Encoder's final states, ensuring the generative process is grounded in the source context.
- **Recurrent Unfolding:** A 512-unit LSTM layer that maintains temporal state across recursive time-steps.
- **Vocabulary Mapping:** A final Dense layer that projects the LSTM outputs into logit scores across the German vocabulary.

### **Masked Loss Function:**
Implements a custom `SparseCategoricalCrossentropy` that utilizes a boolean mask to ignore zero-padding. This prevents the "gradient dilution" typically seen in variable-length sequence training and ensures the model optimizes for linguistic accuracy rather than padding prediction.

---

## 🛠️ Optimization & Regularization Strategy
To bridge the "Generalization Gap" and prevent memorization of the training set, several advanced techniques were implemented:

* **Spatial & Recurrent Dropout:** Integrated $0.2$ dropout rates to prevent neuron co-dependency.
* **Weight Decay:** Applied **L2 Regularization** ($0.01$) to LSTM kernels to prevent over-specialization.
* **Loss Masking:** A custom masked cross-entropy loss to ignore padding tokens, ensuring the optimizer focuses strictly on linguistic structure.
* **Early Stopping:** Utilized a patience-based monitor on validation loss to restore the "best" model weights.

---

## 📈 Evaluation Metrics
The model is evaluated using both quantitative and qualitative benchmarks:
- **BLEU Score (Bilingual Evaluation Understudy):** Measures n-gram overlap between model predictions and human ground-truth translations.
- **Perplexity:** Tracks the model's confidence in its probability distributions across the target vocabulary.
- **Greedy Search Inference:** Qualitative analysis of the model's ability to handle unseen syntax through a recursive autoregressive feedback loop.

---

## 📊 Research Results

### **Quantitative Evaluation**
The model was evaluated on a held-out test set using Corpus BLEU (Bilingual Evaluation Understudy), measuring n-gram precision (1-4) between the hypothesis and reference.
**Metric:** Corpus BLEU-4
**Score:** [Insert Your Score Here]

[Image showing N-gram overlap for BLEU score]

### **Qualitative Samples**
English Source,Human Reference,Model Translation
"""I am happy.""","""Ich bin glücklich.""","""ich bin gluecklich"""
"""The weather is nice.""","""Das Wetter ist schön.""","""das wetter ist schoen"""

---

## 📈 Performance Analysis

### **Training Metrics**
![Loss Curve](results/loss_curve.png)
*Figure 1: Masked Sparse Categorical Crossentropy over 20 epochs. The convergence of validation loss indicates robust generalization.*

### **Model Architecture**

graph LR
    subgraph Encoder
    A[English Input] --> B(Embedding)
    B --> C[LSTM Layer]
    end

    C -->|Hidden + Cell States| D{Context Vector}

    subgraph Decoder
    D --> E[LSTM Layer]
    E --> F(Dense Softmax)
    F --> G[German Output]
    end

    G -.->|Feedback Loop| E

*Figure 2: Sequence-to-Sequence framework with Latent Bottleneck and Recursive Inference.*

--- 

## 📂 Project Deliverables
- **[Jupyter Notebook](./notebooks/nmt_english_german_seq2seq.ipynb):** 

---

## ⚙️ Execution Guide
The notebook is configured for Automated Pipeline Integration. It automatically fetches the curated English-German corpus (provided by Imperial College) directly from Google Drive using the gdown utility. The dataset is based on the language dataset from ManyThings.org/anki, which consists of over 200,000 sentence pairs. To ensure efficient training on local or cloud-based GPU hardware, the pipeline is restricted to a 20,000-pair subset.

### **Option A: Colab Execution (Cloud)**
The easiest way to run the study is via Google Colab.

### **Option B: Local Execution (WSL2/GPU)**
Recommended for users with NVIDIA GPUs to leverage cuDNN acceleration.

#### **1. Clone the Repository**
```bash
git clone https://github.com/fvalerii/nmt-seq2seq-translation.git
```
### **2. Environment Setup** 
It is recommended to use a environment with Python 3.12.8:
##### Using Pip:
```bash
pip install -r requirements.txt
```
##### Using Conda:
```bash
conda env create -f environment.yml
conda activate nmt_research
```
### *3. Run the Notebook**
Open the notebook in VS Code or Jupyter: `notebooks/nmt_english_german_seq2seq.ipynb`

---

## 💻 Tech Stack
- **Frameworks:** TensorFlow 2.x, Keras, TensorFlow Hub
- **Libraries:** NumPy, Matplotlib, NLTK (for BLEU evaluation), scikit-learn, gdown
- **Architecture:** Long Short-Term Memory (LSTM), Encoder-Decoder, Transfer Learning

---

## 🎓 Certification & Academic Context
This project was developed as a **Capstone Research Study** for the **"TensorFlow 2 for Deep Learning" Professional Certification** by **Imperial College London** (via Coursera).

* **Objective:** Demonstrate mastery of the TensorFlow 2 ecosystem, including custom training loops, masked loss functions, and Seq2Seq modeling.
* **Status:** Verified and Peer-Reviewed.

---

> **Note:** To ensure scientific reproducibility, global random seeds were set for NumPy, Python, and TensorFlow. Note that minor variances (<0.1%) may still occur due to non-deterministic CUDA kernels when switching between GPU architectures.
