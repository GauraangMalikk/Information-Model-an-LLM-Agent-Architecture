# Information-Model-an-LLM-Agent-Architecture

A modular multi-agent architecture for evaluating hallucination reduction strategies in large language models (LLMs) using structured planning taxonomies, agent-level reasoning, and semantic similarity metrics.

This system combines agent generation, response storage, planning categorization, and vector-based evaluation using FAISS, PCA, and KMeans clustering — implemented and tested on a local TinyLLaMA server.

---
## 📚 Table of Contents

## 📚 Table of Contents

- [Introduction](paper/intro.md)
- [Methods](methods/methods.md)
- [Results](results/results.md)
- [Literature Review](lit_review/literature_review.md)
- [Credits](README.md#credits)


---

## ✅ Overview

This project explores hallucination mitigation in LLMs by:
- Building a **planning-aware agent architecture** with profiling, memory, planning, and action components
- Storing and indexing responses using **semantic vector search (FAISS)**
- Classifying agent strategies using a **planning taxonomy**
- Measuring response stability and alignment via **embedding-based pairwise analysis**
- Clustering performance using **PCA + KMeans**
- Evaluating complexity vs. performance tradeoffs for each agent

---

## 🧱 Architecture (Blocks A–H)

| Block | Description |
|-------|-------------|
| **A** | System setup, agent execution, DB design, and semantic search |
| **B** | Multi-agent response evaluation using cosine/Euclidean metrics |
| **C** | PCA-reduced task matrix and pairwise score analysis |
| **D** | Heatmaps of pairwise agent similarity |
| **E** | Per-agent aggregated metrics across tasks |
| **F** | KMeans clustering of agent pair performance |
| **G** | Planning taxonomy–aware clustering |
| **H** | Complexity-based agent ranking |

Each block is modular and reproducible.

---

## ⚙️ Setup & Usage

### 1. Clone this repo

```bash
git clone https://github.com/GauraangMalikk/Information-Model-an-LLM-Agent-Architecture.git
cd Information-Model-an-LLM-Agent-Architecture
