# Information-Model-an-LLM-Agent-Architecture

A modular multi-agent architecture for evaluating hallucination reduction strategies in large language models (LLMs) using structured planning taxonomies, agent-level reasoning, and semantic similarity metrics.

This system combines agent generation, response storage, planning categorization, and vector-based evaluation using FAISS, PCA, and KMeans clustering — implemented and tested on a local TinyLLaMA server.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture (Blocks A–H)](#architecture-blocks-a-h)
  - [Block A – System Setup, Agent Execution, DB Design, and Semantic Search](#block-a--system-setup-agent-execution-db-design-and-semantic-search)
  - [Block B – Multi-Agent Response Evaluation](#block-b--multi-agent-response-evaluation)
  - [Block C – PCA-Reduced Task Matrix and Pairwise Score Analysis](#block-c--pca-reduced-task-matrix-and-pairwise-score-analysis)
  - [Block D – Heatmaps of Pairwise Agent Similarity](#block-d--heatmaps-of-pairwise-agent-similarity)
  - [Block E – Per-Agent Aggregated Metrics Across Tasks](#block-e--per-agent-aggregated-metrics-across-tasks)
  - [Block F – KMeans Clustering of Agent Pair Performance](#block-f--kmeans-clustering-of-agent-pair-performance)
  - [Block G – Planning Taxonomy–Aware Clustering](#block-g--planning-taxonomyaware-clustering)
  - [Block H – Complexity-Based Agent Ranking](#block-h--complexity-based-agent-ranking)
- [Setup & Usage](#setup--usage)
- [Folder Structure](#folder-structure)
- [Technologies Used](#technologies-used)
- [Credits](#credits)


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
