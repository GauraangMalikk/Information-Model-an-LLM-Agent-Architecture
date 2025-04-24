# Introduction

Large Language Models (LLMs) have demonstrated exceptional capabilities across a wide range of tasks including summarization, question answering, and code generation. Despite these advancements, a fundamental limitation persists: **hallucination**. This phenomenon refers to the generation of plausible-sounding but factually incorrect or unverifiable content, which undermines the reliability and trustworthiness of LLM-generated outputs, especially in high-stakes applications.

While traditional methods address hallucination through instruction tuning, reinforcement learning, or retrieval-augmented generation, recent work has explored more dynamic, **multi-agent approaches**. These systems leverage **redundancy, disagreement, and consensus** among multiple agents to improve robustness and factual alignment. However, current implementations often lack principled evaluation frameworks or modular analysis of agent behavior.

In this work, we propose a **modular, taxonomy-aware multi-agent architecture** for hallucination mitigation. Our system:
- Encodes agents with distinct **planning strategies** (e.g., single-path vs. multi-path reasoning)
- Stores and evaluates responses in a **semantic vector space** using SentenceTransformer embeddings and FAISS
- Applies **clustering (KMeans)** and **dimensionality reduction (PCA)** to visualize response diversity and convergence
- Tracks **pairwise agreement** and **agent-level performance metrics** across multiple tasks
- Integrates a **complexity factor analysis** to identify agents that perform well while remaining simple in design

This architecture is implemented using TinyLLaMA on a local machine and is designed for extensibility and reproducibility. The overall pipeline is organized into **eight modular blocks (Blocks A–H)**, ranging from system setup and data storage to pairwise comparison and visual analytics.

We hypothesize that:
1. Structuring agents by planning behavior enables deeper insight into LLM output stability
2. Semantic vector representations offer richer evaluation than discrete accuracy metrics
3. A modular, multi-agent setup allows scalable experimentation and analysis

By combining agent diversity, semantic embeddings, and structured reasoning taxonomies, this research aims to advance both the understanding and reduction of hallucination in language models.


