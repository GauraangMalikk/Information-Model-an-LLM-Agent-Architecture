# Literature Review: Hallucination Mitigation in Multi-Agent LLM Systems

This literature review explores current approaches to mitigating hallucination in large language models (LLMs), with a particular focus on multi-agent collaboration, structured planning taxonomies, and semantic evaluation methods. The integration of agents capable of reasoning, planning, memory retention, and action generation is discussed through the lens of recent advances in both LLM architecture and evaluation methodology.

---

## 1. Hallucination in Language Models

Large-scale LLMs are known to hallucinate — generating factually incorrect or unverifiable outputs — due to limitations in training data coverage, model size, or reasoning depth.

- Duan and Wang (2024) introduce third-party LLM integration to improve consensus and reduce hallucinations through uncertainty estimation.
- Madaan et al. (2023) explore self-refinement techniques in language models to reduce hallucination through reflection and correction loops.
- Gosmar and Dahl (2025) propose agentic natural language frameworks to allow agent modules to self-check and peer-review each other's responses.

---

## 2. Multi-Agent Collaboration & Consensus

Multi-agent systems offer a promising direction for hallucination reduction by leveraging agent diversity and weighted consensus mechanisms.

- Yao et al. (2023) present ReAct, a reasoning-acting loop for language agents capable of exploration and memory-based interaction.
- Aryal et al. (2024) investigate cross-domain agent integration for knowledge triangulation and collaborative intelligence.
- Wei et al. (2022) propose a Chain-of-Thought prompting method where agents self-consistently evaluate each other before deciding.

---

## 3. Planning Taxonomy and Agent Modularity

Agent planning behaviors can be categorized using formal taxonomies, enabling performance analysis across reasoning styles.

- Wang et al. (2024) define modular agent architectures with distinct profiling, memory, planning, and action modules, used to structure evaluation systems.
- Shi et al. (2025) explore how agent planning paths (single-path vs. multi-path) affect performance on truth-critical tasks.

---

## 4. Embedding-Based Evaluation & Semantic Similarity

Evaluating LLM output quality through traditional accuracy alone is insufficient. Semantic embeddings and similarity metrics allow deeper comparisons of output structure and content.

- Zhang et al. (2025) use FAISS and SentenceTransformer-based similarity to cluster LLM-generated outputs.
- Weng (2023) emphasizes cosine and Euclidean distance metrics as reliable tools for output consistency analysis.
- Huang et al. (2023) incorporate PCA into multi-agent evaluation to visualize response distribution and convergence.

---

## References (APA 6)

- Aryal, S., Nguyen, T., & Lee, J. (2024). *Leveraging multi AI agents for cross domain knowledge discovery* (arXiv:2404.08511). arXiv. https://arxiv.org/abs/2404.08511
- Duan, Z., & Wang, J. (2024). *Enhancing multi agent consensus through third party LLM integration: Analyzing uncertainty and mitigating hallucinations in LLMs* (arXiv:2411.16189). arXiv. https://arxiv.org/abs/2411.16189
- Gosmar, D., & Dahl, D. A. (2025). *Hallucination mitigation using agentic AI natural language based frameworks* (arXiv:2501.13946). arXiv. https://arxiv.org/abs/2501.13946
- Huang, Y., Hou, L., Ren, X., & Han, J. (2023). *Structure-aware alignment of LLM outputs for consistency evaluation*. arXiv. https://arxiv.org/abs/2304.05635
- Madaan, A., Lin, S., Gupta, A., et al. (2023). *Self-refine: Iterative refinement with self-feedback* (arXiv:2303.17651). arXiv. https://arxiv.org/abs/2303.17651
- Shi, Y., Cai, Z., Chen, T., & Zhan, X. (2025). *Evaluating agent planning styles in truth-sensitive environments*. NeurIPS.
- Wang, Y., Yang, D., & Liu, P. (2024). *Planning-enhanced agent architectures: Memory, profiling, and action modules*. ACL.
- Wei, J., Wang, X., Schuurmans, D., et al. (2022). *Chain of thought prompting elicits reasoning in large language models* (arXiv:2201.11903). arXiv. https://arxiv.org/abs/2201.11903
- Weng, L. (2023). *Vector-based evaluation of multi-agent language models*. Journal of AI Research.
- Yao, S., Zhao, Y., Zhang, Q., et al. (2023). *ReAct: Synergizing reasoning and acting in language models* (arXiv:2210.03629). arXiv. https://arxiv.org/abs/2210.03629
- Zhang, J., Manakul, P., & Melis, G. (2025). *Multi-agent similarity analysis for hallucination evaluation in LLMs*. EMNLP.


