# Information Model for LLM Agent Architecture

This repository contains a local multi-model evaluation framework for checking which local LLM is best suited for a task under different agent architectures.

The workflow runs the same task set through multiple agent strategies and local Ollama models, stores responses, embeds them into a shared semantic space, and compares model behavior by task cluster, consensus, and task/category fit.

## Methodology

![Local model selection framework](docs/local_model_selection_framework.png)

The framework combines:

- **Task matrix**: task/domain examples tested against local model candidates.
- **Agent architecture matrix**: planning, memory, action tools, skill libraries, and source/RAG coverage.
- **Response evaluation**: semantic embeddings, centroid similarity, response quality, task fit, and latency/cost.
- **Recommendation output**: ranked best local model, fallback model, and strongest agent strategy for the task.

## Main Files

- `agents4pfinal.ipynb` — main notebook for running agents, storing responses, and generating analysis.
- `agents.csv` — agent architecture metadata.
- `tasks.csv` — task set used for model/agent comparison.
- `results/` — generated CSVs and visual analysis outputs.
- `docs/local_model_selection_framework.png` — methodology diagram.

## Live Demo

The interactive response-space visualization is hosted here:

[MultiAgent LLM Response Space Explorer](https://huggingface.co/spaces/gauraang/MultiAgent)

## Privacy Note

Local secrets and runtime files are intentionally excluded through `.gitignore`, including `.env`, virtual environments, logs, local database files, and embedded third-party Git repositories.
