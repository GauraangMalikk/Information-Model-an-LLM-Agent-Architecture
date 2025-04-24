# Methods

This project implements a modular architecture for evaluating hallucination reduction strategies in multi-agent LLM systems. The methods are divided into eight interconnected blocks (Blocks A–H), each representing a distinct component of the system pipeline — from agent design and response storage to PCA-driven performance analysis and complexity-based evaluation.

---

## Block A: AI Agent Architecture – Planning, Storage, and Retrieval

### A.1. Import Libraries & Setup
- Core libraries for semantic search, PostgreSQL, FAISS, and LLM APIs are imported.
- A `SYSTEM_PROMPT` guides all agent responses to be factual and concise.
- `clean_response()` function ensures consistent formatting.

### A.2. Database Configuration
- A PostgreSQL URI is defined via `DB_CONFIG`.
- SQLAlchemy and psycopg2 are used for all DB communication.

### A.3. Planning Taxonomy
- `PLANNING_AGENTS` defines categories (e.g., single-path, multi-path).
- `populate_planning_taxonomy()` inserts this structure into the database.

### A.4. Agent Initialization
- `insert_agents_from_csv()` reads agent metadata from `agents.csv`, mapping planning subcategories and capabilities.

### A.5. Response Storage
- `store_agent_response()` saves generated responses under each agent’s record in `generated_response`.
- Duplicates are avoided through conditional updates.

### A.6. API & Agent Wrappers
- Agents call TinyLLaMA via POST requests, with wrappers like `run_zero_shot_cot` modifying the prompts.
- Each agent’s output is stored and tagged by planning strategy.

### A.7. Processing and Execution
- `process_agents()` orchestrates the full pipeline:
  - DB setup
  - Agent execution
  - Response storage

### A.8. Semantic Search
- Agent responses are encoded using SentenceTransformer (`all-MiniLM-L6-v2`).
- FAISS indexing enables top-k similarity retrieval.

---

## Block B: Task-Level Response Embedding and Similarity Analysis

### B.1. Data Retrieval
- Fetches all agent responses from the DB using an indexed PostgreSQL query.

### B.2. Embedding & Grouping
- Embeds responses by task iteration (`idx`).
- Builds separate FAISS indices for each batch of agent responses.

### B.3. Similarity Metrics
- Computes pairwise cosine similarity and Euclidean distance.
- Categorizes relationships into converging, diverging, or partially aligned.

### B.4. Task Metrics
- Measures average distance, similarity, response length, and processing time per batch.
- Stored using `save_task_performance()`.

---

## Block C: PCA-Based Task Matrix Construction

### C.1. Global Embedding and Reduction
- All responses embedded and globally scaled.
- PCA reduces dimensionality for interpretability.

### C.2. Task Matrix
- Constructs a task-agent matrix with total scores (abs PCA vector sum).
- Filters zero vectors.

### C.3. Visualization
- Heatmaps for Euclidean and cosine metrics.
- Stored PCA results exported as CSV.

---

## Block D: Heatmap Visualization of Pair Performance

### D.1. Filtering and Pivoting
- Filters agent_pair_performance by `task_id`.
- Pivots Euclidean and cosine matrices.

### D.2. Visualization
- Matplotlib + Seaborn used for dual heatmaps.
- Shows agent-to-agent distance/similarity per task.

---

## Block E: Agent-Level Aggregated Metrics

### E.1. Metric Averaging
- Merges agent1 and agent2 scores across all tasks.
- Calculates mean Euclidean distance and cosine similarity per agent.

### E.2. Scatter Plotting
- Shows per-agent trends across tasks using scatter plots.

---

## Block F: KMeans-Based Clustering of Pair Performance

### F.1. Clustering Preparation
- Selects 4 features: ED, CosSim, Response Length, Completion Time.
- StandardScaler normalizes data.

### F.2. KMeans + PCA
- Clusters pairs into 3 groups (`k=3`).
- PCA visualizes the results in 2D space.

### F.3. Annotated Plot
- Each point is an agent pair.
- Colored by cluster, labeled by names.

---

## Block G: Planning Taxonomy–Aware Clustering

### G.1. Subcategory Mapping
- Maps agents to planning subcategories.
- Creates `pair_category` for combinations (e.g., "Single–Multi").

### G.2. Clustering + Visualization
- KMeans (k=5) on scaled features.
- PCA + Seaborn scatter with cluster and pair_category labels.

---

## Block H: Complexity-Based Agent Selection

### H.1. Complexity Ratings
- Agents are scored manually via a `complexity_mapping`.

### H.2. Combined Performance Metric
- Combines ED + CosSim into a joint metric.
- Aggregated per agent across all pairs.

### H.3. Best Agent Identification
- Sorts agents by complexity and performance to find the best.
- Scatter plot shows complexity vs. combined score.

---

## Conclusion

This modular architecture enables deep analysis of multi-agent LLM behavior using semantic embeddings, structured reasoning taxonomy, and multi-layered evaluation metrics. It supports empirical grounding of agent performance in terms of consistency, accuracy, stability, and planning strategy effectiveness.
