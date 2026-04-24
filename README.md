# Incident Investigation Copilot

A multi-agent AI system for automated incident root-cause analysis. Given a natural-language question about an incident, a pipeline of specialized agents (planner, log analysis, hypothesis, timeline, critic, memory, report) produces a structured investigation report with root cause, timeline, affected services, and evidence.

Supports both a CLI and an interactive Streamlit dashboard.

## Architecture

```
User query
    └─► Orchestrator
            ├── Planner       — breaks the query into investigation steps
            ├── Log Analysis  — scans logs for anomalies and patterns
            ├── Hypothesis    — generates and ranks root-cause hypotheses
            ├── Timeline      — reconstructs the incident timeline
            ├── Graph Agent   — queries Neo4j knowledge graph (optional)
            ├── Memory Agent  — recalls past incidents from persistent store
            ├── Critic        — validates and scores the investigation
            └── Report        — compiles the final structured report
```

- **LLM:** Groq (fast inference)
- **RAG:** Hybrid BM25 + FAISS vector search over log/incident data
- **Knowledge Graph:** Neo4j (optional — gracefully disabled if not configured)
- **UI:** Streamlit dashboard with live streaming and Plotly charts

## Environment Setup

Create a `.env` file in the project root:

```env
# Required
GROQ_API_KEY=your_groq_api_key

# Optional — enables knowledge graph queries
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Installation

Requires Python 3.10+.

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd incident_investigation_copilot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env           # then fill in your keys
```

## Running

### Streamlit Dashboard (recommended)

```bash
streamlit run dashboard.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### CLI — single query

```bash
python -m agents.main "Why did auth-service fail on 2024-01-15?"
```

### CLI — interactive mode

```bash
python -m agents.main --interactive
```

## Data

Place raw log files or incident JSON in the `data/` directory. The RAG pipeline indexes them automatically on first run. The memory store persists to `data/incident_memory.json`.
