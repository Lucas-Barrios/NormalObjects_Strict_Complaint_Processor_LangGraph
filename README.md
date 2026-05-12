# NormalObjects — Strict Complaint Processor

A multi-step AI complaint-processing workflow built with **LangGraph**, **FastAPI**, and **GPT-4o-mini**.  
Implements *Bloyce's Protocol* — a fictional Stranger Things-themed incident management system — featuring parallel investigation, human-in-the-loop approval, and SQLite-backed persistence.

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/Lucas-Barrios/NormalObjects_Strict_Complaint_Processor_LangGraph.git
cd NormalObjects_Strict_Complaint_Processor_LangGraph

pip install langchain-openai langgraph langgraph-checkpoint-sqlite \
            fastapi uvicorn python-dotenv
```

### 2. Set your OpenAI API key

Create a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

### 3. Run the web UI

```bash
uvicorn web_server:api --reload --port 8000
```

Open **http://localhost:8000** in your browser.

> Drop `--reload` for stable long-running sessions — a hot-reload during an active complaint run will lose its in-memory log stream.

### 4. Run the CLI batch demo (optional)

```bash
python normalobjects_langgraph.py
```

Processes five sample complaints end-to-end in the terminal, including an interactive human-approval prompt.

---

## How It Works

Complaints travel through a directed graph of five stages:

```
[intake] → [validate] → [investigate ×N] → [merge] → [resolve] → [close]
                                                            │
                                               (medium/low) ▼
                                           [human_approval] ──► [close]
```

| Stage | What happens |
|---|---|
| **Intake** | LLM categorises the complaint and extracts who/what/when/where |
| **Validate** | Checks the complaint against category-specific rules |
| **Investigate** | 2–3 parallel sub-agents examine different angles simultaneously |
| **Merge** | Fan-in node synthesises parallel findings into one report |
| **Resolve** | Selects and applies the appropriate Downside Up protocol |
| **Human Approval** | Pauses for sign-off when effectiveness rating is medium or low |
| **Close** | Records customer satisfaction, timestamps closure, flags follow-ups |

Complaints can exit early as `rejected`, `escalated`, `needs_clarification`, or `failed`.

---

## File Map

```
.
├── normalobjects_langgraph.py   # Core LangGraph workflow — all nodes, edges,
│                                #   state schema, retry logic, persistence helpers,
│                                #   and CLI batch runner (__main__)
│
├── web_server.py                # FastAPI server
│                                #   • POST /api/complaints          submit a complaint
│                                #   • POST /api/complaints/{id}/decide  HITL decision
│                                #   • GET  /api/complaints          list all records
│                                #   • GET  /api/complaints/{id}     full state + HITL data
│                                #   • GET  /api/complaints/{id}/logs  SSE live log stream
│                                #   • GET  /                        serves the web UI
│
├── static/
│   └── index.html               # Single-page frontend (Tailwind + Alpine.js)
│                                #   Sidebar complaint list · Submit form · Detail pane
│                                #   Live log terminal · HITL approval panel
│
├── complaints.db                # SQLite database (auto-created on first run)
│                                #   LangGraph checkpoint tables + complaint_summaries audit table
│
├── workflow_graph.md            # Mermaid diagram of the workflow (auto-generated)
├── lab_summary.md               # LangGraph vs LangChain comparison write-up
├── .env                         # OPENAI_API_KEY (not committed)
└── .gitignore
```

---

## Web UI Features

- **Submit complaints** with complainant name, description, and optional custom ID
- **Live log terminal** — color-coded output streams in real time via Server-Sent Events
- **Status badges** — Processing · Closed · Rejected · Escalated · Needs Info · Needs Approval
- **HITL approval panel** — appears automatically for medium/low effectiveness resolutions; supports approve, reject, or free-text override
- **Persistent history** — all completed complaints survive server restarts (SQLite)
- **Auto-refresh** — sidebar polls every 3 seconds for status updates

---

## Key Technologies

| Library | Role |
|---|---|
| `langgraph` | Stateful workflow graph, parallel fan-out via `Send`, `interrupt()` for HITL |
| `langgraph-checkpoint-sqlite` | Durable state persistence across process restarts |
| `langchain-openai` | GPT-4o-mini LLM calls inside each node |
| `fastapi` + `uvicorn` | REST API and SSE log streaming |
| Tailwind CSS + Alpine.js | Reactive single-page frontend (CDN, no build step) |
