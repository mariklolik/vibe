# Data Flow Diagram

Shows how data enters, moves through, and exits the Vibe pipeline.
What is stored, what is logged, and what stays in-memory only.

```mermaid
flowchart LR
    subgraph EXTERNAL [External Sources]
        ARXIV[(arXiv API)]
        HF[(HuggingFace API)]
        S2[(Semantic Scholar API)]
        ANTHROPIC[(Anthropic API\nvia proxy)]
    end

    subgraph PIPELINE [Vibe Pipeline - Local Process]
        direction TB

        subgraph RESEARCH_DATA [Research Stage Data]
            PAPERS[papers.json\ntitle, abstract, authors\ncitation count, year]
            IDEA[ideas/selected_idea.json\ntitle, description\nnovelty score]
        end

        subgraph EXPERIMENT_DATA [Experiment Stage Data]
            HYPO[experiments/hypotheses.json\nhypothesis_id, statement\npre-registered before run]
            SRC[src/\nmodel.py trainer.py\ndata_utils.py metrics.py]
            SCRIPTS[scripts/\nrun_main.py run_baselines.py\nrun_ablations.py]
            RESULTS[experiments/all_results.json\nmetrics per run\n+ SHA-256 hash]
            VERIFY[verification/\nh1.json h2.json ...\np-value, effect_size, PASS/FAIL]
        end

        subgraph PAPER_DATA [Writing Stage Data]
            PAPER_JSON[paper/paper.json\nsection texts, figures refs]
            MAIN_TEX[paper/main.tex\ncompiled LaTeX]
        end

        subgraph LOG_DATA [Observability]
            PROGRESS[progress.txt\ntimestamped append-only log\nSHA-256 hashes inline]
            PROJECT[project.json\nstage tracker for resume]
            REVIEW[review_1.json\nreview_2.json\nPASS/FAIL + reasons]
        end
    end

    subgraph OUTPUT [Researcher-Visible Output]
        FINAL[paper/main.tex\n+ paper/main.pdf]
        CODE[src/ + scripts/\nreproducible code]
        LOGS[progress.txt\naudit trail]
    end

    %% Inputs
    ARXIV -->|abstracts, metadata| PAPERS
    HF -->|abstracts, metadata| PAPERS
    S2 -->|abstracts, metadata| PAPERS
    ANTHROPIC -->|LLM responses| PIPELINE

    %% Internal flow
    PAPERS --> IDEA
    IDEA --> HYPO
    IDEA --> SRC
    SRC --> SCRIPTS
    SCRIPTS -->|subprocess execution| RESULTS
    RESULTS --> VERIFY
    RESULTS --> PAPER_JSON
    VERIFY --> PAPER_JSON
    PAPER_JSON --> MAIN_TEX

    %% Logging
    PAPERS -.->|fetch event| PROGRESS
    RESULTS -.->|hash + result| PROGRESS
    VERIFY -.->|p-value + verdict| PROGRESS
    MAIN_TEX -.->|write event| PROGRESS
    REVIEW -.->|verdict| PROGRESS

    %% Outputs
    MAIN_TEX --> FINAL
    SRC --> CODE
    SCRIPTS --> CODE
    PROGRESS --> LOGS

    %% What is NOT stored
    note1[NOT stored:\n- Full LLM prompts/responses\n- API keys\n- Raw paper PDFs\n- Env variables]
    style note1 fill:#333,color:#aaa,stroke:#555
```

## Data Classification

| Data | Classification | Storage | Transmitted to API? |
|------|---------------|---------|---------------------|
| Paper abstracts (from APIs) | Public | `context/papers.json` | Yes (in LLM prompts) |
| Research idea | Researcher's IP | `ideas/selected_idea.json` | Yes (in LLM prompts) |
| Generated code | Researcher's IP | `src/`, `scripts/` | Yes (in LLM prompts) |
| Experiment results | Researcher's IP | `experiments/` | Yes (in LLM prompts) |
| Paper sections | Researcher's IP | `paper/` | Yes (in LLM prompts) |
| Anthropic API key | Secret | Environment variable only | No (sent as HTTP header, not in prompt) |
| Pipeline log | Operational | `progress.txt` | No |
| SHA-256 hashes | Integrity | `progress.txt` inline | No |
