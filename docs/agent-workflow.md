# Agent Workflow Diagram

```mermaid
flowchart TD
    subgraph Orchestrator["🎯 Orchestrator (manages entire flow)"]

        DC[📊 Data Collector]

        subgraph Parallel["⚡ Parallel Execution"]
            FA[📈 Fundamental Analyst]
            TA[📉 Technical Analyst]
            SA[💬 Sentiment Analyst]
        end

        DB[⚖️ Debate Agent]
        RM[🛡️ Risk Manager]
        SY[📝 Synthesis Agent]
        FL[🔄 Feedback Loop]
    end

    DC --> FA
    DC --> TA
    DC --> SA

    FA --> DB
    TA --> DB
    SA --> DB

    DB --> RM
    RM --> SY
    SY --> FL

    FL --> |"Quality OK"| Output[📄 Final Report]
    FL -.-> |"Needs More Data"| DC

    style DC fill:#e1f5fe
    style FA fill:#fff3e0
    style TA fill:#fff3e0
    style SA fill:#fff3e0
    style DB fill:#f3e5f5
    style RM fill:#ffebee
    style SY fill:#e8f5e9
    style FL fill:#fce4ec
    style Output fill:#c8e6c9
```

## Agent Execution Order

| Step | Agent | Description |
|------|-------|-------------|
| 1 | Data Collector | Gathers stock data from Yahoo Finance |
| 2-4 | Fundamental, Technical, Sentiment Analysts | Run in **parallel** |
| 5 | Debate Agent | Creates bull/bear cases |
| 6 | Risk Manager | Assesses risks |
| 7 | Synthesis Agent | Generates final report |
| 8 | Feedback Loop | Quality assurance |
