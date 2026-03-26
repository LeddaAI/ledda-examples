# LangGraph + Ledda

A 3-phase research pipeline instrumented with [Ledda](https://ledda.ai) for LLM observability.

The agent researches a topic, analyzes findings, and writes an executive summary. Every LLM call, graph node, and tool invocation is automatically traced.

## Setup

```bash
cp .env.example .env
# Fill in LEDDA_API_KEY and OPENROUTER_API_KEY

pip install -r requirements.txt
python main.py
```

Traces appear in your Ledda dashboard within seconds.

## What gets traced

- `LangGraph.workflow` — the full pipeline execution
- `research.task`, `analyze.task`, `summarize.task` — each graph node
- `ChatOpenAI.chat` — each LLM call with model, tokens, messages

## How it works

`ledda_init.py` sets up OpenTelemetry with the Ledda exporter and LangChain auto-instrumentation. Import it **before** any LangChain/LangGraph code:

```python
from ledda_init import init_ledda
provider = init_ledda("my-app")

# Now import LangChain/LangGraph
from langchain_openai import ChatOpenAI
```
