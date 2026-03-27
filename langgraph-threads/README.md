# LangGraph + Threads + Ledda

Demonstrates a classic OTEL + Python threads problem and its fix.

When running LLM calls inside `ThreadPoolExecutor` (common for timeout guards), OpenTelemetry traces can silently vanish. This example shows exactly why and how to fix it.

## The Problem

Two bugs that cause trace loss:

1. **No context propagation** — `ThreadPoolExecutor.submit()` doesn't propagate `contextvars` to worker threads. OTEL stores the active span/trace in contextvars, so spans created in the thread are orphaned.

2. **Premature shutdown** — `executor.shutdown(wait=False)` kills the thread before `BatchSpanProcessor` can flush. Spans are silently discarded.

## The Fix

```python
import contextvars
from opentelemetry import trace

# FIX 1: Capture OTEL context before submitting
ctx = contextvars.copy_context()
future = executor.submit(ctx.run, func)
result = future.result(timeout=timeout)

# FIX 2: Flush spans before shutdown
trace.get_tracer_provider().force_flush(timeout_millis=5000)
```

## Setup

```bash
cp .env.example .env
# Fill in LEDDA_API_KEY and OPENROUTER_API_KEY

pip install -r requirements.txt
```

## Run

```bash
# Part 1a: Broken — LLM calls in threads (fragmented traces)
LEDDA_DEBUG=1 python broken.py

# Part 1b: Broken — entire graph in a thread (traces completely lost)
LEDDA_DEBUG=1 python broken-graph.py

# Part 2: Fixed — threads with proper context + flush (all spans captured)
LEDDA_DEBUG=1 python fixed.py
```

## Files

| File | Description |
|------|-------------|
| `broken.py` | LLM calls in threads — traces are fragmented across multiple trace IDs |
| `broken-graph.py` | Entire graph (build + compile + invoke) in a thread — traces completely lost |
| `fixed.py` | Threads with `copy_context()` + `force_flush()` — traces arrive correctly |
| `ledda_init.py` | Shared OTEL/Ledda initialization |
