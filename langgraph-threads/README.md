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
# Part 1: See the broken version (no spans captured)
LEDDA_DEBUG=1 python broken.py

# Part 2: See the fixed version (all spans captured)
LEDDA_DEBUG=1 python fixed.py
```

## Files

| File | Description |
|------|-------------|
| `broken.py` | Threads without context propagation or flush — traces are lost |
| `fixed.py` | Threads with `copy_context()` + `force_flush()` — traces arrive correctly |
| `ledda_init.py` | Shared OTEL/Ledda initialization |
