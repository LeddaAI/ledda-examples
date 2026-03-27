"""
FIXED: LangGraph + Python threads — traces are correctly captured.

This script applies the two fixes for OTEL + threads:

1. contextvars.copy_context() captures the active OTEL context (parent span,
   trace ID, etc.) and ctx.run() propagates it into the worker thread.

2. force_flush() on the TracerProvider is called BEFORE executor.shutdown()
   to ensure BatchSpanProcessor exports all spans.

Run this script and verify: all spans appear in Ledda, correctly linked.

Usage:
    cp .env.example .env   # fill in your API keys
    LEDDA_DEBUG=1 python fixed.py
"""

# --- OTEL init (MUST happen before LangChain imports) ---
import uuid

from ledda_init import init_ledda, ledda_flush

SESSION_ID = f"session-{uuid.uuid4().hex[:8]}"

provider = init_ledda(
    service_name="langgraph-threads-fixed",
    tenant_id="acme-corp",
    tenant_name="Acme Corporation",
    session_id=SESSION_ID,
    workflow_name="threaded-pipeline-fixed",
)

# --- LangGraph agent ---
import contextvars
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from opentelemetry import trace

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[list, add_messages]
    topic: str
    research: str
    analysis: str
    summary: str


llm = ChatOpenAI(
    model="openai/gpt-5.4-nano",
    openai_api_key=os.environ["OPENROUTER_API_KEY"],
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)


# ---------------------------------------------------------------------------
# Fixed helper: propagates OTEL context and flushes before shutdown.
# ---------------------------------------------------------------------------
def run_in_thread(func, timeout=30):
    """Run func in a thread with a timeout, preserving OTEL context.

    FIX 1: copy_context() captures the active span context so that spans
           created inside the thread are linked to the correct parent trace.
    FIX 2: force_flush() ensures all spans are exported before shutdown.
    """
    # FIX 1: Capture the current OTEL context (includes active span, trace ID)
    ctx = contextvars.copy_context()

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        # ctx.run() executes func with the captured context propagated
        future = executor.submit(ctx.run, func)
        result = future.result(timeout=timeout)

        # FIX 2: Flush spans BEFORE shutting down the executor
        trace.get_tracer_provider().force_flush(timeout_millis=5000)

        return result
    except FuturesTimeoutError:
        # Even on timeout, flush whatever spans were created
        trace.get_tracer_provider().force_flush(timeout_millis=2000)
        raise TimeoutError(f"Execution timed out after {timeout} seconds")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def research(state: State) -> State:
    """Phase 1: Gather key facts about the topic."""
    def _call():
        prompt = (
            f"You are a researcher. List 5 key facts about: {state['topic']}. "
            "Be concise, one line per fact."
        )
        return llm.invoke(
            [("system", "You are a thorough researcher."), ("user", prompt)]
        )

    response = run_in_thread(_call)
    return {"messages": [response], "research": response.content}


def analyze(state: State) -> State:
    """Phase 2: Analyze the research and identify patterns."""
    def _call():
        prompt = (
            f"Given these research facts:\n\n{state['research']}\n\n"
            "Identify 3 interesting patterns or insights. Be concise."
        )
        return llm.invoke(
            [("system", "You are an analytical thinker."), ("user", prompt)]
        )

    response = run_in_thread(_call)
    return {"messages": [response], "analysis": response.content}


def summarize(state: State) -> State:
    """Phase 3: Create a final executive summary."""
    def _call():
        prompt = (
            f"Based on this research:\n{state['research']}\n\n"
            f"And this analysis:\n{state['analysis']}\n\n"
            "Write a 2-sentence executive summary that captures the essence."
        )
        return llm.invoke(
            [("system", "You write crisp executive summaries."), ("user", prompt)]
        )

    response = run_in_thread(_call)
    return {"messages": [response], "summary": response.content}


# Build graph: research -> analyze -> summarize
graph_builder = StateGraph(State)
graph_builder.add_node("research", research)
graph_builder.add_node("analyze", analyze)
graph_builder.add_node("summarize", summarize)
graph_builder.add_edge(START, "research")
graph_builder.add_edge("research", "analyze")
graph_builder.add_edge("analyze", "summarize")
graph_builder.add_edge("summarize", END)
graph = graph_builder.compile()

# --- Run ---
TOPIC = "the impact of large language models on software engineering"

print(f"Running FIXED threaded pipeline on: '{TOPIC}'")
print(f"  Tenant: acme-corp | Session: {SESSION_ID}")
print("  Phase 1: Research -> Phase 2: Analyze -> Phase 3: Summarize")
print("  (Each phase runs LLM calls inside a ThreadPoolExecutor)\n")

result = graph.invoke(
    {"messages": [], "topic": TOPIC, "research": "", "analysis": "", "summary": ""}
)

print("=" * 60)
print("RESEARCH:")
print(result["research"])
print("\n" + "=" * 60)
print("ANALYSIS:")
print(result["analysis"])
print("\n" + "=" * 60)
print("SUMMARY:")
print(result["summary"])

# Final flush — all spans should already be exported, but this catches any stragglers
ledda_flush(provider)
print("\nTraces sent to Ledda.")
