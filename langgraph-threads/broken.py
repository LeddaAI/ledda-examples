"""
BROKEN: LangGraph + Python threads — traces are lost.

This script demonstrates the two classic OTEL + threads pitfalls:

1. ThreadPoolExecutor.submit() does NOT propagate contextvars to worker
   threads. Spans created inside the thread become orphaned — they either
   start a new disconnected trace or are never linked to the parent.

2. executor.shutdown(wait=False) kills the process before BatchSpanProcessor
   has a chance to flush. Spans are silently discarded.

Run this script and notice: zero spans are captured by Ledda.

Usage:
    cp .env.example .env   # fill in your API keys
    LEDDA_DEBUG=1 python broken.py
"""

# --- OTEL init (MUST happen before LangChain imports) ---
import uuid

from ledda_init import init_ledda, ledda_flush

SESSION_ID = f"session-{uuid.uuid4().hex[:8]}"

provider = init_ledda(
    service_name="langgraph-threads-broken",
    tenant_id="acme-corp",
    tenant_name="Acme Corporation",
    session_id=SESSION_ID,
    workflow_name="threaded-pipeline-broken",
)

# --- LangGraph agent ---
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
# This helper mimics a real-world pattern: running LLM calls inside a thread
# with a timeout. This is common in production systems that need to guard
# against slow LLM responses.
# ---------------------------------------------------------------------------
def run_in_thread(func, timeout=30):
    """Run func in a thread with a timeout.

    BUG 1: No OTEL context propagation — spans inside `func` are orphaned.
    BUG 2: shutdown(wait=False) fires before BatchSpanProcessor flushes.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        # No context propagation! The worker thread has an empty contextvars.
        future = executor.submit(func)
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        raise TimeoutError(f"Execution timed out after {timeout} seconds")
    finally:
        # Kills everything immediately — pending span exports are discarded.
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

print(f"Running BROKEN threaded pipeline on: '{TOPIC}'")
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

# Flush — but it's too late, the spans were already lost in the threads.
ledda_flush(provider)
print("\n[broken] The pipeline ran fine, but check Ledda: no spans arrived!")
print("  - BUG 1: ThreadPoolExecutor didn't propagate OTEL context")
print("  - BUG 2: shutdown(wait=False) discarded pending span exports")
