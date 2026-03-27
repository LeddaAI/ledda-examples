"""
BROKEN: Entire LangGraph pipeline runs inside a thread — traces are lost.

A more extreme version of the threading problem: the graph build, compile,
and invoke all happen inside a ThreadPoolExecutor. This is common when an
application wraps the entire agent execution in a thread with a timeout
guard (e.g., a web server handler or a task runner).

Since the whole graph runs in a thread with no OTEL context and no flush,
zero spans make it to Ledda.

Usage:
    cp .env.example .env   # fill in your API keys
    LEDDA_DEBUG=1 python broken-graph.py
"""

# --- OTEL init (MUST happen before LangChain imports) ---
import uuid

from ledda_init import init_ledda, ledda_flush

SESSION_ID = f"session-{uuid.uuid4().hex[:8]}"

provider = init_ledda(
    service_name="langgraph-threads-broken-graph",
    tenant_id="acme-corp",
    tenant_name="Acme Corporation",
    session_id=SESSION_ID,
    workflow_name="threaded-graph-broken",
)

# --- LangGraph agent (built and run entirely inside a thread) ---
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


def research(state: State) -> State:
    """Phase 1: Gather key facts about the topic."""
    prompt = (
        f"You are a researcher. List 5 key facts about: {state['topic']}. "
        "Be concise, one line per fact."
    )
    response = llm.invoke(
        [("system", "You are a thorough researcher."), ("user", prompt)]
    )
    return {"messages": [response], "research": response.content}


def analyze(state: State) -> State:
    """Phase 2: Analyze the research and identify patterns."""
    prompt = (
        f"Given these research facts:\n\n{state['research']}\n\n"
        "Identify 3 interesting patterns or insights. Be concise."
    )
    response = llm.invoke(
        [("system", "You are an analytical thinker."), ("user", prompt)]
    )
    return {"messages": [response], "analysis": response.content}


def summarize(state: State) -> State:
    """Phase 3: Create a final executive summary."""
    prompt = (
        f"Based on this research:\n{state['research']}\n\n"
        f"And this analysis:\n{state['analysis']}\n\n"
        "Write a 2-sentence executive summary that captures the essence."
    )
    response = llm.invoke(
        [("system", "You write crisp executive summaries."), ("user", prompt)]
    )
    return {"messages": [response], "summary": response.content}


# ---------------------------------------------------------------------------
# The entire graph is built, compiled, and invoked inside a single thread.
# This mimics a web handler or task runner that wraps the whole agent call
# in a timeout-guarded thread.
# ---------------------------------------------------------------------------
def run_pipeline(topic: str) -> dict:
    """Build, compile, and run the full LangGraph pipeline."""
    graph_builder = StateGraph(State)
    graph_builder.add_node("research", research)
    graph_builder.add_node("analyze", analyze)
    graph_builder.add_node("summarize", summarize)
    graph_builder.add_edge(START, "research")
    graph_builder.add_edge("research", "analyze")
    graph_builder.add_edge("analyze", "summarize")
    graph_builder.add_edge("summarize", END)
    graph = graph_builder.compile()

    return graph.invoke(
        {"messages": [], "topic": topic, "research": "", "analysis": "", "summary": ""}
    )


# --- Run ---
TOPIC = "the impact of large language models on software engineering"

print(f"Running BROKEN threaded graph on: '{TOPIC}'")
print(f"  Tenant: acme-corp | Session: {SESSION_ID}")
print("  The ENTIRE graph (build + compile + invoke) runs in a thread")
print("  No OTEL context propagation, no flush before shutdown\n")

executor = ThreadPoolExecutor(max_workers=1)
try:
    # BUG 1: No context propagation — the thread has empty contextvars
    future = executor.submit(run_pipeline, TOPIC)
    result = future.result(timeout=120)
except FuturesTimeoutError:
    print("Pipeline timed out!")
    exit(1)
finally:
    # BUG 2: Kills everything before BatchSpanProcessor can export
    executor.shutdown(wait=False, cancel_futures=True)

print("=" * 60)
print("RESEARCH:")
print(result["research"])
print("\n" + "=" * 60)
print("ANALYSIS:")
print(result["analysis"])
print("\n" + "=" * 60)
print("SUMMARY:")
print(result["summary"])

# Flush — but spans created in the thread were in a disconnected context,
# and the aggressive shutdown may have already discarded pending exports.
ledda_flush(provider)
print("\n[broken-graph] The pipeline ran fine, but check Ledda: traces are lost!")
print("  - The entire graph ran in a thread with no OTEL context")
print("  - shutdown(wait=False) discarded pending span exports")
