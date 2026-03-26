"""
LangGraph + Ledda example: 3-phase research pipeline.

A multi-step agent that researches a topic, analyzes findings, and writes
an executive summary. Each step and LLM call is automatically traced
and visible in your Ledda dashboard.

Demonstrates:
- Auto-instrumented LLM calls (model, tokens, messages)
- Tenant routing (group traces by customer)
- Session tracking (link related conversations)
- Workflow naming (label traces in the dashboard)

Usage:
    cp .env.example .env   # fill in your API keys
    pip install -r requirements.txt
    python main.py
"""

# --- OTEL init (MUST happen before LangChain imports) ---
import uuid

from ledda_init import init_ledda, ledda_flush

SESSION_ID = f"session-{uuid.uuid4().hex[:8]}"

provider = init_ledda(
    service_name="langgraph-research-pipeline",
    tenant_id="acme-corp",
    tenant_name="Acme Corporation",
    session_id=SESSION_ID,
    workflow_name="research-pipeline",
)

# --- LangGraph agent ---
import os

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

print(f"Running 3-phase research pipeline on: '{TOPIC}'")
print(f"  Tenant: acme-corp | Session: {SESSION_ID}")
print("  Phase 1: Research -> Phase 2: Analyze -> Phase 3: Summarize\n")

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

# Flush traces before exit (set LEDDA_DEBUG=1 to see span details)
ledda_flush(provider)
print("\nTraces sent to Ledda.")
