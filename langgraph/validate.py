"""
Validate that OTEL traces are being captured and exported to Ledda.

Sends a single lightweight LLM call and reports exactly what was captured:
span count, trace ID, attributes, and export status.

Usage:
    python validate.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Check env vars before doing anything expensive
missing = [v for v in ("LEDDA_API_KEY", "OPENROUTER_API_KEY") if not os.environ.get(v)]
if missing:
    print(f"FAIL: Missing environment variables: {', '.join(missing)}")
    print("     Copy .env.example to .env and fill in your keys.")
    sys.exit(1)

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan, SpanProcessor, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class SpanCollector(SpanProcessor):
    """Collects finished spans for inspection."""

    def __init__(self):
        self.spans = []

    def on_start(self, span: Span, parent_context=None) -> None:
        pass

    def on_end(self, span: ReadableSpan) -> None:
        self.spans.append(span)

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# Setup
endpoint = os.environ.get("LEDDA_OTLP_ENDPOINT", "https://otlp.ledda.ai")
api_key = os.environ["LEDDA_API_KEY"]

resource = Resource.create({"service.name": "ledda-validate"})
provider = TracerProvider(resource=resource)

collector = SpanCollector()
provider.add_span_processor(collector)

exporter = OTLPSpanExporter(
    endpoint=f"{endpoint}/v1/traces",
    headers={"Authorization": f"Bearer {api_key}"},
)
provider.add_span_processor(SimpleSpanProcessor(exporter))

trace.set_tracer_provider(provider)

print("1. Instrumenting LangChain...", end=" ")
try:
    LangchainInstrumentor().instrument()
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("2. Sending a single LLM call...", end=" ", flush=True)
try:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="anthropic/claude-sonnet-4",
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=10,
    )
    response = llm.invoke([("user", "Say OK")])
    print(f"OK (response: {response.content.strip()!r})")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("3. Flushing traces...", end=" ")
success = provider.force_flush(timeout_millis=10000)
print(f"{'OK' if success else 'FAIL (timeout)'}")

# Report
print(f"\n{'=' * 50}")
print(f"Spans captured: {len(collector.spans)}")

if not collector.spans:
    print("\nFAIL: No spans were created.")
    print("  - Is 'langchain' package installed? (not just langchain-core)")
    print("  - Is 'langchain-community' package installed?")
    print("  - Was LangchainInstrumentor().instrument() called BEFORE imports?")
    sys.exit(1)

trace_ids = set()
for span in collector.spans:
    ctx = span.context
    tid = format(ctx.trace_id, "032x")
    sid = format(ctx.span_id, "016x")
    trace_ids.add(tid)

    attrs = dict(span.attributes) if span.attributes else {}
    model = attrs.get("gen_ai.request.model", "")
    provider_name = attrs.get("gen_ai.system", "")
    input_tok = attrs.get("gen_ai.usage.prompt_tokens", attrs.get("gen_ai.usage.input_tokens", ""))
    output_tok = attrs.get("gen_ai.usage.completion_tokens", attrs.get("gen_ai.usage.output_tokens", ""))

    detail = ""
    if model:
        detail = f" | model={model} provider={provider_name} tokens={input_tok}/{output_tok}"

    print(f"  [{span.name}] span={sid}{detail}")

print(f"\nTrace ID: {', '.join(trace_ids)}")
print(f"Endpoint: {endpoint}")
print(f"\nSUCCESS: {len(collector.spans)} span(s) captured and exported.")
