"""
Ledda OTEL initialization for LangGraph/LangChain apps.

Call init_ledda() BEFORE importing any LangChain/LangGraph modules.
Set LEDDA_DEBUG=1 to print span details after flush.
"""

import os

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


class LeddaAttributeProcessor(SpanProcessor):
    """Injects Ledda routing attributes onto every span."""

    def __init__(self, tenant_id: str = "", tenant_name: str = "",
                 session_id: str = "", workflow_name: str = ""):
        self._attrs = {}
        if tenant_id:
            self._attrs["ledda.association.properties.tenant_id"] = tenant_id
        if tenant_name:
            self._attrs["ledda.association.properties.tenant_name"] = tenant_name
        if session_id:
            self._attrs["ledda.association.properties.session_id"] = session_id
        if workflow_name:
            self._attrs["ledda.workflow.name"] = workflow_name

    def on_start(self, span: Span, parent_context=None) -> None:
        for key, value in self._attrs.items():
            span.set_attribute(key, value)

    def on_end(self, span: ReadableSpan) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _SpanCollector(SpanProcessor):
    """Collects finished spans for debug reporting."""

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

    def report(self):
        if not self.spans:
            print("\n[ledda] WARNING: No spans captured!")
            print("  - Is 'langchain' installed? (not just langchain-core)")
            print("  - Is 'langchain-community' installed?")
            return

        trace_ids = set()
        print(f"\n[ledda] {len(self.spans)} span(s) captured:")
        for span in self.spans:
            ctx = span.context
            tid = format(ctx.trace_id, "032x")
            trace_ids.add(tid)
            attrs = dict(span.attributes) if span.attributes else {}
            model = attrs.get("gen_ai.request.model", "")
            tokens_in = attrs.get("gen_ai.usage.prompt_tokens",
                                  attrs.get("gen_ai.usage.input_tokens", ""))
            tokens_out = attrs.get("gen_ai.usage.completion_tokens",
                                   attrs.get("gen_ai.usage.output_tokens", ""))
            detail = f" model={model} tokens={tokens_in}/{tokens_out}" if model else ""
            print(f"  [{span.name}]{detail}")

        print(f"[ledda] Trace ID(s): {', '.join(trace_ids)}")


# Module-level collector, set when debug is enabled
_collector = None


def init_ledda(
    service_name: str = "langgraph-app",
    tenant_id: str = "",
    tenant_name: str = "",
    session_id: str = "",
    workflow_name: str = "",
) -> TracerProvider:
    """Initialize OpenTelemetry with Ledda exporter and LangChain auto-instrumentation."""
    global _collector
    load_dotenv()

    endpoint = os.environ.get("LEDDA_OTLP_ENDPOINT", "https://otlp.ledda.ai")
    api_key = os.environ["LEDDA_API_KEY"]
    debug = os.environ.get("LEDDA_DEBUG", "") in ("1", "true")

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if any([tenant_id, tenant_name, session_id, workflow_name]):
        provider.add_span_processor(LeddaAttributeProcessor(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            session_id=session_id,
            workflow_name=workflow_name,
        ))

    if debug:
        _collector = _SpanCollector()
        provider.add_span_processor(_collector)
        print(f"[ledda] Debug mode ON | endpoint={endpoint}")

    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{endpoint}/v1/traces",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        )
    )
    trace.set_tracer_provider(provider)
    LangchainInstrumentor().instrument()

    return provider


def ledda_flush(provider: TracerProvider):
    """Flush traces and print debug report if LEDDA_DEBUG is enabled."""
    provider.force_flush()
    if _collector:
        _collector.report()
