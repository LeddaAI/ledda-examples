"""
Ledda OTEL initialization for LangGraph/LangChain apps.

Call init_ledda() BEFORE importing any LangChain/LangGraph modules.
"""

import os
from contextlib import contextmanager

from dotenv import load_dotenv
from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

_tracer = trace.get_tracer("ledda")


def init_ledda(service_name: str = "langgraph-app") -> TracerProvider:
    """Initialize OpenTelemetry with Ledda exporter and LangChain auto-instrumentation."""
    load_dotenv()

    endpoint = os.environ.get("LEDDA_OTLP_ENDPOINT", "https://otlp.ledda.ai")
    api_key = os.environ["LEDDA_API_KEY"]

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        SimpleSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{endpoint}/v1/traces",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        )
    )
    trace.set_tracer_provider(provider)
    LangchainInstrumentor().instrument()

    return provider


@contextmanager
def ledda_trace(name: str, tenant_id: str = "", tenant_name: str = "",
                session_id: str = "", workflow_name: str = ""):
    """Wrap your pipeline in a root span with Ledda routing attributes.

    These attributes control how traces appear in Ledda:
    - tenant_id / tenant_name: group traces by customer or team
    - session_id: link traces in the same conversation
    - workflow_name: label the trace in the dashboard

    Usage:
        with ledda_trace("my-pipeline", tenant_id="acme", session_id="s-123"):
            graph.invoke(...)
    """
    with _tracer.start_as_current_span(name) as span:
        if tenant_id:
            span.set_attribute("ledda.association.properties.tenant_id", tenant_id)
        if tenant_name:
            span.set_attribute("ledda.association.properties.tenant_name", tenant_name)
        if session_id:
            span.set_attribute("ledda.association.properties.session_id", session_id)
        if workflow_name:
            span.set_attribute("ledda.workflow.name", workflow_name)
        yield span
