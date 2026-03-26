"""
Ledda OTEL initialization for LangGraph/LangChain apps.

Call init_ledda() BEFORE importing any LangChain/LangGraph modules.
"""

import os

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


class LeddaAttributeProcessor(SpanProcessor):
    """Injects Ledda routing attributes onto every span.

    This ensures tenant, session, and workflow info is present on all spans
    regardless of which batch they arrive in.
    """

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


def init_ledda(
    service_name: str = "langgraph-app",
    tenant_id: str = "",
    tenant_name: str = "",
    session_id: str = "",
    workflow_name: str = "",
) -> TracerProvider:
    """Initialize OpenTelemetry with Ledda exporter and LangChain auto-instrumentation.

    Args:
        service_name: Identifies your app in traces.
        tenant_id: Group traces by customer/team in Ledda.
        tenant_name: Display name for the tenant.
        session_id: Link traces in the same conversation.
        workflow_name: Label this trace type in the dashboard.
    """
    load_dotenv()

    endpoint = os.environ.get("LEDDA_OTLP_ENDPOINT", "https://otlp.ledda.ai")
    api_key = os.environ["LEDDA_API_KEY"]

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    # Inject Ledda routing attributes on every span
    if any([tenant_id, tenant_name, session_id, workflow_name]):
        provider.add_span_processor(LeddaAttributeProcessor(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            session_id=session_id,
            workflow_name=workflow_name,
        ))

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
