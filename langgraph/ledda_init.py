"""
Ledda OTEL initialization for LangGraph/LangChain apps.

Call init_ledda() BEFORE importing any LangChain/LangGraph modules.
"""

import os

from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


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
