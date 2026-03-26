# Ledda Examples

Code examples for integrating [Ledda](https://ledda.ai) LLM observability into your applications.

## Examples

| Example | Framework | Description |
|---------|-----------|-------------|
| [langgraph/](langgraph/) | LangGraph / LangChain | 3-phase research pipeline with auto-instrumentation |

## Getting started

Each example has its own `README.md` with setup instructions. The general pattern is:

1. Create an API key at [app.ledda.ai](https://app.ledda.ai)
2. Copy `.env.example` to `.env` and fill in your keys
3. Install dependencies and run

All examples use OpenTelemetry for tracing — traces appear in your Ledda dashboard within seconds.
