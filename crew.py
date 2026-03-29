from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")
os.environ["OTEL_SERVICE_NAME"] = os.environ.get("OTEL_SERVICE_NAME_CREW", "crewai")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "true")
import litellm
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from opentelemetry import trace
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

SERVED = os.environ.get("MODEL_NAME", "texttinyllama")
GATEWAY = os.environ.get("GATEWAY_OPENAI_BASE") or os.environ.get("OPENAI_API_BASE")
if not GATEWAY:
    use_lb = os.environ.get("GATEWAY_USE_LOAD_BALANCER", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if use_lb:
        host = os.environ.get("GATEWAY_LB_HOST", "127.0.0.1")
        port = os.environ.get("GATEWAY_LB_PORT", "8780")
        GATEWAY = f"http://{host}:{port}/v1"
    else:
        host = os.environ.get("GATEWAY_HOST", "127.0.0.1")
        port = os.environ.get("GATEWAY_PORT", "8765")
        GATEWAY = f"http://{host}:{port}/v1"
if GATEWAY.rstrip("/").endswith("/v1"):
    pass
else:
    GATEWAY = GATEWAY.rstrip("/") + "/v1"


def _gateway_health_url() -> str:
    g = GATEWAY.rstrip("/")
    if g.endswith("/v1"):
        root = g[: -len("/v1")]
    else:
        root = g
    return root.rstrip("/") + "/health"

OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317"
)


def _otlp_traces_enabled() -> bool:
    v = os.environ.get("OTEL_TRACES_EXPORTER", "none").strip().lower()
    return v in ("otlp", "grpc", "1", "true", "yes")


def _crew_vllm_wait_seconds_raw() -> str:
    return os.environ.get("CREW_VLLM_WAIT_S", "240").strip()


def _crew_vllm_poll_seconds_raw() -> str:
    return os.environ.get("CREW_VLLM_POLL_S", "8").strip()


def _crew_llm_stream_enabled() -> bool:
    v = os.environ.get("CREW_LLM_STREAM", "true").strip().lower()
    return v not in ("0", "false", "no", "off")


def _exit_if_gateway_shows_bad_vllm_url() -> None:
    try:
        with urllib.request.urlopen(_gateway_health_url(), timeout=5) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return
    if body.get("status") == "misconfigured":
        warn = body.get("warning") or "Gateway reports VLLM_BASE_URL is misconfigured."
        dup = body.get("default_upstream", "")
        print(f"\nGateway /health: {warn}\n", file=sys.stderr)
        if dup and dup != "(unset)":
            print(f"  Current default_upstream from gateway: {dup!r}\n", file=sys.stderr)
        print(
            "  1. On Lambda: run `vllm serve` (README Step 6) and keep it running.\n"
            "  2. On your laptop: SSH tunnel Step 7 so vLLM is at http://127.0.0.1:8000\n"
            "  3. In .env set VLLM_BASE_URL=http://127.0.0.1:8000 (no trailing slash).\n"
            "  4. With the gateway running, save .env — it re-reads VLLM_BASE_URL on each request.\n"
            "Then run crew again.\n",
            file=sys.stderr,
        )
        sys.exit(2)


def _wait_for_vllm_via_gateway() -> None:
    raw = _crew_vllm_wait_seconds_raw().lower()
    if raw in ("0", "false", "no", "off"):
        return
    try:
        max_wait = float(raw)
    except ValueError:
        max_wait = 240.0
    if max_wait <= 0:
        return
    try:
        interval = float(_crew_vllm_poll_seconds_raw() or "8")
    except ValueError:
        interval = 8.0
    interval = max(2.0, min(interval, 60.0))

    models_url = GATEWAY.rstrip("/") + "/models"
    deadline = time.monotonic() + max_wait
    last_note = ""

    print(
        f"Waiting for vLLM via gateway (GET /v1/models, up to {int(max_wait)}s; "
        f"CREW_VLLM_WAIT_S=0 to skip)…",
        file=sys.stderr,
    )
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=90) as resp:
                if resp.status != 200:
                    last_note = f"HTTP {resp.status}"
                    time.sleep(interval)
                    continue
                payload = json.loads(resp.read().decode())
                if not isinstance(payload, dict):
                    last_note = "unexpected /v1/models shape"
                    time.sleep(interval)
                    continue
                if payload.get("error") == "upstream_plaintext":
                    last_note = payload.get("detail", "upstream_plaintext")
                    time.sleep(interval)
                    continue
                data = payload.get("data")
                if isinstance(data, list) and len(data) > 0:
                    print("vLLM is ready — running crew.\n", file=sys.stderr)
                    return
                last_note = "empty /v1/models data"
                time.sleep(interval)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                payload = json.loads(body)
                if payload.get("error") == "upstream_plaintext":
                    last_note = payload.get("detail", body[:120])
                else:
                    last_note = body[:200]
            except json.JSONDecodeError:
                last_note = body[:200] or f"HTTP {e.code}"
            time.sleep(interval)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_note = str(e)
            time.sleep(interval)
        except (json.JSONDecodeError, TypeError, ValueError):
            last_note = "invalid JSON from /v1/models"
            time.sleep(interval)

    print(
        "\nTimed out waiting for vLLM behind the gateway.\n"
        "The gateway was listening; /v1/models did not succeed in time.\n"
        "Check: vLLM running on Lambda (Step 6), SSH tunnel (Step 7), VLLM_BASE_URL=http://127.0.0.1:8000, "
        "first model download finished — or raise CREW_VLLM_WAIT_S (seconds).\n"
        f"(Last seen: {last_note!r})\n",
        file=sys.stderr,
    )
    sys.exit(3)


def _otlp_grpc_hostport(url: str) -> str:
    u = url.strip()
    for p in ("http://", "https://", "grpc://"):
        if u.startswith(p):
            u = u[len(p) :]
    return u.split("/")[0].split("?")[0]


def _flush_tracer_provider() -> None:
    prov = trace.get_tracer_provider()
    flush = getattr(prov, "force_flush", None)
    if callable(flush):
        try:
            flush(timeout_millis=15_000)
        except Exception:
            pass
    shutdown = getattr(prov, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass


def _setup_otel(service_name: str = "crewai") -> trace.Tracer:
    resource = Resource.create(
        {"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name)}
    )
    processor = None
    if _otlp_traces_enabled():
        exporter = OTLPSpanExporter(endpoint=_otlp_grpc_hostport(OTEL_ENDPOINT), insecure=True)
        processor = BatchSpanProcessor(exporter)
    current = trace.get_tracer_provider()
    attached = False
    if processor is not None:
        try:
            current.add_span_processor(processor)
            attached = True
        except Exception:
            pass
        if not attached:
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(processor)
            try:
                trace.set_tracer_provider(provider)
            except Exception:
                pass
    try:
        HTTPXClientInstrumentor().instrument()
    except Exception:
        pass
    return trace.get_tracer(__name__)


def build_crew(technique: str) -> Crew:
    litellm.drop_params = True
    litellm.headers = {"X-Technique": technique}
    if technique == "beam_search":
        litellm.extra_body = {"use_beam_search": True, "best_of": 4}
    else:
        litellm.extra_body = {}

    llm_kw: dict[str, Any] = {
        "model": f"openai/{SERVED}",
        "api_key": os.environ.get("OPENAI_API_KEY", "dummy"),
        "base_url": GATEWAY,
        "temperature": 0.2,
    }
    if _crew_llm_stream_enabled():
        llm_kw["stream"] = True
    llm = LLM(**llm_kw)

    researcher = Agent(
        role="Researcher",
        goal="Collect concise facts for a short brief.",
        backstory="You summarize sources clearly in 3–5 bullet points.",
        llm=llm,
        verbose=True,
    )
    writer = Agent(
        role="Writer",
        goal="Turn research into a tight paragraph.",
        backstory="You write clear prose under 120 words.",
        llm=llm,
        verbose=True,
    )

    task_r = Task(
        description=(
            "Topic: benefits of chunked prefill in LLM serving. "
            "List key ideas only, no preamble."
        ),
        expected_output="Bullet list of 3–5 points.",
        agent=researcher,
    )
    task_w = Task(
        description="Using the research only, write one short paragraph for a student.",
        expected_output="One paragraph, max 120 words.",
        agent=writer,
        context=[task_r],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[task_r, task_w],
        process=Process.sequential,
        verbose=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CrewAI crew with a labeled inference technique.")
    parser.add_argument(
        "--technique",
        default="baseline",
        help=(
            "X-Technique label and optional VLLM_BACKEND_MAP_JSON key. "
            "Built-ins: baseline, chunked_prefill, …, beam_search (adds use_beam_search), "
            "ab_nospec/ab_spec from scripts/run_server_ab.sh parallel mode, or any string."
        ),
    )
    args = parser.parse_args()

    _exit_if_gateway_shows_bad_vllm_url()
    _wait_for_vllm_via_gateway()

    crew = build_crew(args.technique)
    tracer = _setup_otel()
    if _otlp_traces_enabled():
        print(
            "OTLP export on → open Jaeger http://127.0.0.1:16686 and search service "
            f"{os.environ.get('OTEL_SERVICE_NAME', 'crewai')!r} (and gateway).",
            file=sys.stderr,
        )
    try:
        with tracer.start_as_current_span(
            "crew.run",
            attributes={"llm.technique": args.technique},
        ) as span:
            _carrier: dict[str, Any] = {}
            inject(_carrier)
            span.set_attribute("w3c_headers_count", len(_carrier))
            result = crew.kickoff()
            span.set_attribute("crew.result_chars", len(str(result)))
    finally:
        _flush_tracer_provider()

    print("--- Crew output ---")
    print(result)


if __name__ == "__main__":
    main()
