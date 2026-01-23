"""Prometheus metrics definitions for the arXiv classifier API.

Centralized metric definitions to ensure consistency across the application.
"""

from prometheus_client import Counter, Gauge, Histogram, Info, Summary

# Request metrics
REQUEST_COUNT = Counter(
    "arxiv_api_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "arxiv_api_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Inference metrics
INFERENCE_TIME = Histogram(
    "arxiv_api_inference_seconds",
    "Model inference time in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

INFERENCE_COUNT = Counter(
    "arxiv_api_inference_total",
    "Total number of inference requests",
    ["status"],
)

INPUT_TEXT_SIZE = Summary(
    "arxiv_api_input_text_size_chars",
    "Size of input text (title + abstract) in characters",
)

# Error metrics
ERROR_COUNT = Counter(
    "arxiv_api_errors_total",
    "Total number of errors",
    ["error_type"],
)

# Model info
MODEL_INFO = Info(
    "arxiv_api_model",
    "Information about the loaded model",
)

# System metrics
CPU_USAGE = Gauge(
    "arxiv_api_cpu_usage_percent",
    "CPU usage percentage",
)

MEMORY_USAGE = Gauge(
    "arxiv_api_memory_usage_bytes",
    "Memory usage in bytes",
)
