"""Per-request RAGAS-proxy logging.

Production can't run the full RAGAS library on every request (each metric
needs an LLM judge call). Instead we log the **deterministic structural
proxy** Day 3 already uses for the RAGAS composite — faithfulness,
relevancy, ctx_precision, ctx_recall computed from string overlap. It's
the same proxy used to pick the Day-3 champion, and it gives us live
quality numbers without an LLM-in-the-loop on every request.
"""

from .ragas_log import RAGASProxyLogger, score_request, get_logger  # noqa: F401
