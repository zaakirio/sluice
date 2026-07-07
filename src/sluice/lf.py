from __future__ import annotations

import logging
import os

log = logging.getLogger("sluice")


def build_langfuse():
    """Returns a Langfuse client when LANGFUSE_PUBLIC_KEY and
    LANGFUSE_SECRET_KEY are set and the SDK is installed, else None.
    None means every instrumentation site is a no-op."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning(
            "LANGFUSE_PUBLIC_KEY/SECRET_KEY are set but the langfuse package "
            "is not installed; install with `uv sync --extra obs`"
        )
        return None
    return Langfuse()
