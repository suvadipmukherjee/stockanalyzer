"""Claude integration: send the snapshot + template, stream back the report."""

from __future__ import annotations

import sys
from typing import Callable

import anthropic

from .data import Snapshot
from .prompt import SYSTEM_PROMPT, build_user_prompt

DEFAULT_MODEL = "claude-opus-5-5"


class AnalysisError(RuntimeError):
    pass


def analyze(
    snapshot: Snapshot,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = "medium",
    focus: str | None = None,
    on_text: Callable[[str], None] | None = None,
    client: anthropic.Anthropic | None = None,
) -> str:
    """Return the Markdown report. Streams text chunks to `on_text` if given.

    The client reads ANTHROPIC_API_KEY (or an `ant auth login` profile) from
    the environment; no key is ever hard-coded.
    """
    client = client or anthropic.Anthropic()

    request = dict(
        model=model,
        max_tokens=16000,
        # Fixed system prompt, marked for caching so repeat runs are cheaper.
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_user_prompt(snapshot, focus)}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
    )
    # If the model declines (rare for this kind of request), let the API
    # re-run on a suitable fallback model instead of returning nothing.
    if model.startswith(("claude-opus-5", "claude-fable", "claude-sonnet-5-5")):
        request["betas"] = ["server-side-fallback-2026-07-01"]
        request["fallbacks"] = "default"

    try:
        with client.beta.messages.stream(**request) as stream:
            for text in stream.text_stream:
                if on_text:
                    on_text(text)
            message = stream.get_final_message()
    except TypeError as e:
        if "authentication" not in str(e).lower():
            raise
        raise AnalysisError(
            "No Claude credentials found. Set ANTHROPIC_API_KEY (see README)."
        ) from e
    except anthropic.AuthenticationError as e:
        raise AnalysisError(
            "Claude rejected the credentials. Set ANTHROPIC_API_KEY (see README)."
        ) from e
    except anthropic.RateLimitError as e:
        raise AnalysisError("Rate limited by the Claude API. Wait a moment and retry.") from e
    except anthropic.APIStatusError as e:
        raise AnalysisError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise AnalysisError("Could not reach the Claude API. Check your network.") from e

    if message.stop_reason == "refusal":
        raise AnalysisError("Claude declined to produce this report.")
    if message.stop_reason == "max_tokens":
        print("\n[warning] report was cut off at max_tokens", file=sys.stderr)

    return "".join(b.text for b in message.content if b.type == "text")
