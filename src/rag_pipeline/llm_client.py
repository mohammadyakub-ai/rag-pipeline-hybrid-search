"""Chat-completion client that prefers the official OpenRouter SDK.

The pipeline only needs a tiny OpenAI-compatible surface:
``client.chat.completions.create(model=..., messages=..., temperature=..., **kw)``.
This module picks the provider and exposes exactly that interface.

Provider selection (first match wins):

1. An explicit ``api_key`` argument -> ``openrouter`` if it looks like an
   OpenRouter key (``sk-or-v1-...``), otherwise ``openai``.
2. ``OPENROUTER_API_KEY`` env var -> the official ``openrouter`` SDK.
3. ``OPENAI_API_KEY`` env var -> the ``openai`` package, unless the value is an
   OpenRouter key pasted into the OpenAI slot, in which case it is routed
   through OpenRouter too.
4. no key at all            -> an unconfigured client (``is_configured`` False).
"""

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional

OPENROUTER_KEY_PREFIX = "sk-or-v1-"


def is_openrouter_key(key: str) -> bool:
    """Return True if the key is an OpenRouter API key (``sk-or-v1-...``)."""
    return key.startswith(OPENROUTER_KEY_PREFIX)


class ChatCompletionClient:
    """Minimal OpenAI-compatible chat client backed by OpenRouter or OpenAI.

    The ``chat`` / ``completions`` properties return the client itself so that
    call sites keep the familiar ``client.chat.completions.create(...)`` shape.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.provider: Optional[str] = None
        self._client = None

        key = api_key
        if key is not None:
            provider = "openrouter" if is_openrouter_key(key) else "openai"
        elif os.environ.get("OPENROUTER_API_KEY"):
            key = os.environ["OPENROUTER_API_KEY"]
            provider = "openrouter"
        elif os.environ.get("OPENAI_API_KEY"):
            key = os.environ["OPENAI_API_KEY"]
            provider = "openrouter" if is_openrouter_key(key) else "openai"
        else:
            return

        if provider == "openrouter":
            try:
                from openrouter import OpenRouter
            except ImportError:
                raise RuntimeError(
                    "openrouter package not installed. Run: pip install openrouter"
                )
            self._client = OpenRouter(api_key=key)
        else:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("openai package not installed")
            self._client = OpenAI(api_key=key)
        self.provider = provider

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model: str, messages=None, temperature: float = 0.0, **kwargs):
        client = self._client
        if client is None:
            raise RuntimeError(
                "LLM client not configured. Set OPENROUTER_API_KEY or "
                "OPENAI_API_KEY, or pass an api_key."
            )
        if self.provider == "openrouter":
            if model and "/" not in model:
                model = f"openai/{model}"
            return client.chat.send(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs,
            )
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )