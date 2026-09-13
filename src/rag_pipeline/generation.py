from typing import List, Optional

from .retrieval import RetrievalResult


DEFAULT_SYSTEM_PROMPT = """You MUST cite every factual claim using [N] notation where N is the chunk number from the context below.
Every sentence that states a fact MUST end with [N].
Example: FastAPI is based on Pydantic [1] and Starlette [2].
Never write a factual sentence without a [N] citation.
If the context does not contain the answer, say:
'I cannot find this information in the provided documentation.'
Do not make any claim that is not supported by the context.
"""

DEFAULT_ANSWER_FORMAT = """Answer format:
- Provide a direct answer.
- End each factual sentence with its citation(s), e.g. "The feature requires a bearer token [2]."
- If only part of the question can be answered, answer that part and clearly flag the remainder as unanswered.
- If nothing in the context answers the question, respond with "I don't have enough information in the context to answer this question." and do NOT cite any source.
"""


class GroundedPrompt:
    """Builds the system + user messages for grounded generation.

    Retrieved chunks are formatted as numbered context blocks, and the answer
    is required to cite them via bracketed references ([1], [2], ...).
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        answer_format: Optional[str] = None,
        context_intro: Optional[str] = None,
    ):
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.answer_format = answer_format or DEFAULT_ANSWER_FORMAT
        self.context_intro = (
            context_intro
            or "Below are the reference context blocks. Each block is numbered; "
            "cite them in your answer using that number in brackets."
        )

    def format_context(self, results: List[RetrievalResult]) -> str:
        """Serialize retrieved chunks into numbered context blocks."""
        if not results:
            return self.context_intro + "\n\n(No context provided.)"
        blocks = [self.context_intro, ""]
        for i, res in enumerate(results, start=1):
            src = res.metadata.get("source_file", "unknown")
            heading = res.metadata.get("section_heading")
            blocks.append(self._block(i, res.content, src, heading))
        return "\n".join(blocks)

    def _block(self, index: int, content: str, source: str, heading: Optional[str]) -> str:
        header_parts = [f"[Source: {source}]"]
        if heading:
            header_parts.append(f"heading: {heading}")
        return (
            f"<context id=\"{index}\">\n"
            f"Reference {index}. ({' | '.join(header_parts)})\n"
            f"{content.strip()}\n"
            f"</context>"
        )

    def build_messages(
        self, question: str, results: List[RetrievalResult]
    ) -> List[dict]:
        """Return OpenAI-style messages: system + user with numbered context."""
        context = self.format_context(results)
        user_content = (
            f"{context}\n\n"
            f"{self.answer_format}\n\n"
            f"Question: {question}"
        )
        return [
            {"role": "system", "content": self.system_prompt.strip()},
            {"role": "user", "content": user_content},
        ]


class Generator:
    """Calls the LLM with a grounded prompt and returns the response.

    Uses the OpenRouter SDK when an OpenRouter key is configured, falling back
    to OpenAI; swap in your own `respond` callable for other providers. Parsing
    of citations happens in Phase 3.2.
    """

    def __init__(
        self,
        prompt: Optional[GroundedPrompt] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        respond: Optional[callable] = None,
    ):
        self.prompt = prompt or GroundedPrompt()
        self.model = model
        self.temperature = temperature
        self._respond = respond
        self._client = None
        if respond is None:
            from .llm_client import ChatCompletionClient
            self._client = ChatCompletionClient(api_key=api_key)

    @property
    def is_configured(self) -> bool:
        if self._respond is not None:
            return True
        return self._client is not None and self._client.is_configured

    def generate(
        self, question: str, results: List[RetrievalResult], **kwargs
    ) -> dict:
        """Return a dict with 'answer', 'source_chunks', and 'messages'.

        `system_prompt` may be passed in kwargs to override the default system
        prompt for this call (e.g. a judge or verifier instruction).
        """
        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt is not None:
            prompt = GroundedPrompt(system_prompt=system_prompt)
        else:
            prompt = self.prompt
        # Answer generation gets a larger token budget. Judge/verifier calls
        # always pass a custom `system_prompt`, so they fall through to the
        # tighter 1024-token default applied inside `_call`.
        if system_prompt is None:
            kwargs.setdefault("max_tokens", 2048)
        messages = prompt.build_messages(question, results)
        answer = self._call(messages, kwargs)
        return {
            "answer": answer,
            "source_chunks": results,
            "messages": messages,
            "model": self.model,
        }

    def _call(self, messages: List[dict], kwargs: dict) -> str:
        if self._respond is not None:
            return self._respond(messages)
        if self._client is None or not self._client.is_configured:
            raise RuntimeError(
                "Generator not configured. Set OPENROUTER_API_KEY or "
                "OPENAI_API_KEY, pass api_key, or provide a `respond` callable."
            )
        # Bound the response length by default. Without an explicit
        # max_tokens, OpenRouter's SDK default (16384) exceeds typical
        # account budgets and the request is rejected with a 402
        # PaymentRequired error before the model responds.
        kwargs.setdefault("max_tokens", 1024)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **kwargs,
        )
        return resp.choices[0].message.content