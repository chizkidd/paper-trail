"""Shared plumbing for LLM backends: a token stream that remembers why it failed."""
from typing import Callable, Iterator

OK            = "ok"
NO_TOKEN      = "no HF token configured"
AUTH          = "HF token rejected or lacks Inference Providers permission"
UNAVAILABLE   = "model not available on the HF router"
RATE_LIMITED  = "HF free credits used up or rate limited"
TIMEOUT       = "request timed out"
NOT_REACHABLE = "Ollama not reachable"
EMPTY         = "model returned no text"
ERROR         = "request failed"


class LLMStream:
    """
    Iterable of text tokens. After iteration, `.status` is OK or a human-readable
    reason (one of the constants above) saying why nothing usable came back.
    """

    def __init__(self, source: Callable[["LLMStream"], Iterator[str]]):
        self.status = OK
        self._it = self._run(source)

    def _run(self, source: Callable[["LLMStream"], Iterator[str]]) -> Iterator[str]:
        produced = False
        for tok in source(self):
            produced = True
            yield tok
        if not produced and self.status == OK:
            self.status = EMPTY

    def __iter__(self) -> Iterator[str]:
        return self._it

    def collect(self) -> str:
        return "".join(self).strip()
