"""Embeddings over OpenAI-compatible `POST <base>/embeddings`, plus cosine similarity."""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Mapping, Sequence

from .providers import OPENAI_COMPATIBLE, ProviderError, _HttpClient, _require_key

Post = Callable[[str, Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]]


def embed(texts: Sequence[str], provider: str = "openai", model: str = "text-embedding-3-small",
          post: Post | None = None) -> list[list[float]]:
    cls = OPENAI_COMPATIBLE.get(provider.lower())
    if cls is None:
        raise ValueError(f"embed supports {', '.join(OPENAI_COMPATIBLE)}; got {provider!r}")
    key = _require_key(cls, None)
    root = (os.environ.get(cls.BASE_URL_ENV) or cls.DEFAULT_BASE_URL).rstrip("/")
    send = post or _HttpClient().post
    response = send(root + "/embeddings", {"Authorization": f"Bearer {key}"},
                    {"model": model, "input": list(texts)})
    try:
        items = sorted(response["data"], key=lambda item: int(item.get("index", 0)))
        vectors = [[float(x) for x in item["embedding"]] for item in items]
    except (KeyError, TypeError, ValueError) as error:
        raise ProviderError(f"unexpected embeddings response: {str(response)[:500]}") from error
    if len(vectors) != len(texts):
        raise ProviderError(f"expected {len(texts)} embeddings, got {len(vectors)}")
    return vectors


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors differ in length")
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


__all__ = ["cosine", "embed"]
