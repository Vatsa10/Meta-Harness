"""`meta-harness models`: list the model ids an OpenAI-compatible provider serves."""

from __future__ import annotations

import sys

from ..providers import OPENAI_COMPATIBLE, ProviderError, list_models
from ..redact import redact

NAME = "models"
HELP = "list available harness models"


def add_arguments(parser) -> None:
    parser.add_argument("--provider", choices=sorted(OPENAI_COMPATIBLE), default="unikey")
    parser.add_argument("--base-url", help="API base URL (default: environment, then the provider's)")


def run(args) -> int:
    try:
        ids = list_models(args.provider, base_url=args.base_url)
    except (ProviderError, ValueError) as exc:
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return 2
    print("\n".join(ids))
    return 0
