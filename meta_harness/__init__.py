"""Filesystem-backed, end-to-end harness optimization."""

from .core import (
    CandidateEvaluator,
    CommandProposer,
    EvaluationResult,
    FilesystemExperience,
    HarnessValidationError,
    ParetoFrontier,
    SearchConfig,
    SearchRunner,
)
from .retrieval import BM25Index, TfidfIndex, reciprocal_rank_fusion
from .harnesses import (
    DraftVerificationHarness,
    LabelPrimedQueryHarness,
    MathRetrievalHarness,
    TerminalBootstrap,
    environment_snapshot,
)
from .providers import AnthropicModel, OpenAICompatibleModel, ProviderError, model_from_environment
from .proposer import PromptedProposer
from .terminal import ShellPolicy, TerminalAgent, TerminalResult

__all__ = [
    "CandidateEvaluator", "CommandProposer", "EvaluationResult",
    "FilesystemExperience", "HarnessValidationError", "ParetoFrontier",
    "SearchConfig", "SearchRunner",
    "BM25Index", "TfidfIndex", "reciprocal_rank_fusion",
    "DraftVerificationHarness", "LabelPrimedQueryHarness", "MathRetrievalHarness",
    "TerminalBootstrap", "environment_snapshot",
    "AnthropicModel", "OpenAICompatibleModel", "ProviderError", "model_from_environment",
    "PromptedProposer", "ShellPolicy", "TerminalAgent", "TerminalResult",
]
