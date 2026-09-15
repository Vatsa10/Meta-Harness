"""Filesystem-backed, end-to-end harness optimization."""

from .core import (
    CandidateEvaluator,
    CommandProposer,
    EvaluationResult,
    FilesystemExperience,
    HarnessValidationError,
    MeteredModel,
    ParetoFrontier,
    SearchConfig,
    SearchRunner,
    TraceRecorder,
)
from .agent_proposer import SKILL_PATH, ClaudeCodeProposer, build_view
from .cache import CachedModel, DiskCache
from .datasets import classification_tasks, math_tasks, read_records, split_tasks, terminal_tasks
from .metrics import METRICS, estimate_tokens, exact_match, math_equivalence, normalize_answer, terminal_metric
from .retrieval import BM25Index, TfidfIndex, reciprocal_rank_fusion
from .harnesses import (
    DraftVerificationHarness,
    LabelPrimedQueryHarness,
    MathRetrievalHarness,
    TerminalBootstrap,
    environment_snapshot,
)
from .providers import (
    UNIKEY_ANTHROPIC_BASE_URL,
    UNIKEY_BASE_URL,
    AnthropicModel,
    OpenAICompatibleModel,
    ProviderError,
    UnikeyModel,
    list_unikey_models,
    model_from_environment,
)
from .proposer import PromptedProposer
from .sandbox import validate_in_subprocess
from .terminal import ShellPolicy, TerminalAgent, TerminalResult
from .terminal_harness import DockerPolicy, TerminalHarness

__all__ = [
    "CandidateEvaluator", "CommandProposer", "EvaluationResult",
    "FilesystemExperience", "HarnessValidationError", "MeteredModel", "ParetoFrontier",
    "SearchConfig", "SearchRunner", "TraceRecorder",
    "SKILL_PATH", "ClaudeCodeProposer", "build_view",
    "CachedModel", "DiskCache",
    "classification_tasks", "math_tasks", "read_records", "split_tasks", "terminal_tasks",
    "METRICS", "estimate_tokens", "exact_match", "math_equivalence", "normalize_answer",
    "terminal_metric",
    "BM25Index", "TfidfIndex", "reciprocal_rank_fusion",
    "DraftVerificationHarness", "LabelPrimedQueryHarness", "MathRetrievalHarness",
    "TerminalBootstrap", "environment_snapshot",
    "UNIKEY_ANTHROPIC_BASE_URL", "UNIKEY_BASE_URL", "AnthropicModel", "OpenAICompatibleModel",
    "ProviderError", "UnikeyModel", "list_unikey_models", "model_from_environment",
    "PromptedProposer", "validate_in_subprocess",
    "ShellPolicy", "TerminalAgent", "TerminalResult",
    "DockerPolicy", "TerminalHarness",
]
