"""`meta-harness search`: plug-and-play harness search over a harness file, an eval set and a metric."""

from __future__ import annotations

import json
import os
import shlex
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..cache import CachedModel
from ..chat_types import Usage
from ..core import MAX_AUTOMATIC_RESUMES, CandidateEvaluator, CommandProposer, SearchConfig, SearchRunner
from ..datasets import read_records, split_tasks
from ..domains import get_domain, metric_for
from ..optimizers import OptimizerContext, build as build_optimizer
from ..protocol import resolve_metric
from ..providers import Price, ProviderError, cost_usd, model_from_environment, price_for
from ..redact import redact

NAME = "search"
HELP = "search for a better harness: seed harness files + a JSONL/JSON/CSV eval set + a metric"

PROVIDERS = ("anthropic", "openai", "compatible", "unikey", "scripted")
BASE_URL_PROVIDERS = ("compatible", "openai", "unikey")
DOMAINS = ("classification", "math", "generic", "terminal")


class UsageError(Exception):
    """A user-facing setup error: printed as one line on stderr, exit code 2."""


def add_arguments(p) -> None:
    p.add_argument("--harness", action="append", required=True, metavar="PATH",
                   help="seed harness file (repeatable)")
    data = p.add_mutually_exclusive_group()
    data.add_argument("--data", help="JSONL/JSON/CSV eval set, split by --test-fraction")
    data.add_argument("--search", help="explicit search split (with --test)")
    p.add_argument("--test", help="explicit held-out test split (with --search)")
    p.add_argument("--test-fraction", type=float, default=0.3)
    p.add_argument("--train", help="online domains: explicit train split (disjoint from search and test)")
    p.add_argument("--train-fraction", type=float, default=0.4,
                   help="online domains with --data: share of the non-test rows used as the train split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metric", default=None,
                   help="exact|contains|numeric, a domain metric, or module:function "
                        "(default: the domain's metric, exact for generic)")
    p.add_argument("--dataset-version", default=None,
                   help="terminal domain: TerminalBench dataset version recorded in run.json")
    p.add_argument("--provider", choices=PROVIDERS, required=True, help="where the harness model runs")
    p.add_argument("--model", required=True, help="harness model id (scripted: the script path)")
    p.add_argument("--base-url", help="API base URL for compatible/openai/unikey (default: environment)")
    p.add_argument("--proposer-provider", choices=PROVIDERS, help="default: --provider")
    p.add_argument("--proposer-model", help="default: --model")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--view", choices=["scores", "summary", "full"], default="full")
    p.add_argument("--max-tool-calls", type=int, default=40)
    p.add_argument("--domain", choices=DOMAINS, default="generic", help="selects the prompt section")
    p.add_argument("--root", default=".meta-harness")
    p.add_argument("--test-root")
    p.add_argument("--cache-dir")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--proposer-command",
                   help="escape hatch: external agent writing .py files to META_HARNESS_OUTPUT")
    p.add_argument("--proposer-timeout", type=float, default=1800.0)
    p.add_argument("--validation-timeout", type=float, default=20.0)
    p.add_argument("--task-timeout", type=float, default=600.0)
    p.add_argument("--isolation", choices=["none", "process", "docker"], default="process",
                   help="where candidates run: a limited child process with no API keys (default), "
                        "a networkless Docker container, or in-process")
    p.add_argument("--memory-mb", type=int, default=2048, help="candidate memory limit (MiB)")
    p.add_argument("--cpu-seconds", type=int, default=600, help="candidate CPU-time limit (seconds)")
    p.add_argument("--resume", action="store_true", help="continue the run stored under --root")
    p.add_argument("--force-resume", action="store_true",
                   help="resume even past the automatic-resume limit (resets .resume_attempts)")
    p.add_argument("--price", action="append", default=[], metavar="PROVIDER:MODEL=IN,OUT[,CR,CW]",
                   help="USD per million tokens (repeatable); feeds the billed cost")


# -- helpers --------------------------------------------------------------------------------------

def split_command(command: str) -> list[str]:
    """Split a shell-ish command string. On Windows, backslashes are path separators,
    not escapes, so posix-mode shlex would mangle every absolute path."""
    if os.name != "nt":
        return shlex.split(command)
    return [token[1:-1] if len(token) > 1 and token[0] == token[-1] == '"' else token
            for token in shlex.split(command, posix=False)]


def resolve_command(tokens: Sequence[str]) -> list[str]:
    """Absolutize tokens naming a real file: the proposer runs with cwd set to the experience root."""
    return [str(Path(token).resolve()) if token and Path(token).exists() else token
            for token in tokens]


def parse_prices(specs: Sequence[str]) -> dict[str, Price]:
    """`provider:model=in,out[,cache_read,cache_write]` -> {model: Price}."""
    prices: dict[str, Price] = {}
    for spec in specs:
        target, sep, numbers = spec.rpartition("=")
        provider, colon, model = target.partition(":")
        if not sep or not colon or provider.lower() not in PROVIDERS or not model:
            raise UsageError(f"--price must be provider:model=in,out[,cache_read,cache_write], got {spec!r}")
        try:
            values = [float(v) for v in numbers.split(",")]
        except ValueError:
            raise UsageError(f"--price values must be numbers, got {numbers!r}") from None
        if len(values) not in (2, 3, 4) or any(v < 0 for v in values):
            raise UsageError(f"--price needs 2 to 4 non-negative numbers, got {numbers!r}")
        prices[f"{provider.lower()}:{model}"] = Price(*values)
        prices.setdefault(model, Price(*values))   # bare-model fallback (first --price wins)
    return prices


def bare_prices(prices: Mapping[str, Price]) -> dict[str, Price]:
    """The --price overrides keyed by bare model id, as core's Budget.price looks them up."""
    return {name: price for name, price in prices.items() if name.split(":", 1)[0] not in PROVIDERS}


def _make_model(provider: str, model: str, base_url: str | None) -> Any:
    kwargs = {"base_url": base_url} if base_url and provider in BASE_URL_PROVIDERS else {}
    if provider == "scripted" and not Path(model).is_file():
        raise UsageError(f"scripted model script not found: {model}")
    return model_from_environment(provider, model, **kwargs)


def _read(path: str, domain) -> list[dict]:
    if not Path(path).is_file():
        raise UsageError(f"data file not found: {path}")
    return domain.adapt(read_records(path))


def _load_tasks(args, domain) -> tuple[list[dict], list[dict]]:
    if args.data:
        records = _read(args.data, domain)
        if len(records) < 2:
            raise UsageError(f"{args.data} needs at least 2 rows to split, found {len(records)}")
        return split_tasks(records, seed=args.seed, test_fraction=args.test_fraction)
    if args.search:
        search = _read(args.search, domain)
        test = _read(args.test, domain) if args.test else []
        if not search:
            raise UsageError(f"{args.search} has no rows")
        return search, test
    raise UsageError("give --data, or --search (with an optional --test)")


def _task_key(task: Mapping[str, Any]) -> str:
    if task.get("id") is not None:
        return f"id:{task['id']}"
    return "input:" + json.dumps(task.get("input"), sort_keys=True, default=str)


def assert_disjoint(**splits: Sequence[Mapping[str, Any]]) -> None:
    """Refuse overlapping splits (by id, else by input)."""
    names = list(splits)
    keys = {name: {_task_key(t) for t in splits[name]} for name in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = keys[a] & keys[b]
            if shared:
                raise UsageError(f"{a} and {b} splits overlap on {len(shared)} task(s), "
                                 f"e.g. {sorted(shared)[0]}")


def _train_tasks(args, domain, search_tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Online protocol: (train, search) with train disjoint from search. Never trains on search."""
    if args.train:
        return _read(args.train, domain), search_tasks
    provided = getattr(domain, "train_tasks", None)
    if callable(provided):
        train = provided(args.search)
        if train:
            return [dict(t) for t in train], search_tasks
    if args.data:
        if not 0.0 < args.train_fraction < 1.0:
            raise UsageError("--train-fraction must be strictly between 0 and 1")
        if len(search_tasks) < 2:
            raise UsageError("too few non-test rows for a train/search split; give --train")
        search, train = split_tasks(search_tasks, seed=args.seed + 1, test_fraction=args.train_fraction)
        return train, search
    raise UsageError(f"domain {args.domain!r} uses the online protocol and needs a train split "
                     "disjoint from search: give --train, or --data for a seeded three-way split")


def default_metric(name: str, domain) -> tuple[str, Any]:
    """(resolved name, metric): the domain's own METRICS[name] (terminal), else the shared
    task-type map, else exact."""
    own = getattr(domain, "METRICS", None) or {}
    if name in own:
        return name, own[name]
    try:
        metric = metric_for(name)
    except KeyError:
        return "exact", resolve_metric("exact")
    return getattr(metric, "__name__", name), metric


def _verdicts(root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    candidates = root / "candidates"
    if candidates.is_dir():
        for directory in sorted(candidates.iterdir()):
            receipt = SearchRunner.read_receipt(directory) if directory.is_dir() else None
            if receipt and receipt.get("verdict"):
                counts[str(receipt["verdict"])] += 1
    return dict(sorted(counts.items()))


def _price(provider: str, model: str, prices: Mapping[str, Price]) -> Price | None:
    return prices.get(f"{provider}:{model}") or price_for(model, prices)


def _billed(budget, model: tuple[str, str], proposer_model: tuple[str, str],
            prices: Mapping[str, Price]) -> float | None:
    total = 0.0
    for (provider, name), usage, used in ((model, budget.model_usage, budget.model_calls),
                                          (proposer_model, budget.proposer_usage, budget.proposer_calls)):
        if not used:
            continue
        cost = cost_usd(Usage(**usage), _price(provider, name, prices))
        if cost is None:
            return None
        total += cost
    return total


# -- the command ----------------------------------------------------------------------------------

def _prepare(args):
    """Setup phase: validates everything and builds the runner; user errors raise here."""
    if args.test and not args.search:
        raise UsageError("--test needs --search")
    if not 0.0 < args.test_fraction < 1.0:
        raise UsageError("--test-fraction must be strictly between 0 and 1")
    harnesses = [Path(h) for h in args.harness]
    missing = [str(h) for h in harnesses if not h.is_file()]
    if missing:
        raise UsageError(f"harness file not found: {', '.join(missing)}")
    root = Path(args.root)
    resume = args.resume or args.force_resume
    if resume and not (root / "run.json").is_file():
        raise UsageError(f"cannot resume: {root} has no run.json")
    prices = parse_prices(args.price)
    try:
        domain = get_domain(args.domain)
    except KeyError as exc:
        raise UsageError(str(exc.args[0] if exc.args else exc)) from None
    if args.metric:
        metric_name, metric_source = args.metric, "flag"
        metric = resolve_metric(args.metric, extra=getattr(domain, "METRICS", None) or None)
    else:
        (metric_name, metric), metric_source = default_metric(args.domain, domain), "domain_default"
    search_tasks, test_tasks = _load_tasks(args, domain)
    protocol = getattr(domain, "PROTOCOL", "plain")
    train_tasks: list[dict] = []
    if protocol == "online":
        train_tasks, search_tasks = _train_tasks(args, domain, search_tasks)
        if not train_tasks:
            raise UsageError("the online protocol needs a non-empty train split")
        assert_disjoint(train=train_tasks, search=search_tasks, test=test_tasks)
    elif args.train:
        raise UsageError(f"--train applies only to online-protocol domains, not {args.domain!r}")
    if resume and not args.force_resume:
        try:
            attempts = int((root / ".resume_attempts").read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            attempts = 0
        if attempts + 1 > MAX_AUTOMATIC_RESUMES:
            raise UsageError(f"{root} was already resumed {attempts} times automatically "
                             f"(limit {MAX_AUTOMATIC_RESUMES}); inspect it and use --force-resume")

    if args.memory_mb <= 0 or args.cpu_seconds <= 0:
        raise UsageError("--memory-mb and --cpu-seconds must be positive")
    if args.isolation == "docker" and not getattr(domain, "HARNESS_WRAPPER", None):
        from ..sandboxes import preflight
        check = preflight()
        if not check.ok:  # abort with the install hint; never fall back silently
            raise UsageError(f"--isolation docker: {check.problem}. {check.hint}")

    base_model = _make_model(args.provider, args.model, args.base_url)
    proposer_provider = args.proposer_provider or args.provider
    proposer_model_name = args.proposer_model or args.model
    cache_dir = Path(args.cache_dir) if args.cache_dir else root.parent / ".meta-harness-cache"
    model = CachedModel(base_model, cache_dir, enabled=not args.no_cache)

    run_meta = {"provider": args.provider, "model": args.model,
                "proposer_provider": proposer_provider, "proposer_model": proposer_model_name,
                "optimizer": "command" if args.proposer_command else "metaharness",
                "view": args.view, "domain": args.domain, "metric": metric_name,
                "metric_source": metric_source, "isolation": args.isolation}
    if args.domain == "terminal":
        from ..loaders import terminalbench
        version = args.dataset_version or terminalbench.DEFAULT_VERSION
        run_meta.update(terminalbench.run_meta(version))
    config = SearchConfig(root=root, iterations=args.iterations, candidates_per_iteration=args.candidates,
                          validation_timeout=args.validation_timeout, repeats=args.repeats,
                          max_workers=args.max_workers, task_timeout=args.task_timeout,
                          test_root=args.test_root, run_meta=run_meta)
    if args.proposer_command:
        proposer = CommandProposer(resolve_command(split_command(args.proposer_command)),
                                   timeout=args.proposer_timeout)
    else:
        same = proposer_provider == args.provider
        proposer_model = _make_model(proposer_provider, proposer_model_name, args.base_url if same else None)
        summarizer = proposer_model if args.view == "summary" else None
        ctx = OptimizerContext.for_config(config, proposer_model, domain, view=args.view,
                                          max_tool_calls=args.max_tool_calls, seed=args.seed,
                                          summarizer=summarizer)
        proposer = build_optimizer("metaharness", ctx)
    evaluator = CandidateEvaluator(model, metric, task_timeout=args.task_timeout, protocol=protocol,
                                   train_tasks=train_tasks,
                                   harness_wrapper=getattr(domain, "HARNESS_WRAPPER", None),
                                   source_validator=getattr(domain, "VALIDATE", None),
                                   import_only_validation=bool(getattr(domain, "VALIDATE_IMPORT_ONLY", False)),
                                   isolation=args.isolation, memory_mb=args.memory_mb,
                                   cpu_seconds=args.cpu_seconds)
    runner = SearchRunner(config, evaluator, proposer, prices=bare_prices(prices))
    return lambda: _execute(args, runner, model, config, root, harnesses, search_tasks, test_tasks,
                            train_tasks, resume, prices, proposer_provider, proposer_model_name)


def _execute(args, runner, model, config, root, harnesses, search_tasks, test_tasks, train_tasks,
             resume, prices, proposer_provider, proposer_model_name) -> int:
    """The engine phase: its exceptions propagate with their traceback (not turned into exit 2)."""
    if args.force_resume:
        (root / ".resume_attempts").unlink(missing_ok=True)
    frontier = runner.run(harnesses, search_tasks, test_tasks or None, resume=resume,
                          force_resume=args.force_resume)
    budget = runner.budget
    test_path = Path(config.test_root) / "test_results.json"
    print(json.dumps({
        "root": str(config.root),
        "search_tasks": len(search_tasks),
        "test_tasks": len(test_tasks),
        "train_tasks": len(train_tasks),
        "cache": {"hits": model.hits, "misses": model.misses},
        "frontier": [asdict(item) for item in frontier],
        "test_results": str(test_path) if test_path.is_file() else None,
        "context_tokens": {"model": budget.model_usage["input"], "proposer": budget.proposer_usage["input"]},
        "cost_usd": _billed(budget, (args.provider, args.model), (proposer_provider, proposer_model_name),
                            prices),
        "verdicts": _verdicts(root),
        "budget": asdict(budget),
    }, indent=2, default=str))
    return 0


def run(args) -> int:
    try:
        execute = _prepare(args)
    except (UsageError, ProviderError, ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
        message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
        print(f"error: {redact(message).splitlines()[0] if message else type(exc).__name__}",
              file=sys.stderr)
        return 2
    return execute()
