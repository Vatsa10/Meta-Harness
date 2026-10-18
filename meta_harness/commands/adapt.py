"""`meta-harness adapt`: wrap a production agent as a config harness.

    adapt init --entrypoint mod:fn --config cfg.json [--schema schema.json] --out seed_dir [--path DIR]
    adapt sync seed_dir                    re-render harness.py after editing config.json
    adapt check --candidate DIR --seed DIR the config-only gate, standalone
"""

from __future__ import annotations

import json
import sys

from ..adapters.config_harness import ConfigHarnessError, init_seed, sync_seed, validate_candidate
from ..adapters.schema import SchemaError

NAME = "adapt"
HELP = "wrap a production agent (module:function(task, config, model)) as a config harness"


def add_arguments(parser) -> None:
    sub = parser.add_subparsers(dest="adapt_command", required=True)
    init = sub.add_parser("init", help="scaffold seed_dir/harness.py, config.json and schema.json")
    init.add_argument("--entrypoint", required=True, help="module:function(task, config, model) -> dict")
    init.add_argument("--config", required=True, help="the agent's config JSON")
    init.add_argument("--schema", help="schema JSON (types, enum, minimum/maximum, maxLength, "
                                       "required, frozen); default: any object")
    init.add_argument("--out", required=True, help="seed directory to write")
    init.add_argument("--path", default=".", help="directory the entrypoint module is imported from "
                                                  "(default: the current directory)")
    init.add_argument("--no-import-check", action="store_true",
                      help="do not import the entrypoint while scaffolding")
    sync = sub.add_parser("sync", help="re-render harness.py from a hand-edited config.json / schema.json")
    sync.add_argument("seed_dir")
    check = sub.add_parser("check", help="run the config-only gate on a candidate directory")
    check.add_argument("--candidate", required=True)
    check.add_argument("--seed", required=True)


def run(args) -> int:
    try:
        if args.adapt_command == "init":
            path = init_seed(args.entrypoint, args.config, args.out, schema_path=args.schema,
                             entry_dir=args.path, check_import=not args.no_import_check)
            print(json.dumps({"harness": str(path), "seed_dir": str(path.parent)}))
        elif args.adapt_command == "sync":
            print(json.dumps({"harness": str(sync_seed(args.seed_dir))}))
        else:
            validate_candidate(args.candidate, args.seed)
            print(json.dumps({"ok": True}))
        return 0
    except (ConfigHarnessError, SchemaError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
