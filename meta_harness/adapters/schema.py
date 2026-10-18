"""Config schema for adapter harnesses: a small, stdlib-only subset of JSON Schema.

A schema node is a dict with:
    type            "object" | "array" | "string" | "number" | "integer" | "boolean" | "null"
                    (or a list of those)
    enum            allowed values
    minimum/maximum numeric bounds (inclusive)
    maxLength       maximum string length (minLength too)
    maxItems        maximum array length (minItems too)
    properties      object: {key: schema}
    required        object: keys that must be present
    additionalProperties   object: false refuses keys not in `properties` (default false)
    items           array: schema for every element

The ROOT schema may also hold `frozen`: a list of dotted paths ("router.threshold",
"tools.0.name", or a whole subtree "tools") whose values a candidate may not change from the seed.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

TYPES = ("object", "array", "string", "number", "integer", "boolean", "null")
_MISSING = object()


class SchemaError(ValueError):
    """The schema itself is malformed."""


def _type_ok(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise SchemaError(f"unknown type {kind!r}; known: {', '.join(TYPES)}")


def check_schema(schema: Any, path: str = "$") -> None:
    """Raise SchemaError when the schema is not well formed."""
    if not isinstance(schema, Mapping):
        raise SchemaError(f"{path}: schema node must be an object")
    kinds = schema.get("type")
    if kinds is not None:
        for kind in kinds if isinstance(kinds, list) else [kinds]:
            if kind not in TYPES:
                raise SchemaError(f"{path}: unknown type {kind!r}")
    for key in ("minimum", "maximum"):
        if key in schema and not _type_ok(schema[key], "number"):
            raise SchemaError(f"{path}: {key} must be a number")
    for key in ("maxLength", "minLength", "maxItems", "minItems"):
        if key in schema and not (_type_ok(schema[key], "integer") and schema[key] >= 0):
            raise SchemaError(f"{path}: {key} must be a non-negative integer")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise SchemaError(f"{path}: enum must be a list")
    if "required" in schema and not (isinstance(schema["required"], list)
                                     and all(isinstance(k, str) for k in schema["required"])):
        raise SchemaError(f"{path}: required must be a list of strings")
    if "frozen" in schema:
        if path != "$":
            raise SchemaError(f"{path}: frozen is allowed only at the root")
        if not (isinstance(schema["frozen"], list) and all(isinstance(k, str) and k for k in schema["frozen"])):
            raise SchemaError("$: frozen must be a list of dotted paths")
    for key, sub in (schema.get("properties") or {}).items():
        check_schema(sub, f"{path}.{key}")
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]")


def _validate(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
    kinds = schema.get("type")
    if kinds is not None:
        options = kinds if isinstance(kinds, list) else [kinds]
        if not any(_type_ok(value, kind) for kind in options):
            errors.append(f"{path}: expected {' or '.join(options)}, got {type(value).__name__}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")
    if _type_ok(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value!r} is below the minimum {schema['minimum']!r}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value!r} is above the maximum {schema['maximum']!r}")
    if isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: string of length {len(value)} exceeds maxLength {schema['maxLength']}")
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: string of length {len(value)} is under minLength {schema['minLength']}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: {len(value)} items exceed maxItems {schema['maxItems']}")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} items are under minItems {schema['minItems']}")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate(item, schema["items"], f"{path}.{index}", errors)
    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], f"{path}.{key}", errors)
            elif schema.get("additionalProperties", False) is False and "properties" in schema:
                errors.append(f"{path}: unknown key {key!r}")


def lookup(config: Any, dotted: str) -> Any:
    """The value at a dotted path ("tools.0.name"), or a private MISSING sentinel."""
    node = config
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            return _MISSING
    return node


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def validate_config(config: Any, schema: Mapping[str, Any],
                    seed_config: Any = None) -> list[str]:
    """Every violation of `schema` by `config` (empty when valid). With `seed_config`, also every
    `frozen` path whose value differs from the seed's."""
    check_schema(schema)
    errors: list[str] = []
    _validate(config, schema, "$", errors)
    if seed_config is not None:
        for dotted in schema.get("frozen") or []:
            before, after = lookup(seed_config, dotted), lookup(config, dotted)
            if (before is _MISSING) != (after is _MISSING) or (
                    before is not _MISSING and _canonical(before) != _canonical(after)):
                errors.append(f"$.{dotted}: frozen key changed from the seed")
    return errors


__all__ = ["SchemaError", "check_schema", "lookup", "validate_config", "TYPES"]
