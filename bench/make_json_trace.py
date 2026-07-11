"""Generate a JSON / structured-output rehearsal trace for round 2.

Round 1 (`data/input/trace-round1.jsonl`) is 100% plain chat — nothing exercises the
xgrammar path, so it can't measure jump-forward. This emits `data/input/trace-json.jsonl`:
compact, varied `response_format: json_schema` requests whose deterministic structural
tokens (keys, braces, quotes, commas) are exactly what jump-forward skips.

Schemas span flat / nested / array / enum shapes so the jump hit-rate isn't measured on a
single degenerate case. Bodies mirror round-1 fields (model, messages, temperature=0, seed,
max_tokens) so `bench/replay.py --trace data/input/trace-json.jsonl` works unchanged.

    python bench/make_json_trace.py            # -> data/input/trace-json.jsonl (default 24 reqs)
    python bench/make_json_trace.py --n 60 --out /tmp/big.jsonl

Deterministic: same args -> byte-identical trace (no RNG), so A/B runs compare like-for-like.
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "input" / "trace-json.jsonl"
MODEL = "Qwen3.5-2B"  # matches --served-model-name in docker-compose-optimized.yaml

# (name, json_schema, extraction prompt). Kept small but structurally diverse: the point is
# to vary how much of the output is grammar-forced, not to be an exhaustive schema zoo.
SCHEMAS = [
    (
        "flat_person",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
            },
            "required": ["name", "age", "email"],
            "additionalProperties": False,
        },
        "Extract the person as JSON: Dr. Alice Nguyen, 41, reachable at alice@example.com.",
    ),
    (
        "enum_ticket",
        {
            "type": "object",
            "properties": {
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "category": {"type": "string", "enum": ["bug", "feature", "question"]},
                "resolved": {"type": "boolean"},
            },
            "required": ["priority", "category", "resolved"],
            "additionalProperties": False,
        },
        "Classify this ticket as JSON: 'The login page crashes on submit — please fix ASAP.'",
    ),
    (
        "nested_order",
        {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "customer": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "vip": {"type": "boolean"},
                    },
                    "required": ["name", "vip"],
                    "additionalProperties": False,
                },
                "total": {"type": "number"},
            },
            "required": ["order_id", "customer", "total"],
            "additionalProperties": False,
        },
        "Extract the order as JSON: order X-1007 for VIP customer Sam Park, total 249.99.",
    ),
    (
        "array_items",
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "qty": {"type": "integer"},
                        },
                        "required": ["sku", "qty"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
        "List the cart items as JSON: 2x SKU A100, 1x SKU B200, 5x SKU C300.",
    ),
]


def _record(i: int, schema_name: str, schema: dict, prompt: str) -> dict:
    return {
        "request_id": f"json-{i:04d}",
        "timestamp_ms": i * 250,  # 4 req/s open-loop; replay.py honors this
        "workload_type": "structured_json",
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You output only JSON matching the schema."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 200,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=24, help="number of requests (cycles the schemas)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for i in range(args.n):
            name, schema, prompt = SCHEMAS[i % len(SCHEMAS)]
            f.write(json.dumps(_record(i, name, schema, prompt)) + "\n")
    print(f"wrote {args.n} structured requests -> {args.out}")


def _self_check() -> None:
    """No I/O: every generated body is valid JSON and carries a json_schema response_format."""
    for i in range(len(SCHEMAS) * 2):
        name, schema, prompt = SCHEMAS[i % len(SCHEMAS)]
        rec = _record(i, name, schema, prompt)
        json.dumps(rec)  # round-trips
        rf = rec["body"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] is schema
        assert rec["body"]["temperature"] == 0 and rec["body"]["seed"] == 42
    print("make_json_trace self-check ok")


if __name__ == "__main__":
    import sys

    if "--self-check" in sys.argv:
        _self_check()
    else:
        main()
