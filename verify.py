"""Check that displayed evidence remains traceable to the complete run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from record import inputs


def check(record: dict, root: Path) -> None:
    if record.get("schema_version") == 2:
        from verify_v2 import check_v2
        return check_v2(record, root)
    if record.get("schema_version") != 1:
        raise ValueError("Unknown evidence schema")
    for key, filename in [("protocol_sha256", "protocol.json"), ("code_sha256", "record.py"), ("uv_lock_sha256", "uv.lock")]:
        if record[key] != hashlib.sha256((root / filename).read_bytes()).hexdigest():
            raise ValueError(f"Source digest mismatch: {filename}")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if record["protocol"] != protocol:
        raise ValueError("Embedded protocol differs")
    if record["weight_sha256"] != "1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617":
        raise ValueError("Wrong weight identity")
    context, queries = inputs(protocol)
    if record["context"] != context or record["queries"] != queries:
        raise ValueError("Unexpected input coordinates, ordering or labels")
    conditions = record["conditions"]
    if [c["name"] for c in conditions] != protocol["conditions"]:
        raise ValueError("Missing, reordered or additional conditions")
    if record["environment"]["gpu_used"] is not False or record["environment"]["cpu_threads"] != 2:
        raise ValueError("Wrong execution device or threads")
    for c in conditions:
        labels = [r["inside"] for r in context]
        if c["name"] == "one-label-changed":
            labels[5] = 1 - labels[5]
        if c["context_labels"] != labels:
            raise ValueError("Undeclared context label change")
        names = c["raw_columns"]
        if len(names) != 2 or set(names) != {"0", "1"}:
            raise ValueError("Class identity mismatch")
        if len(c["probabilities"]) != len(queries) or len(c["raw_probabilities"]) != len(queries):
            raise ValueError("Incomplete query output")
        for raw, pair in zip(c["raw_probabilities"], c["probabilities"], strict=True):
            if len(pair) != 2 or len(raw) != 2 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pair):
                raise ValueError("Invalid probability")
            if not math.isclose(sum(pair), 1, abs_tol=1e-6):
                raise ValueError("Probabilities do not sum to 1")
            if pair != [raw[names.index("0")], raw[names.index("1")]]:
                raise ValueError("Probability column identity lost")
        if not math.isfinite(c["inference_seconds"]) or c["inference_seconds"] <= 0:
            raise ValueError("Invalid timing")
    original, changed, repeated = [c["probabilities"] for c in conditions]
    flips = sum((b > a) != (d > c) for (a, b), (c, d) in zip(original, changed, strict=True))
    delta = max(abs(a[1] - b[1]) for a, b in zip(original, changed, strict=True))
    repeat_delta = max(abs(a[i] - b[i]) for a, b in zip(original, repeated, strict=True) for i in (0, 1))
    summary = record["summary"]
    if flips != summary["changed_predictions"] or not math.isclose(delta, summary["max_probability_change"], abs_tol=1e-7) or not math.isclose(repeat_delta, summary["repeat_max_probability_difference"], abs_tol=1e-7):
        raise ValueError("Summary does not follow from raw records")
    if repeat_delta > 1e-6:
        raise ValueError("Fixed repeat exceeds declared tolerance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    check(json.loads(args.record.read_text(encoding="utf-8")), Path(__file__).resolve().parent)
    print("Verified: complete conditions, exact inputs, named classes, probabilities, summaries and source digests")
