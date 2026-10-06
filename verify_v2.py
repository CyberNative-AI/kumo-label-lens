"""Recompute every v2 reach value directly from retained class probabilities."""
import hashlib
import json
import math
import statistics
from record import inputs
from verify_v1 import check as check_v1


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_v2(record, root):
    if record.get("status") != "complete":
        raise ValueError("V2 execution incomplete")
    freeze = json.loads((root / "freeze-v2.json").read_text())
    if record["freeze_sha256"] != digest(root / "freeze-v2.json") or record["input_hashes"] != freeze["files"]:
        raise ValueError("Freeze identity mismatch")
    for name, expected in freeze["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen source mismatch: {name}")
    protocol = json.loads((root / "protocol-v2.json").read_text())
    if protocol != record["protocol"] or record["weight_sha256"] != protocol["weight_sha256"]:
        raise ValueError("Protocol/checkpoint identity mismatch")
    baseline = json.loads((root / "results.json").read_text())
    check_v1(baseline, root)
    context, queries = inputs(protocol)
    if record["context"] != context or record["queries"] != queries:
        raise ValueError("Wrong inputs or order")
    env = record["environment"]
    if env["gpu_used"] is not False or env["cpu_threads"] != 2 or env["sklearn"] != protocol["comparator_settings"]["scikit_learn"]:
        raise ValueError("Wrong execution environment")
    if any(env[k] != baseline["environment"][k] for k in ("torch", "sdm")):
        raise ValueError("Kumo environment changed")
    if not math.isfinite(record["total_record_seconds"]) or not 0 < record["total_record_seconds"] < 180:
        raise ValueError("Recording time exceeded bounds")
    if not 0 < record["maximum_rss_kib"] < 2048 * 1024:
        raise ValueError("Recording memory exceeded bounds")
    if set(record["models"]) != {"kumo", "knn", "lr"}:
        raise ValueError("Missing/additional model")
    summaries = {}
    for key, model in record["models"].items():
        conditions = model["conditions"]
        if [c["name"] for c in conditions] != protocol["conditions"][key]:
            raise ValueError("Incomplete/reordered/extra conditions")
        measured = []
        for index, c in enumerate(conditions):
            flip = index - 1 if index else None
            if c["status"] != "success" or c["flip_index"] != flip:
                raise ValueError("Failed or incorrect condition")
            labels = [r["inside"] for r in context]
            if flip is not None:
                labels[flip] = 1 - labels[flip]
            if c["context_labels"] != labels:
                raise ValueError("Wrong label intervention")
            if c["origin"] != ("retained-v1" if key == "kumo" and index == 0 else "v2-execution"):
                raise ValueError("Execution/reuse attribution mismatch")
            names, raw, probabilities = c["raw_columns"], c["raw_probabilities"], c["probabilities"]
            if len(names) != 2 or set(names) != {"0", "1"} or len(raw) != 169 or len(probabilities) != 169:
                raise ValueError("Incomplete output or wrong class identity")
            for native, pair in zip(raw, probabilities, strict=True):
                if len(native) != 2 or len(pair) != 2 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in pair + native):
                    raise ValueError("Invalid probability")
                if not math.isclose(sum(pair), 1, abs_tol=1e-6) or pair != [native[names.index("0")], native[names.index("1")]]:
                    raise ValueError("Probability normalization/identity lost")
            if not math.isfinite(c["inference_seconds"]) or c["inference_seconds"] <= 0:
                raise ValueError("Invalid condition timing")
            if key != "kumo":
                params = c["estimator_parameters"]
                required = {"n_neighbors": 3, "weights": "uniform", "metric": "minkowski", "p": 2,
                            "algorithm": "auto", "leaf_size": 30, "metric_params": None, "n_jobs": None} if key == "knn" else {
                            "solver": "lbfgs", "C": 1.0, "max_iter": 100, "l1_ratio": 0.0,
                            "class_weight": None, "random_state": None, "dual": False, "fit_intercept": True,
                            "intercept_scaling": 1, "n_jobs": None, "penalty": "deprecated", "tol": .0001,
                            "verbose": 0, "warm_start": False}
                if params != required:
                    raise ValueError("Wrong/changed comparator recipe")
            if key == "kumo" and index == 0:
                if c["source"] != {"file": "results.json", "sha256": digest(root / "results.json"), "condition": "original"}:
                    raise ValueError("Wrong baseline provenance")
                if any(c[k] != v for k, v in baseline["conditions"][0].items()):
                    raise ValueError("Retained baseline changed")
            if index:
                changes = [i for i, (a, b) in enumerate(zip(conditions[0]["probabilities"], probabilities, strict=True))
                           if (a[1] > a[0]) != (b[1] > b[0])]
                distances = [math.hypot(queries[i]["x"] - context[flip]["x"], queries[i]["y"] - context[flip]["y"]) for i in changes]
                expected = {"changed_predictions": len(changes),
                            "max_probability_change": max(abs(a[1] - b[1]) for a, b in zip(conditions[0]["probabilities"], probabilities, strict=True)),
                            "farthest_changed_distance": max(distances) if distances else None}
                if c["reach"] != expected:
                    raise ValueError("Per-flip reach disagrees with probabilities")
                measured.append(expected)
        summaries[key] = {}
        for metric in measured[0]:
            values = [r[metric] if r[metric] is not None else 0 for r in measured]
            summaries[key][metric] = {"median": statistics.median(values), "max": max(values)}
    if record["summary"] != summaries:
        raise ValueError("Aggregate summary disagrees with all 16 flips")
    distinct = any(abs(summaries["kumo"][metric][stat] - summaries[other][metric][stat]) >= threshold
                   for other in ("knn", "lr") for stat in ("median", "max")
                   for metric, threshold in (("changed_predictions", 1), ("max_probability_change", .01)))
    if record["decision"] != ("CONTINUE_PRIVATE_REVIEW" if distinct else "STOP"):
        raise ValueError("Stop decision disagrees with frozen rule")
