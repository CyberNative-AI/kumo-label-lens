"""Record all v2 conditions in an isolated, network-denied CPU executor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import time
import warnings

from record import inputs, sha256
from metrics_v2 import aggregate, decision, reach
from verify_v1 import check as check_v1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    freeze = json.loads((root / "freeze-v2.json").read_text())
    for name, digest in freeze["files"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen input mismatch: {name}")
    protocol = json.loads((root / "protocol-v2.json").read_text())
    if sha256(args.weights) != protocol["weight_sha256"]:
        raise ValueError("Weight hash mismatch")
    import numpy as np
    import sklearn
    import torch
    import sdm
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sdm import CategoricalTensor, Stype, TableTensor

    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(protocol["seed"])
    loaded = time.perf_counter()
    model = sdm.models.KumoTabular(task="classification", size="small", pretrained=False, device="cpu")
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    next(iter(model.models.values())).load_state_dict(weights, strict=True)
    del weights
    model.eval()
    if not all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters()):
        raise ValueError("Wrong device or dtype")
    load_seconds = time.perf_counter() - loaded
    baseline = json.loads((root / "results.json").read_text())
    check_v1(baseline, root)
    context, queries = inputs(protocol)
    # Identical float32 feature values, identical row order for every model.
    xc = np.array([[r["x"], r["y"]] for r in context], dtype=np.float32)
    xq = np.array([[r["x"], r["y"]] for r in queries], dtype=np.float32)
    def features(array):
        return TableTensor(columns={Stype.numerical: ("x", "y")}, numerical=torch.from_numpy(array))
    x_context, x_query = features(xc), features(xq)
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), "unavailable")
    record = {
        "schema_version": 2, "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "freeze_sha256": sha256(root / "freeze-v2.json"),
        "input_hashes": freeze["files"], "weight_sha256": sha256(args.weights),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "sdm": sdm.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
                        "cpu_model": cpu_model, "cpu_threads": 2, "gpu_used": False},
        "context": context, "queries": queries, "model_load_seconds": load_seconds,
        "models": {key: {"conditions": []} for key in ("kumo", "knn", "lr")},
        "status": "running",
    }
    def save():
        record["total_record_seconds"] = time.perf_counter() - start
        record["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    def validate(raw, columns):
        if len(columns) != 2 or set(columns) != {"0", "1"} or raw.shape != (169, 2):
            raise ValueError("Class identity or shape mismatch")
        ordered = raw[:, [columns.index("0"), columns.index("1")]]
        if not np.isfinite(raw).all() or not ((raw >= 0) & (raw <= 1)).all() or not np.allclose(raw.sum(-1), 1, atol=1e-6, rtol=0):
            raise ValueError("Invalid probability output")
        return ordered.tolist()

    original = dict(baseline["conditions"][0])
    original.update(status="success", origin="retained-v1", flip_index=None,
                    source={"file": "results.json", "sha256": sha256(root / "results.json"), "condition": "original"})
    record["models"]["kumo"]["conditions"].append(original)
    save()
    for key in ("kumo", "knn", "lr"):
        for flip in ([*range(16)] if key == "kumo" else [None, *range(16)]):
            labels = [r["inside"] for r in context]
            if flip is not None:
                labels[flip] = 1 - labels[flip]
            condition = {"name": "original" if flip is None else f"flip-{flip:02}",
                         "flip_index": flip, "context_labels": labels, "origin": "v2-execution"}
            t = time.perf_counter()
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if key == "kumo":
                        y = TableTensor(columns={Stype.categorical: ("inside",)},
                                        categorical=CategoricalTensor.from_tensor(torch.tensor(labels).unsqueeze(-1)))
                        torch.manual_seed(protocol["seed"])
                        generator = torch.Generator(device="cpu").manual_seed(protocol["seed"])
                        with torch.inference_mode():
                            output = model(x_context=x_context, y_context=y, x_query=x_query, num_estimators=1, generator=generator)
                        columns = list(output.columns[Stype.numerical])
                        raw = output.numerical.numpy()
                    else:
                        estimator = KNeighborsClassifier(n_neighbors=3) if key == "knn" else LogisticRegression()
                        estimator.fit(xc, np.array(labels))
                        raw = estimator.predict_proba(xq)
                        columns = [str(c) for c in estimator.classes_]
                        condition["estimator_parameters"] = estimator.get_params()
                    condition.update(status="success", raw_columns=columns, raw_probabilities=raw.tolist(),
                                     probabilities=validate(raw, columns), warnings=[str(w.message) for w in caught])
                if flip is not None:
                    before = record["models"][key]["conditions"][0]["probabilities"]
                    condition["reach"] = reach(before, condition["probabilities"], queries, context[flip])
            except Exception as error:
                condition.update(status="failed", error_type=type(error).__name__, error=str(error))
                record["status"] = "aborted"
                record["models"][key]["conditions"].append(condition)
                save()
                raise
            condition["inference_seconds"] = time.perf_counter() - t
            record["models"][key]["conditions"].append(condition)
            save()
            print(json.dumps({"model": key, "condition": condition["name"], "seconds": condition["inference_seconds"]}), flush=True)
    record["summary"] = {key: aggregate([c["reach"] for c in value["conditions"][1:]]) for key, value in record["models"].items()}
    record["decision"] = decision(record["summary"])
    record["status"] = "complete"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    save()


if __name__ == "__main__":
    main()
