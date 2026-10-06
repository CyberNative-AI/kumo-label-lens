"""Record every declared condition. Run in an isolated CPU executor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import time


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(protocol: dict) -> tuple[list[dict], list[dict]]:
    def rows(axis: list[float]) -> list[dict]:
        return [
            {"x": x, "y": y, "inside": int(x * x + y * y < 0.36)}
            for y in axis for x in axis
        ]
    return rows(protocol["context_axis"]), rows(protocol["query_axis"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if sha256(args.weights) != args.weight_sha256:
        raise ValueError("Weight SHA-256 mismatch")
    import torch
    import sdm
    from sdm import CategoricalTensor, Stype, TableTensor

    start = time.perf_counter()
    torch.set_num_threads(protocol["settings"]["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(protocol["seed"])
    load_start = time.perf_counter()
    model = sdm.models.KumoTabular(
        task="classification", size="small", pretrained=False, device="cpu"
    )
    # Loading a pinned file explicitly avoids the upstream floating tag and network.
    weights = torch.load(args.weights, map_location="cpu", weights_only=True)
    next(iter(model.models.values())).load_state_dict(weights, strict=True)
    del weights
    model.eval()
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in model.parameters())
    model_load_seconds = time.perf_counter() - load_start
    context, queries = inputs(protocol)

    def features(rows: list[dict]) -> TableTensor:
        return TableTensor(
            columns={Stype.numerical: ("x", "y")},
            numerical=torch.tensor([[r["x"], r["y"]] for r in rows], dtype=torch.float32),
        )

    x_context, x_query = features(context), features(queries)
    conditions = []
    for name in protocol["conditions"]:
        labels = [r["inside"] for r in context]
        if name == "one-label-changed":
            index = protocol["changed_context_index"]
            labels[index] = 1 - labels[index]
        y_context = TableTensor(
            columns={Stype.categorical: ("inside",)},
            categorical=CategoricalTensor.from_tensor(torch.tensor(labels).unsqueeze(-1)),
        )
        torch.manual_seed(protocol["seed"])
        generator = torch.Generator(device="cpu").manual_seed(protocol["seed"])
        condition_start = time.perf_counter()
        with torch.inference_mode():
            output = model(
                x_context=x_context, y_context=y_context, x_query=x_query,
                num_estimators=1, generator=generator,
            )
        elapsed = time.perf_counter() - condition_start
        probs = output.numerical
        assert probs.shape == (len(queries), 2)
        columns = list(output.columns[Stype.numerical])
        assert len(columns) == 2 and set(columns) == {"0", "1"}, columns
        raw_probabilities = probs.tolist()
        # The default recipe may shuffle classes. Bind by name, never position.
        probs = probs[:, [columns.index("0"), columns.index("1")]]
        assert torch.isfinite(probs).all()
        assert (probs >= 0).all() and (probs <= 1).all()
        assert torch.allclose(probs.sum(-1), torch.ones(len(queries)), atol=1e-6)
        conditions.append({
            "name": name, "context_labels": labels, "inference_seconds": elapsed,
            "raw_columns": columns, "raw_probabilities": raw_probabilities,
            "probabilities": probs.tolist(),
        })
        print(json.dumps({"condition": name, "seconds": elapsed}), flush=True)
    original = torch.tensor(conditions[0]["probabilities"])
    changed = torch.tensor(conditions[1]["probabilities"])
    repeated = torch.tensor(conditions[2]["probabilities"])
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unavailable")
    record = {
        "schema_version": 1, "recorded_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(root / "protocol.json"),
        "code_sha256": sha256(Path(__file__)), "uv_lock_sha256": sha256(root / "uv.lock"),
        "weight_sha256": args.weight_sha256, "protocol": protocol,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "sdm": sdm.__version__, "architecture": platform.machine(), "cpu_model": cpu_model, "cpu_threads": torch.get_num_threads(), "gpu_used": False},
        "context": context, "queries": queries, "conditions": conditions,
        "model_load_seconds": model_load_seconds,
        "total_record_seconds": time.perf_counter() - start,
        "maximum_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "summary": {
            "changed_predictions": int((original.argmax(-1) != changed.argmax(-1)).sum()),
            "max_probability_change": float((original[:, 1] - changed[:, 1]).abs().max()),
            "repeat_max_probability_difference": float((original - repeated).abs().max()),
        },
    }
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
