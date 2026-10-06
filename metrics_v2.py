"""Reach in the frozen query grid, with class ties resolved to class 0."""
import math
import statistics


def reach(before, after, queries, point):
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True))
               if (a[1] > a[0]) != (b[1] > b[0])]
    return {
        "changed_predictions": len(changed),
        "max_probability_change": max(abs(a[1] - b[1]) for a, b in zip(before, after, strict=True)),
        "farthest_changed_distance": max((math.hypot(queries[i]["x"] - point["x"],
                                                     queries[i]["y"] - point["y"]) for i in changed), default=None),
    }


def aggregate(reaches):
    return {key: {"median": statistics.median(values), "max": max(values)}
            for key in ("changed_predictions", "max_probability_change", "farthest_changed_distance")
            for values in [[r[key] if r[key] is not None else 0 for r in reaches]]}


def finding(summary):
    parts = []
    for key, name in [("kumo", "Kumo"), ("knn", "k-NN(3)"), ("lr", "logistic regression")]:
        s = summary[key]
        count, delta, distance = (s[k] for k in ("changed_predictions", "max_probability_change", "farthest_changed_distance"))
        parts.append(f"{name}: changed classes {count['median']:g}/{count['max']:g} of 169, "
                     f"largest probability shift {delta['median'] * 100:.1f}/{delta['max'] * 100:.1f} pp, "
                     f"farthest class change {distance['median']:.2f}/{distance['max']:.2f} units")
    return "Across all 16 single-label flips, median/max reach was " + "; ".join(parts) + "."


def decision(summary):
    # Predeclared distribution-level criterion, never a selected example.
    distinct = any(abs(summary["kumo"][metric][stat] - summary[other][metric][stat]) >= threshold
                   for other in ("knn", "lr")
                   for metric, threshold in (("changed_predictions", 1), ("max_probability_change", .01))
                   for stat in ("median", "max"))
    return "CONTINUE_PRIVATE_REVIEW" if distinct else "STOP"
