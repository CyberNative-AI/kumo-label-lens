"""Build discrete class overlays from recorded probabilities, without inference."""
import math


def rgb(probability):
    # Match JavaScript Math.round for this nonnegative probability scale.
    return [math.floor(a + (b - a) * probability + .5)
            for a, b in zip((12, 25, 42), (56, 214, 198), strict=True)]


def luminance(channels):
    values = [v / 255 for v in channels]
    values = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in values]
    return sum(v * w for v, w in zip(values, (.2126, .7152, .0722), strict=True))


def mark_ink(probability):
    light = luminance(rgb(probability))
    return "#000000" if (light + .05) / .05 >= 1.05 / (light + .05) else "#ffffff"


def overlays(record):
    output = {}
    for key, model in record["models"].items():
        baseline = [int(p[1] > p[0]) for p in model["conditions"][0]["probabilities"]]
        conditions = []
        for condition in model["conditions"]:
            predictions = [int(p[1] > p[0]) for p in condition["probabilities"]]
            edges = []
            # Boundaries lie halfway between adjacent query centres. These are
            # discrete argmax boundaries, not interpolated model predictions.
            for i, value in enumerate(predictions):
                x, y = i % 13, i // 13
                cx, cy = 24 + x * 16, 216 - y * 16
                if x < 12 and value != predictions[i + 1]:
                    edges.append([cx + 8, cy - 8, cx + 8, cy + 8])
                if y < 12 and value != predictions[i + 13]:
                    edges.append([cx - 8, cy - 8, cx + 8, cy - 8])
            changed = [i for i, (a, b) in enumerate(zip(baseline, predictions, strict=True)) if a != b]
            flip = condition["flip_index"]
            ring = None
            if flip is not None and changed:
                point = record["context"][flip]
                radius = max(math.hypot(record["queries"][i]["x"] - point["x"],
                                        record["queries"][i]["y"] - point["y"]) for i in changed)
                ring = {"cx": 120 + point["x"] * 80, "cy": 120 - point["y"] * 80,
                        "radius_units": radius, "radius_px": radius * 80}
            conditions.append({"classes": predictions, "boundary_edges": edges,
                               "changed_cells": changed, "ring": ring,
                               "mark_inks": [mark_ink(p[1]) for p in condition["probabilities"]]})
        output[key] = conditions
    return output


def finding_lead(summary):
    kumo, knn, lr = (summary[key] for key in ("kumo", "knn", "lr"))
    return (f"Across all 16 flips, one wrong label changed Kumo’s predicted class at a median of "
            f"{kumo['changed_predictions']['median']:g} of 169 grid points, up to "
            f"{kumo['farthest_changed_distance']['max']:.2f} units from the flipped row. "
            f"k-NN(3) changed a median of {knn['changed_predictions']['median']:g}, never beyond "
            f"{knn['farthest_changed_distance']['max']:.2f} units. "
            f"Logistic regression changed at most {lr['changed_predictions']['max']:g}.")


def summary_rows(summary):
    rows = []
    for key, name in (("kumo", "Kumo"), ("knn", "k-NN(3)"), ("lr", "Logistic regression")):
        cells = [f'<th scope="row">{name}</th>']
        for metric, label, scale, precision in (
            ("changed_predictions", "Changed classes", 1, None),
            ("max_probability_change", "Largest probability shift", 100, 1),
            ("farthest_changed_distance", "Farthest class change", 1, 2),
        ):
            values = summary[key][metric]
            pair = [f"{values[stat] * scale:g}" if precision is None else f"{values[stat] * scale:.{precision}f}"
                    for stat in ("median", "max")]
            cells.append(f'<td data-label="{label}">{pair[0]} / {pair[1]}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return '\n'.join(rows)
