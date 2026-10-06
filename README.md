# Kumo Label Lens

How far does a single wrong context label reach? Open `index.html` offline, select
any of the 16 label buttons, and compare before/after maps for NVIDIA Kumo Tabular
Small, 3-nearest neighbors and logistic regression. Inspect every query with the
X/Y sliders. Solid black/white lines separate adjacent recorded grid classes
(argmax; class-0 ties). After maps outline and hatch every changed cell, and show
a dashed circle centered on the flipped row with radius equal to the farthest
changed query. No change means no ring. Rings can extend past the map edge. Marks
use black or white ink chosen for at least 3:1 contrast against the probability
fill; boundaries and rings use both. All overlays are derived at build time from
raw probabilities, with no interpolation. No model executes in the browser and no network request is needed.

**Recorded from model execution:** 51 conditions are retained: one reused Kumo
baseline and 50 new executions (16 Kumo flips and 17 conditions per classical
model). **Prerecorded offline visualization:** the page selects those saved
records, with no interpolation or new prediction. `results-v2.json` is primary
evidence; the maps illustrate it. Evidence: **Executed; Artifact checked** by a
separate internal reviewer in the same organization. Not independently reproduced.

## Full finding and limits

Across all 16 single-label flips, median/max reach was Kumo: changed classes
13.5/14 of 169, largest probability shift 22.3/29.3 percentage points, farthest
class change 1.32/1.70 units; k-NN(3): changed classes 8.5/11 of 169, largest
probability shift 33.3/33.3 points, farthest class change 0.58/0.58 units; logistic
regression: changed classes 0/1 of 169, largest probability shift 19.2/26.2 points,
farthest class change 0.00/0.42 units.

These are descriptive observations from one tiny two-feature synthetic table,
one seed, one checkpoint and fixed recipes. There is no accuracy, calibration,
benchmark reproduction or general robustness claim. Recipes differ in capacity
and inductive bias. Logistic regression's linear boundary is poorly suited to
circular labels; its mostly unchanged classes are not evidence of superior
robustness. Probabilities can move without crossing a class boundary. k-NN's
equidistant-neighbor choice can depend on the fixed row order.

For each flip, reach records the count of argmax changes over the 169 queries,
maximum absolute change in P(inside), and Euclidean distance from the flipped
context row to the farthest query with a changed class. Distance is in synthetic
coordinate units; if no class changes, it is `null`, displayed as “none,” and
counts as 0 when aggregating the 16 distances. Tied class probabilities resolve
to class 0 (outside). Every flip and query is retained, without selection.

The frozen distribution-level stop test passed: Kumo differs from a comparator
by at least one query in median/max changed-class count or at least one
percentage point in median/max largest probability shift. The recorded decision
is `CONTINUE_PRIVATE_REVIEW`; that is not publication approval or outside use.

## Comparator and sources

The nearest named alternative is scikit-learn's
[Classifier comparison](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
example, which illustrates classical decision boundaries on 2-D synthetic
inputs. This lens adds an in-context tabular model, all 16 label interventions
and full saved probabilities. It is narrower: one fixed table and no accuracy
comparison. We do not reproduce that example's datasets, scaling or test scores.

The [NVIDIA release article](https://huggingface.co/blog/nvidia/kumo-tabular)
(September 29, 2026) and
[model card](https://huggingface.co/nvidia/Kumo-Tabular) are architecture and
benchmark sources. The community
[WebGPU Space by mrfakename](https://huggingface.co/spaces/mrfakename/kumo-tabular-webgpu)
is an additional source; its predictions were not tested here.
The [k-NN API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
and [logistic-regression API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
document the selected classical recipes.

## Exact execution and provenance

- Kumo upstream revision: `391961ae82b4bf54af1aca31e9fc0a0c678117ce`.
- Checkpoint: `nvidia/Kumo-Tabular`, revision
  `3c3e10bbdb590ace29e7026847f92db3c603096d`, `small/classifier.pt`.
- SHA-256: `1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617`.
- Kumo: CPU float32, 1 estimator, upstream default preprocessing recipe,
  seed 20261006, 2 threads, deterministic torch algorithms, no weight updates.
- Classical models: scikit-learn 1.9.1,
  `KNeighborsClassifier(n_neighbors=3)` and `LogisticRegression()`, all other
  defaults. Every condition is fitted from scratch on the same original-order
  float32 feature arrays. No added scaling or feature engineering. Estimator
  parameters and warning lists are retained in every new classical record.
- Inputs: 4 × 4 context grid on [-0.9, -0.3, 0.3, 0.9]; 13 × 13 query grid on
  [-1.2, 1.2], increment 0.2, y ascending then x. Labels:
  `inside = int(x*x + y*y < 0.36)`. Company-authored synthetic data; no personal
  or customer information. Exact inputs are embedded in both protocols/results.
- Python 3.12.13, PyTorch 2.14.0+cpu, NumPy 2.5.3, AMD Ryzen 7 3700X.
  V2 recorder after imports: 8.499 s, including a 0.305 s model load;
  process maximum RSS 641,692 KiB. The container invocation including startup
  and cleanup took 16.879 s. Setup/downloads are excluded from recorder timing.
- CPU Docker: 2 CPUs, 2 GiB memory, no swap, 128 PIDs, 180-second outer limit,
  network denied, read-only inputs/environment/checkpoint, one writable output
  directory, 64 MiB tmpfs, dropped capabilities, no GPU mounts. Exit 0 and
  container removal are recorded in `execution-v2.json`. Incremental spend: $0.
- Base image:
  `python@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36`.

`protocol-v2.json` and its execution sources were frozen at
2026-10-06 16:58:07 UTC, before the v2 container started at 16:58:12 UTC.
`freeze-v2.json` binds exact source, protocol, dependency and retained-baseline
hashes. A separate `environment-v2/uv.lock` adds the comparators while preserving
v1's frozen environment files. No v2 protocol amendment or execution failure
occurred; every new condition's warning list is empty.

V1 is retained evidence and the build base, not the selected deliverable or a
publication candidate. `protocol.json`, `results.json`, `record.py`,
`pyproject.toml`, `uv.lock`, `amendments.json` and `provenance.json` are historical
v1 files. Its first attempt failed because the consumer required a fixed class
column order. The append-only amendment preserves that failure and the repair:
bind probabilities to named classes 0/1, retaining both native and canonical
columns. V2 uses that corrected mapping. The original v1 baseline is reused
exactly; no v1 inference was rerun. V1's zero-difference same-seed repeat is a
harness repeat, not independent scientific reproduction.

## Verify without executing models

From this folder, on Python 3.12 through uv:

```sh
uv run --python 3.12 --no-project python verify.py results.json
uv run --python 3.12 --no-project python verify.py results-v2.json
uv run --python 3.12 --no-project python -m unittest -v test_evidence.py test_v2.py test_visualization.py
uv run --python 3.12 --no-project python build_page.py results-v2.json --output .lab/rebuilt.html
```

Create `.lab/` before the last command and compare rebuilt bytes to `index.html`.
No model dependency or checkpoint is needed for verification. Checks reject
incomplete/failed conditions, wrong class identity, changed baseline, nonfinite
probabilities, changed comparator settings, wrong reach/summary/stop decisions,
and mismatched frozen inputs. Browser checks compare every displayed map cell,
all 8,112 query readouts and all 144 reach readouts to `results-v2.json`.

## Execute v2 on a reader-controlled isolated CPU machine

Do not execute downloaded model/library code on an administrative/control-plane
machine. On a reader-controlled isolated executor, prepare the pinned v2
environment with Python 3.12 and uv. Preparation needs public dependency
network access; use a bounded 240-second / 2-CPU / 2-GiB container and a ≤1-GiB
download budget. Existing downloaded weights can be reused after hash checking.
Inference requires no network. This bundle redistributes no weights or upstream
source. Preserve any failure, partial output and environment differences.

```sh
kumo_root="$(pwd)"
kumo_uv="$(command -v uv)"
mkdir -p .lab/build
# Create the environment with the same interpreter path as the inference image.
timeout --signal=TERM --kill-after=10 240 docker run --rm \
  --cpus 2 --memory 2g --memory-swap 2g --pids-limit 128 \
  --read-only --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  -e UV_CACHE_DIR=/build/cache -e UV_PROJECT_ENVIRONMENT=/build/environment \
  -e UV_PYTHON_DOWNLOADS=never \
  --mount "type=bind,src=$kumo_root,dst=/work,readonly" \
  --mount "type=bind,src=$kumo_root/.lab/build,dst=/build" \
  --mount "type=bind,src=$kumo_uv,dst=/usr/local/bin/uv,readonly" \
  --workdir /work/environment-v2 \
  python@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36 \
  uv sync --frozen
uv run --python 3.12 --no-project python download_weights.py .lab/weights.pt
uv run --python 3.12 --no-project python run_v2.py \
  --environment .lab/build/environment --weights .lab/weights.pt \
  --output-directory .lab/rerun-v2
uv run --python 3.12 --no-project python verify.py .lab/rerun-v2/results-v2.json
```

`run_v2.py` uses `shell=False`, read-only mounts and the documented limits. It
records timeout/exit/cleanup receipts and retains partial output. The Kumo
before-map remains the retained v1 result even in this command: this re-executes
only v2 additions, not the v1 baseline. Reusing that baseline is not independent
reproduction of v1 or scientific replication of the comparative finding.
Abort on source/checkpoint mismatch, wrong classes/device, nonfinite output,
timeout or memory exhaustion. No alternate seeds or recipes after failure.

## License, contributors and corrections

Original code and synthetic data: Apache-2.0. Kumo weights:
[OpenMDW 1.1](https://huggingface.co/nvidia/Kumo-Tabular/blob/3c3e10bbdb590ace29e7026847f92db3c603096d/LICENSE),
which permits sharing generated outputs. scikit-learn: BSD-3-Clause; no
scikit-learn code or weights are redistributed. Dependencies are identified by
lockfiles. AI-assisted implementation and writing by CyberNative AI LLC; model
outputs from NVIDIA Kumo Tabular and scikit-learn. No human author, testimony,
approval or independent review is invented. CyberNative AI LLC is the accountable
publisher; corrections: hello@cybernative.ai. Material public corrections will
be dated and linked from the affected artifact.
