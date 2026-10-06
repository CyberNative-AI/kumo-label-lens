"""Embed verified records into a dependency-free offline page."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
from verify import check
from visualization import overlays, finding_lead, summary_rows

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("record", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
record = json.loads(args.record.read_text(encoding="utf-8"))
check(record, root)
script = (root / "explorer.js").read_text(encoding="utf-8").rstrip()
summary = record["summary"]
if record.get('schema_version') != 2:
    raise ValueError('The v2 page requires results-v2.json; retained v1 page is in its original bundle')
observation = finding_lead(summary)
surprise = 'In this fixed grid, Kumo’s label flips changed classes farther from the flipped row than the classical comparators did at the median and maximum. k-NN had larger maximum probability shifts, while logistic regression usually kept the same classes. These are different behaviors, not a ranking of model quality. Every flip and unchanged query is retained.'
page = (root / "page.html").read_text(encoding="utf-8")
for key, value in {
    "__SCRIPT_HASH__": base64.b64encode(hashlib.sha256(script.encode()).digest()).decode(),
    "__RECORD__": json.dumps(record, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c"),
    "__OBSERVATION__": observation,
    "__SUMMARY_ROWS__": summary_rows(summary),
    "__OVERLAYS__": json.dumps(overlays(record), ensure_ascii=False, allow_nan=False),
    "__SURPRISE__": surprise,
    "__SCRIPT__": script,
}.items():
    page = page.replace(key, value)
args.output.write_text(page, encoding="utf-8")
print(f"Built {args.output.name}: {len(page.encode())} bytes")
