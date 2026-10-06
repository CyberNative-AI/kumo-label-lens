"""Download the one immutable checkpoint, anonymously, within a fixed size limit."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
if args.output.exists():
    raise FileExistsError("Refusing to replace an existing checkpoint")
url = f"https://huggingface.co/nvidia/Kumo-Tabular/resolve/{protocol['model_revision']}/{protocol['model_file']}"
digest = hashlib.sha256()
size = 0
try:
    with urllib.request.urlopen(url, timeout=30) as response, args.output.open("xb") as output:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > 160 * 1024 * 1024:
                raise ValueError("Checkpoint exceeds download limit")
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != "1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617":
        raise ValueError("Checkpoint hash mismatch")
except Exception:
    args.output.unlink(missing_ok=True)
    raise
print(f"Verified checkpoint: {size} bytes, SHA-256 {digest.hexdigest()}")
