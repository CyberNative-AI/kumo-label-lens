"""Execute the frozen payload in a bounded CPU Docker container; retain receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    name = "kumo-label-lens-" + uuid.uuid4().hex[:12]
    image = "python@sha256:229a2c5bfa27522db7815ea81f9bed70af17ccb9de9fc7ad142b1877b5830d36"
    command = ["docker", "run", "--name", name, "--rm", "--network", "none", "--cpus", "2",
               "--memory", "2g", "--memory-swap", "2g", "--pids-limit", "128", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}",
               "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "OPENBLAS_NUM_THREADS=2", "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2",
               "--mount", f"type=bind,src={root},dst=/work,readonly",
               "--mount", f"type=bind,src={args.environment.resolve()},dst=/environment,readonly",
               "--mount", f"type=bind,src={args.weights.resolve()},dst=/weights.pt,readonly",
               "--mount", f"type=bind,src={output},dst=/output", "--workdir", "/work", image,
               "/environment/bin/python", "record_v2.py", "--weights", "/weights.pt", "--output", "/output/results-v2.json"]
    receipt = {"started_at": datetime.now(timezone.utc).isoformat(), "image": image,
               "bounds": {"network": "none", "cpu_cores": 2, "memory_mib": 2048, "swap_mib": 0,
                          "pids": 128, "seconds": 180, "gpu_mounts": 0, "tmpfs_mib": 64},
               "freeze_sha256": hashlib.sha256((root / "freeze-v2.json").read_bytes()).hexdigest()}
    try:
        with (output / "execution.log").open("w") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
        receipt["exit_code"] = process.returncode
    except subprocess.TimeoutExpired:
        receipt.update(exit_code=124, abort="180 second limit")
    finally:
        cleanup = subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20, check=False)
        # --rm already removes completed containers. Verify absence without listing others.
        remaining = subprocess.run(["docker", "container", "ls", "--all", "--filter", f"name=^{name}$", "--format", "{{.ID}}"],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
        receipt["container_removed"] = not remaining.stdout.strip()
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        (output / "execution-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    if receipt["exit_code"] or not receipt["container_removed"]:
        raise SystemExit(receipt["exit_code"] or 1)


if __name__ == "__main__":
    main()
