"""Controlled end-to-end probes of writer concurrency and storage placement."""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

GPU_MEMORY_LIMIT_MIB = 77500
MAX_SECONDS = 900
ACTIVE_MEMORY_MIB = 40000
NVIDIA_SMI = shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi"


def process_identity(pid: str) -> str | None:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def prepare(root: Path, config: Path, count: int) -> None:
    root.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(config.read_text())
    rows = []
    missing = 0
    for line in Path(cfg["dataset"]["input_manifest"]).read_text().splitlines():
        row = json.loads(line)
        paths = row["videos"] if isinstance(row["videos"], list) else [row["videos"]]
        for path in paths:
            if Path(path).is_file():
                rows.append({**row, "videos": [path]})
            else:
                missing += 1
    random.Random(20260916).shuffle(rows)  # noqa: S311 - reproducible benchmark ordering
    selected = rows[:count]
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in selected))
    cfg["dataset"]["input_manifest"] = str(manifest.resolve())
    for workers in (2, 8):
        cfg["stages"]["clip_writer"]["num_workers"] = workers
        cfg["stages"]["clip_writer"]["cpus"] = {2: 1, 8: 0.5}[workers]
        (root / f"writer{workers}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    info = {
        "count": len(selected),
        "missing_excluded": missing,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_config": str(config.resolve()),
    }
    (root / "corpus.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


def run(  # noqa: C901, PLR0912, PLR0913, PLR0915
    root: Path,
    workers: int,
    storage: str,
    label: str | None,
    reuse_detector: bool,
    fast_metadata: bool,
    rebalance_cpu: bool,
    config_override: Path | None = None,
) -> None:
    case = label or f"writer{workers}-{storage}"
    result_path = root / f"{case}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    gpu_ids = subprocess.check_output(  # noqa: S603 - local diagnostic executable
        [NVIDIA_SMI, "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    monitored = {
        line.split(",")[1].strip() for line in gpu_ids.splitlines() if int(line.split(",")[0]) in (4, 5, 6, 7)
    }
    known_workers: dict[str, str | None] = {}

    def foreign_processes(*, allow_workers: bool = True) -> list[str]:
        processes = subprocess.check_output(  # noqa: S603 - local diagnostic executable
            [
                NVIDIA_SMI,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        foreign = []
        for line in processes.splitlines():
            gpu, pid, name, _memory = [value.strip() for value in line.split(",", 3)]
            if gpu not in monitored:
                continue
            identity = process_identity(pid)
            if allow_workers and "ray::" in name:
                known_workers[pid] = identity
                continue
            # NVML can retain a dead worker's GPU context briefly without its process name.
            if (
                allow_workers
                and name == "[No data]"
                and pid in known_workers
                and (identity is None or identity == known_workers[pid])
            ):
                continue
            foreign.append(line)
        return foreign

    foreign = foreign_processes(allow_workers=False)
    if foreign:
        message = f"GPUs are occupied by unrelated processes: {foreign}"
        raise RuntimeError(message)
    output = Path(tempfile.mkdtemp(prefix=f"curator-probe-{case}-")) if storage == "local" else root / f"{case}-output"
    repo = Path(__file__).resolve().parents[3]
    config = config_override or root / f"writer{workers}.yaml"
    if reuse_detector or fast_metadata or rebalance_cpu or config_override:
        cfg = yaml.safe_load(config.read_text())
        cfg["dataset"]["input_manifest"] = str((root / "manifest.jsonl").resolve())
        if config_override is None or reuse_detector:
            cfg["stages"]["video_scene_split"]["params"]["reuse_detector_process"] = reuse_detector
        if config_override is None or fast_metadata:
            cfg["stages"]["clip_writer"]["params"]["use_pyav_metadata"] = fast_metadata
        if rebalance_cpu:
            cfg["stages"]["video_scene_split"]["num_workers"] = 12
            cfg["stages"]["clip_transcoding"]["num_workers"] = 24
            cfg["stages"]["uniform_3_frame_sampling"]["num_workers"] = 8
        config = root / f"{case}.yaml"
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = ["bash", str(repo / "tutorials/video/run.sh"), str(output), str(config)]
    start = time.monotonic()
    samples = []
    reason = None
    with (root / f"{case}-launcher.log").open("w") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)  # noqa: S603
        while proc.poll() is None:
            values = subprocess.check_output(  # noqa: S603 - local diagnostic executable
                [NVIDIA_SMI, "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            sample = {"elapsed": time.monotonic() - start, "gpus": {}}
            for line in values.splitlines():
                gpu, util, memory = map(int, line.split(","))
                if gpu in (4, 5, 6, 7):
                    sample["gpus"][gpu] = {"util": util, "memory_mib": memory}
            samples.append(sample)
            foreign = foreign_processes()
            if foreign:
                sample["foreign_processes"] = foreign
                reason = f"Unrelated GPU workload appeared: {foreign}"
            if any(v["memory_mib"] > GPU_MEMORY_LIMIT_MIB for v in sample["gpus"].values()):
                reason = reason or "GPU memory safety stop above 77500 MiB"
            if sample["elapsed"] > MAX_SECONDS:
                reason = "900 second timeout"
            if reason:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                break
            time.sleep(1)
        code = proc.wait(timeout=30)
    elapsed = time.monotonic() - start
    # RayClient can leave orphan workers after the driver has exited.
    with (root / f"{case}-cleanup.log").open("w") as stream:
        subprocess.run(  # noqa: S603 - the current environment's Ray CLI
            [str(Path(os.sys.executable).parent / "ray"), "stop", "--force"],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    launcher = (root / f"{case}-launcher.log").read_text()
    match = re.search(r"Log: (.*)", launcher)
    log = Path(match.group(1)) if match else None
    log_text = log.read_text(errors="replace") if log else ""
    # A memory threshold is only an approximate active window, not a compute-only interval.
    active = [s for s in samples if sum(v["memory_mib"] for v in s["gpus"].values()) > ACTIVE_MEMORY_MIB]
    summary = {}
    for gpu in (4, 5, 6, 7):
        vals = [s["gpus"][gpu] for s in active]
        summary[gpu] = {
            "mean_util": sum(v["util"] for v in vals) / max(1, len(vals)),
            "busy_fraction": sum(v["util"] > 0 for v in vals) / max(1, len(vals)),
            "peak_memory_mib": max((s["gpus"][gpu]["memory_mib"] for s in samples), default=0),
        }
    errors = []
    stats = {
        "json_files": 0,
        "video_stats_files": 0,
        "passed": 0,
        "aesthetic_filtered": 0,
        "flow_filtered": 0,
        "ocr_filtered": 0,
    }
    for path in output.rglob("*.json"):
        data = json.loads(path.read_text())
        stats["json_files"] += 1
        if "num_clips_passed" in data:
            stats["video_stats_files"] += 1
            for key, field in [
                ("passed", "num_clips_passed"),
                ("aesthetic_filtered", "num_clips_filtered_by_aesthetic"),
                ("flow_filtered", "num_clips_filtered_by_optical_flow"),
                ("ocr_filtered", "num_clips_filtered_by_ocr"),
            ]:
                stats[key] += data.get(field, 0)
        if re.search(r"out of memory|OutOfMemory|timeout", json.dumps(data), re.IGNORECASE):
            errors.append(str(path))
    result = {
        "case": case,
        "elapsed_seconds": elapsed,
        "exit_code": code,
        "stop_reason": reason,
        "output": str(output),
        "log": str(log),
        "gpu": summary,
        "outputs": stats,
        "error_metadata": errors,
        "traceback": "Traceback" in log_text,
    }
    (root / f"{case}-gpu.json").write_text(json.dumps(samples))
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--workers", type=int, choices=[2, 8], default=2)
    parser.add_argument("--storage", choices=["shared", "local"], default="shared")
    parser.add_argument("--label")
    parser.add_argument("--reuse-detector", action="store_true")
    parser.add_argument("--fast-metadata", action="store_true")
    parser.add_argument("--rebalance-cpu", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root.resolve(), args.config, args.count)
    else:
        run(
            args.root.resolve(),
            args.workers,
            args.storage,
            args.label,
            args.reuse_detector,
            args.fast_metadata,
            args.rebalance_cpu,
            args.config,
        )
