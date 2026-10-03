#!/usr/bin/env python3
"""Small CUDA/CPU placement probe and CPU-only result verification."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import socket


def verify_records(records):
    if len(records) != 16 or {r["rank"] for r in records} != set(range(16)):
        raise ValueError("Expected exactly 16 distinct worker ranks")
    hosts = defaultdict(list)
    for record in records:
        hosts[record["host"]].append(record)
        if record["visible_gpus"] != 1 or record["cuda_result"] != 10:
            raise ValueError(f"Rank {record['rank']}: CUDA device access failed")
        if len(record["allowed_cpus"]) != 72 or len(set(record["allowed_cpus"])) != 72:
            raise ValueError(f"Rank {record['rank']}: expected 72 allowed CPUs")
        if not record["gpu_uuid"].startswith("GPU-"):
            raise ValueError(f"Rank {record['rank']}: missing physical GPU UUID")
    if len(hosts) != 4:
        raise ValueError(f"Expected four hosts, found {len(hosts)}")
    for host, workers in hosts.items():
        if len(workers) != 4 or {r["local_rank"] for r in workers} != set(range(4)):
            raise ValueError(f"{host}: expected four distinct local worker ranks")
        if len({r["gpu_uuid"] for r in workers}) != 4:
            raise ValueError(f"{host}: workers share a physical GPU")
        used = set()
        for worker in workers:
            cpus = set(worker["allowed_cpus"])
            if used & cpus:
                raise ValueError(f"{host}: CPU affinity masks overlap")
            used.update(cpus)
    return hosts


def probe(result_dir):
    import ctypes
    import uuid
    import torch

    if int(os.environ["SLURM_NTASKS"]) != 16:
        raise RuntimeError("Expected a 16-task launch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one CUDA-visible device")
    torch.cuda.set_device(0)
    result = int((torch.arange(4, device="cuda:0") + 1).sum().item())
    torch.cuda.synchronize()

    # Query the device backing this worker's actual CUDA context. Avoid global
    # nvidia-smi indices, which need not match CUDA's visible-device ordering.
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    driver.cuCtxGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuCtxGetDevice.restype = ctypes.c_int
    if driver.cuCtxGetDevice(ctypes.byref(device)) != 0:
        raise RuntimeError("Cannot identify the active CUDA context's device")
    raw_uuid = (ctypes.c_ubyte * 16)()
    driver.cuDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    driver.cuDeviceGetUuid.restype = ctypes.c_int
    if driver.cuDeviceGetUuid(ctypes.byref(raw_uuid), device.value) != 0:
        raise RuntimeError("Cannot read the active device's physical GPU UUID")

    record = dict(rank=int(os.environ["SLURM_PROCID"]),
                  local_rank=int(os.environ["SLURM_LOCALID"]),
                  host=socket.gethostname(),
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  visible_gpus=torch.cuda.device_count(),
                  gpu_uuid="GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid))),
                  allowed_cpus=sorted(os.sched_getaffinity(0)),
                  cuda_result=result)
    print(json.dumps(record, sort_keys=True), flush=True)
    path = result_dir / f"worker_{record['rank']:02d}.json"
    # The batch script creates a fresh shared directory for each launch check.
    with path.open("x") as stream:
        json.dump(record, stream, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        records = [json.loads(p.read_text()) for p in args.result_dir.glob("worker_*.json")]
        hosts = verify_records(records)
        for host, workers in sorted(hosts.items()):
            print(f"{host}: ranks={sorted(r['rank'] for r in workers)}, "
                  "four distinct GPUs, four disjoint 72-CPU masks", flush=True)
        print("PASS: 16 CUDA workers across four hosts", flush=True)
    else:
        probe(args.result_dir)


if __name__ == "__main__":
    main()
