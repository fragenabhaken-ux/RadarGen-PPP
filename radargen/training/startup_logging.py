"""Lightweight startup diagnostics; no CUDA synchronization or collectives."""

from datetime import datetime, timezone
import os
import time

_STARTED = time.monotonic()


def startup_log(event, **fields):
    """Emit immediately on every rank, including before distributed initialization."""
    stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[PPP startup {stamp} rank={os.environ.get('RANK', '0')} "
          f"local_rank={os.environ.get('LOCAL_RANK', '0')} pid={os.getpid()} "
          f"elapsed={time.monotonic()-_STARTED:.1f}s] {event} {details}", flush=True)
