"""Optional TensorBoard events; existing reports remain the durable output."""
from contextlib import contextmanager
import math
from numbers import Real
from pathlib import Path


@contextmanager
def tensorboard_writer(directory, enabled=False, *, purge_step=None):
    if type(enabled) is not bool:
        raise ValueError("tensorboard must be true or false")
    writer = None
    if enabled:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(Path(directory) / "tensorboard"), purge_step=purge_step)
    try:
        yield writer
    finally:
        if writer is not None:
            writer.close()


def log_scalars(writer, prefix, values, step):
    """Log finite numeric leaves of nested metric dictionaries."""
    if writer is None:
        return
    for key, value in values.items():
        tag = f"{prefix}/{key}"
        if isinstance(value, dict):
            log_scalars(writer, tag, value, step)
        elif isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
            writer.add_scalar(tag, value, step)
