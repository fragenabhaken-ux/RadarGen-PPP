"""Memory summary uses training steps after optimizer warmup, never diagnostics."""
import pytest
from scripts.summarize_ppp_memory import summarize


def row(update, start, peak, reserved=20, phase='train', batch=1):
    return dict(phase=phase, update=update, microbatch_size=batch, accumulation=4,
                gpu_total_gib=100, gpu_free_gib=78, gpu_reserved_gib=reserved,
                gpu_peak_reserved_gib=reserved, gpu_peak_allocated_gib=peak,
                gpu_step_start_allocated_gib=start)


def test_steady_memory_extrapolation_and_diagnostic_exclusion():
    rows = [row(0, 0, 99, phase='ready'), row(1, 8, 18), row(2, 10, 19),
            row(6, 10, 20), row(7, 10, 22, reserved=24)]
    result = summarize(rows)
    assert result['optimizer_updates'] == 4 and result['steady_updates'] == 2
    assert result['step_start_allocated_gib'] == 10
    assert result['step_extra_allocated_gib'] == 12
    assert result['peak_allocated_gib'] == 22
    assert result['peak_reserved_gib'] == 24
    assert result['extrapolated_allocated_gib']['4'] == 58
    assert result['provisional_microbatch_candidate'] == 6


def test_insufficient_or_non_unit_batch_cannot_claim_extrapolation():
    with pytest.raises(ValueError, match='two optimizer'):
        summarize([row(1, 8, 18)])
    with pytest.raises(ValueError, match='microbatch one'):
        summarize([row(2, 8, 18, batch=2)])
