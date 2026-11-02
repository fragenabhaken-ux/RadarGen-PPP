"""Production batch selection requires completed DDP updates and memory headroom."""
from scripts.run_dit_joint_lr_range import probe_summary


def rows(reserved=72, free=20):
    return [dict(phase='ready', gpu_total_gib=95, gpu_free_gib=1,
                 gpu_peak_allocated_gib=94,gpu_peak_reserved_gib=94)] + [
        dict(phase='train',gpu_total_gib=95,gpu_free_gib=free,
             gpu_peak_allocated_gib=70,gpu_peak_reserved_gib=reserved) for _ in range(5)]


def test_successful_probe_requires_reserved_and_device_memory_margin():
    result = probe_summary(rows(),8,0)
    assert result['safe'] and result['effective_batch']==32
    assert result['rank0_peak_allocated_gib']==70
    assert not probe_summary(rows(reserved=83),8,0)['safe']
    assert not probe_summary(rows(free=10),8,0)['safe']


def test_partial_or_failed_probe_cannot_start_sweep():
    assert not probe_summary(rows()[:-1],8,0)['safe']
    assert not probe_summary(rows(),8,1)['safe']
