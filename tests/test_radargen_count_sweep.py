import numpy as np
import pytest
from scripts.ss50_radargen_count_sweep import threshold_for_mean, points_from_map


def test_shared_threshold_hits_mean_without_equalizing_frames():
    maps=[np.array([[.1,.2],[0.,0.]],dtype=np.float32),np.array([[.3,.4],[.5,.6]],dtype=np.float32)]
    threshold=threshold_for_mean(maps,2)
    counts=[np.count_nonzero(m>threshold) for m in maps]
    assert counts==[0,4]
    assert np.mean(counts)==2
    assert threshold_for_mean(maps,3)==0
    with pytest.raises(ValueError): threshold_for_mean(maps,4)


def test_threshold_ties_are_retained_or_excluded_together():
    maps=[np.array([[.1,.2,.2,.2,.3]],dtype=np.float32)]
    threshold=threshold_for_mean(maps,2)
    assert np.count_nonzero(maps[0]>threshold) in (1,4)
    assert abs(np.count_nonzero(maps[0]>threshold)-2)==1


def test_recovered_points_use_pixel_centers_rounded_rcs_and_strict_threshold():
    amplitude=np.array([[.1,.2],[.3,0]],dtype=np.float32)
    rcs=np.array([[1.1,2.8],[4.2,9.]])
    doppler=np.array([[.1,.2],[.3,.4]])
    points=points_from_map(amplitude,rcs,doppler,.1,2.)
    np.testing.assert_allclose(points,[[1.,1.,3.,.2],[-1.,-1.,4.,.3]])
