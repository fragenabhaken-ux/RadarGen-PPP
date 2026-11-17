"""Check PPP law, coordinate/mark fidelity, and shared scene-density scaling."""
import numpy as np
from scripts.ss50_sampling_budget import sample_scaled


def test_scaled_ppp_preserves_density_ratio_and_poisson_variance():
    mass = np.array([[1., 0.], [0., 3.]])
    rcs = np.array([[11., 12.], [13., 14.]])
    doppler = -rcs
    rng = np.random.default_rng(7)
    counts = []
    for _ in range(2000):
        points = sample_scaled(mass, rcs, doppler, 2., 2., rng)
        counts.append(len(points))
        assert np.all(points[:, 2] == -points[:, 3])
        assert set(points[:, 2]).issubset({11., 14.})
        assert np.all(points[points[:, 2] == 11., :2] == [-1., 1.])
        assert np.all(points[points[:, 2] == 14., :2] == [1., -1.])
    assert abs(np.mean(counts) - 8.) < .25
    assert abs(np.var(counts) - 8.) < .8
    # A shared scale retains a twofold difference between scenes.
    counts_dense = [len(sample_scaled(2*mass, rcs, doppler, 2., 2., rng)) for _ in range(2000)]
    assert abs(np.mean(counts_dense) / np.mean(counts) - 2.) < .08
