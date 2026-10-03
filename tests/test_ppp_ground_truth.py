"""Small CPU tests for continuous HDF5 evaluation targets."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import h5py
import numpy as np
from evaluation.ppp_ground_truth import read_continuous_targets


class GroundTruthTests(unittest.TestCase):
    def test_preserves_coordinates_marks_and_multiplicity(self):
        with h5py.File('test', 'w', driver='core', backing_store=False) as h:
            g = h.create_group('ppp_targets')
            # Identical locations must remain two distinct detections.
            expected = np.array([[.123, -.456, -7, 2], [.123, -.456, 8, -3]], dtype=np.float32)
            g['xy'], g['rcs'], g['doppler'] = expected[:, :2], expected[:, 2], expected[:, 3]
            np.testing.assert_array_equal(read_continuous_targets(h, 50), expected)
            g['xy'][0, 0] = 50
            with self.assertRaisesRegex(ValueError, 'range'):
                read_continuous_targets(h, 50)
            g['xy'][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                read_continuous_targets(h, 50)


if __name__ == '__main__':
    unittest.main()
