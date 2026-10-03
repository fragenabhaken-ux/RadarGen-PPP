"""NumPy-only tests of WP8 diagnostic links and count semantics."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
path = Path(__file__).resolve().parents[1]/'radargen/training/ppp_overfit.py'
spec = importlib.util.spec_from_file_location('ppp_overfit_diagnostics',path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DiagnosticTests(unittest.TestCase):
    def test_counts_and_physical_marks(self):
        fields = np.zeros((3,2,2)); fields[0] = np.log(8); fields[1] = 2; fields[2] = -3
        mask = np.array([[True,False],[False,True]])
        stats,mass,marks = module.summarize_fields(fields,mask,np.full((2,2),8),np.full((2,2),-27),
                                                  {'rcs':(-20,66),'doppler':(-10,10)})
        self.assertAlmostEqual(stats['expected_count'],8)
        self.assertEqual(stats['active_cell_count'],2)
        self.assertAlmostEqual(stats['count_abs_error_active'],6)
        self.assertAlmostEqual(stats['mass_fraction_at_target_cells'],.5)
        self.assertEqual(stats['rcs_mae_at_targets'],0)
        self.assertEqual(stats['doppler_mae_at_targets'],0)
        self.assertEqual(stats['doppler_outside_fraction_intensity_weighted'],1)
        np.testing.assert_allclose(mass,2)
        np.testing.assert_array_equal(marks[1],-27)

    def test_rejects_invalid_fields(self):
        fields = np.zeros((3,2,2)); fields[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            module.summarize_fields(fields,np.ones((2,2),bool),np.zeros((2,2)),np.zeros((2,2)),{})


if __name__ == '__main__':
    unittest.main()
