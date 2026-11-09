"""Report only complete, matched full-validation results; rank unrounded means."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from compare_full_validation import RUNS, generate


class FullComparisonChecks(unittest.TestCase):
    def populate(self, root):
        for i, (_, name) in enumerate(RUNS):
            directory = root / name
            directory.mkdir()
            summary = dict(complete=True, evaluated_samples=2, expected_samples=2,
                           metrics={'chamfer_distance_xy': dict(mean=1 + i, std=0)}, foreground={})
            (directory / 'summary.json').write_text(json.dumps(summary))
            (directory / 'validation_config.json').write_text(json.dumps(dict(
                max_samples=None, keyframes_only=True, selected_indices=[0, 1], inventory_sha256='same')))
            (directory / 'metrics.jsonl').write_text(''.join(json.dumps(dict(scene='s', frame=i))+'\n' for i in range(2)))

    def test_pending_then_complete_and_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'comparison.md'
            generate(root, output)
            self.assertIn('Pending / incomplete', output.read_text())
            self.assertNotIn('## DINO:', output.read_text())
            self.populate(root)
            generate(root, output)
            text = output.read_text()
            self.assertIn('## DINO:', text)
            self.assertIn('## DiT:', text)
            self.assertEqual(text.count('## DINO:'), 2)
            self.assertEqual(text.count('## DiT:'), 2)
            self.assertIn('| Scope | Metric | DINO four-node (2684031) | DINO final (2744861) |', text)
            self.assertIn('| Scope | Metric | DiT four-node (2684032) | DiT final (2744862) |', text)
            self.assertEqual(text.count('| Entire Area | CD Loc. ↓ | **1.000 ± 0.000** |'), 2)
            self.assertNotIn('<u>', text)
            metrics = root / RUNS[-1][1] / 'metrics.jsonl'
            metrics.write_text(metrics.read_text().replace('"frame": 1', '"frame": 9'))
            with self.assertRaisesRegex(ValueError, 'identities or inventories differ'):
                generate(root, output)
            self.assertEqual(output.read_text(), text)

    def test_reject_subsampled_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.populate(root)
            path = root / RUNS[0][1] / 'validation_config.json'
            config = json.loads(path.read_text())
            config['max_samples'] = 50
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'Not an unrestricted'):
                generate(root, root / 'comparison.md')
