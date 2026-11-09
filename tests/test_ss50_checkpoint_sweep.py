"""SS50 checkpoint ranking must use F1 and reject mismatched benchmark records."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_ss50_checkpoint_sweep import generate


class SweepChecks(unittest.TestCase):
    def prepare(self, root):
        entries = []
        for name, kind, score in [('previous', 'baseline', .3), ('step1000', 'finetuned', .29), ('step2000', 'finetuned', .32)]:
            checkpoint = root / (name + '.pth')
            checkpoint.write_bytes(b'checkpoint')
            stat = checkpoint.stat()
            directory = root / name
            directory.mkdir()
            entries.append(dict(label=name, kind=kind, checkpoint=str(checkpoint), results=str(directory), epoch=1, step=1000))
            config = dict(checkpoint_path=str(checkpoint), checkpoint_size=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns,
                          max_samples=50, validation_version=6, inventory_sha256='same', seed=42, selection_seed=42)
            (directory / 'validation_config.json').write_text(json.dumps(config))
            metrics = {k: dict(mean=v, std=0.) for k, v in [('distance_attr_f1', score), ('distance_attr_precision', score),
                       ('distance_attr_recall', score), ('chamfer_distance_xy', 1.)]}
            summary = dict(complete=True, evaluated_samples=50, expected_samples=50, metrics=metrics, foreground=dict(hit_rate=.7))
            (directory / 'summary.json').write_text(json.dumps(summary))
            (directory / 'metrics.jsonl').write_text(''.join(json.dumps(dict(scene='s',frame=i))+'\n' for i in range(50)))
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps(dict(entries=entries)))
        return manifest, entries

    def test_winner_provisional_and_mismatch(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, entries = self.prepare(root)
            output = root / 'report.md'
            generate(manifest, output)
            text = output.read_text()
            self.assertIn('**Best overall on SS50:** step2000 (F1 = 0.320000)', text)
            self.assertIn('F1 difference +0.020000', text)
            summary_path = Path(entries[2]['results']) / 'summary.json'
            summary = json.loads(summary_path.read_text())
            summary['complete'] = False
            summary_path.write_text(json.dumps(summary))
            generate(manifest, output)
            self.assertIn('Rankings below are provisional', output.read_text())
            self.assertNotIn('**Best overall on SS50:**', output.read_text())
            rows_path = Path(entries[1]['results']) / 'metrics.jsonl'
            rows_path.write_text(rows_path.read_text().replace('"frame": 0', '"frame": 999'))
            with self.assertRaisesRegex(ValueError, 'frame identities, inventory, or seeds differ'):
                generate(manifest, output)
