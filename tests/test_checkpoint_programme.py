import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from scripts.run_ablation_checkpoint_programme import resolve, validation, detector, config_for


def entry_at(directory):
    root = Path(directory)
    (root / 'train/checkpoints').mkdir(parents=True)
    template = root / 'template.yaml'
    template.write_text('seed: 42\nreference_validation_records: old.jsonl\n')
    return dict(train_root=str(root / 'train'), bundle_root=str(root / 'bundle'),
                checkpoint=None, minimum_step=1000, step=None,
                model_id_template='resnet_step{step}', template_config=str(template),
                ss50_output=str(root / 'ss50'), detector_root=str(root / 'detector'))


def test_resolve_final_after_training_dependency():
    with TemporaryDirectory() as directory:
        entry = entry_at(directory)
        for step in (1000, 2000, 2599):
            (Path(entry['train_root']) / f'checkpoints/epoch_1_step_{step}.pth').touch()
        result = resolve(entry)
        assert result['step'] == 2599
        assert result['model_id'] == 'resnet_step2599'
        assert json.loads((Path(entry['bundle_root']) / 'resolved_checkpoint.json').read_text()) == result


def test_resolve_rejects_uncontinued_model():
    with TemporaryDirectory() as directory:
        entry = entry_at(directory)
        (Path(entry['train_root']) / 'checkpoints/epoch_1_step_1000.pth').touch()
        with pytest.raises(ValueError):
            resolve(entry)


def test_full_and_ss50_protocols_differ_only_in_selection_and_output():
    with TemporaryDirectory() as directory:
        entry = entry_at(directory)
        entry['checkpoint'] = '/checkpoint.pth'
        full, small = config_for(entry), config_for(entry, ss50=True)
        assert full['max_samples'] is None and small['max_samples'] == 50
        assert 'reference_validation_records' not in full
        assert small['reference_validation_records'].endswith('/metrics.jsonl')
        assert full['save_plots'] and full['save_predictions'] and not full['tensorboard']


def test_completed_validation_is_reused_and_wrong_checkpoint_rejected():
    with TemporaryDirectory() as directory:
        entry = entry_at(directory)
        entry['checkpoint'] = str(Path(directory) / 'checkpoint.pth')
        output = Path(entry['bundle_root']) / 'metrics'
        output.mkdir(parents=True)
        summary = dict(complete=True, evaluated_samples=1906, checkpoint=entry['checkpoint'])
        path = output / 'summary.json'
        path.write_text(json.dumps(summary))
        with patch('scripts.run_ablation_checkpoint_programme.subprocess.run') as run:
            validation(entry)
            run.assert_not_called()
        summary['checkpoint'] = '/wrong.pth'
        path.write_text(json.dumps(summary))
        with pytest.raises(ValueError):
            validation(entry)


def test_complete_detector_result_is_reused():
    with TemporaryDirectory() as directory:
        entry = entry_at(directory)
        path = Path(directory) / 'cached.json'
        path.write_text(json.dumps(dict(full_archive=True, sample_count=1906)))
        entry['cached_detector_metrics'] = str(path)
        with patch('scripts.run_ablation_checkpoint_programme.subprocess.run') as run:
            assert detector(entry) == str(path)
            run.assert_not_called()
