import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from scripts.run_ablation_full_validation_scenes import bind_worker_gpu, run_worker


@pytest.mark.parametrize('rank', range(4))
def test_explicit_gpu_binding_when_slurm_exposes_all_gpus(rank):
    with patch.dict('os.environ', {'SLURM_STEP_GPUS': '0,1,2,3',
                                  'CUDA_VISIBLE_DEVICES': '0,1,2,3'}, clear=True):
        bind_worker_gpu(rank)
        import os
        assert os.environ['CUDA_VISIBLE_DEVICES'] == str(rank)


def make_spec(directory):
    root = Path(directory)
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'body': {'scenes': [
        {'scene_token': f'scene{i}'} for i in range(7)]}}))
    return dict(work_dir=str(root / 'execution'), manifest=str(manifest),
                dataset_config='dataset.yaml', scene_output_root=str(root / 'scenes'),
                model_id='test_model', label='test', validation_config='validation.yaml',
                validation_output=str(root / 'metrics'), keyframes=2)


@pytest.mark.parametrize('reverse', [False, True])
def test_scene_workers_partition_without_overlap(reverse):
    with TemporaryDirectory() as directory:
        spec = make_spec(directory)
        spec['reverse_scene_order'] = reverse
        selected = []
        with patch('scripts.run_ablation_full_validation_scenes.subprocess.run') as run:
            for rank in (1, 2, 3):
                run_worker(spec, rank)
                status = json.loads((Path(spec['work_dir']) / f'worker_{rank}_status.json').read_text())
                assert status['completed']
                assert status['scenes'] == status['assigned_scenes']
                ordered = [f'scene{i}' for i in range(7)]
                if reverse:
                    ordered.reverse()
                assert status['scenes'] == ordered[rank - 1::3]
                selected.extend(status['scenes'])
            assert run.call_count == 7
        assert sorted(selected) == [f'scene{i}' for i in range(7)]


def test_validation_resumes_and_checks_pngs_before_comparison():
    with TemporaryDirectory() as directory:
        spec = make_spec(directory)
        output = Path(spec['validation_output'])
        (output / 'plots' / 'scene').mkdir(parents=True)
        (output / 'validation_config.json').write_text('{}')
        (output / 'summary.json').write_text(json.dumps({'complete': True, 'evaluated_samples': 2}))
        for index in range(2):
            (output / 'plots' / 'scene' / f'{index}.png').touch()
        with patch('scripts.run_ablation_full_validation_scenes.subprocess.run') as run, \
             patch('scripts.run_ablation_full_validation_scenes.write_comparison') as compare:
            run_worker(spec, 0)
            assert '--resume' in run.call_args.args[0]
            compare.assert_called_once_with(spec)


def test_worker_failure_is_recorded():
    with TemporaryDirectory() as directory:
        spec = make_spec(directory)
        with patch('scripts.run_ablation_full_validation_scenes.subprocess.run', side_effect=RuntimeError('failed')):
            with pytest.raises(RuntimeError):
                run_worker(spec, 1)
        status = json.loads((Path(spec['work_dir']) / 'worker_1_status.json').read_text())
        assert not status['completed']
        assert 'failed' in status['error']
