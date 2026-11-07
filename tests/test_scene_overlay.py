import json

import h5py
import numpy as np
import pytest

from evaluation.scene_comparison import _export, render_scene
from evaluation.scene_overlay import render_overlays


def scene_fixture(tmp_path):
    record = dict(frame_index=0, reference_timestamp_us=1000000,
                  reference_sample_data_token='sample')
    records = [('unused', record)]
    common = dict(scene='test_scene', coordinate_range=10, fps=10, frames=[record],
                  dataset_contract={}, manifest_identity='test')
    points = np.array([[-5, 5, 12, -3], [5, -5, 20, 4]], dtype=np.float32)
    gt_meta = dict(common, model='ground_truth', label='Ground truth')
    _export(tmp_path / 'ground_truth.h5', gt_meta, records, lambda *_: (points, None))
    meta = dict(common, model='ppp', label='Test PPP', heatmap_kind='expected_points_per_cell')
    fields = dict(heatmap=np.array([[1., 2.], [3., 4.]]),
                  rcs=np.array([[12., 13.], [19., 20.]]),
                  doppler=np.array([[-3., -2.], [3., 4.]]))
    _export(tmp_path / 'predictions/test.h5', meta, records, lambda *_: (points, fields['heatmap']))
    dense = tmp_path / 'predictions/test_dense.h5'
    _export(dense, dict(meta, dense_fields_version=1), records, lambda *_: (points, fields))
    return dense, fields


def test_dense_fields_and_both_render_styles(tmp_path):
    dense, fields = scene_fixture(tmp_path)
    with h5py.File(dense) as f:
        for key, values in fields.items():
            np.testing.assert_array_equal(f['frames/000000'][key][()], values)
        assert f['frames/000000/rcs'].attrs['units'] == 'dBsm'
        assert f['frames/000000/doppler'].attrs['units'] == 'm/s'
    output = tmp_path / 'render'
    render_scene(tmp_path, ['test'], output, marks=['heatmap', 'doppler', 'rcs'],
                 formats=['png'], dpi=40)
    assert (output / 'frame_000000.png').stat().st_size > 1000
    overlay = tmp_path / 'overlays' / 'test'
    assert (overlay / 'frame_000000.png').stat().st_size > 1000
    config = json.loads((overlay / 'render_config.json').read_text())
    assert config['panels'] == ['intensity', 'doppler', 'rcs']
    assert config['ground_truth_overlay']


def test_overlay_rejects_misaligned_ground_truth(tmp_path):
    dense, _ = scene_fixture(tmp_path)
    with h5py.File(dense, 'r+') as f:
        meta = json.loads(f.attrs['provenance'])
        meta['manifest_identity'] = 'different'
        f.attrs['provenance'] = json.dumps(meta)
    with pytest.raises(ValueError, match='different frame/dataset identities'):
        render_overlays(tmp_path, ['test'], tmp_path / 'render', formats=['png'],
                        fps=10, dpi=40, gamma=.2, doppler_limit=30, rcs_limits=(-20, 66))
