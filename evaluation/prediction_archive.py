"""Lossless, frame-addressable validation predictions, with a single writer per run."""
import json
from pathlib import Path

import h5py
import numpy as np


class PredictionArchive:
    """One file per checkpoint; frames become durable before their metric records."""

    def __init__(self, directory, provenance, dataset, *, resume=False):
        self.path = Path(directory) / 'predictions.h5'
        self.provenance = provenance
        self.dataset = dataset
        self.resume = resume
        self.handle = None

    def __enter__(self):
        self.handle = h5py.File(self.path, 'r+' if self.resume else 'x')
        try:
            if self.resume:
                if (self.handle.attrs.get('format_version') != 1 or
                        json.loads(self.handle['provenance_json'].asstr()[()]) != self.provenance):
                    raise ValueError('Prediction archive provenance differs from this validation run')
            else:
                h = self.handle
                h.attrs.update(format_version=1, complete=False,
                               coordinate_range=self.dataset.point_limit,
                               points_columns='x,y,rcs,doppler', points_units='m,m,dBsm,m/s',
                               field_origin='upper', field_extent='[-radius,radius,-radius,radius]')
                # Dataset rather than an attribute: full inventories can exceed HDF5 attribute limits.
                h.create_dataset('provenance_json', data=json.dumps(self.provenance),
                                 dtype=h5py.string_dtype('utf-8'))
                h.create_dataset('normalization_json', data=json.dumps(self.dataset.normalization),
                                 dtype=h5py.string_dtype('utf-8'))
                selected = self.provenance.get('selected_indices', list(range(self.provenance['sample_count'])))
                records = [self.dataset.frame_index[i] for i in selected]
                index = h.create_group('index')
                for name, values in (
                    ('scene', [r[1]['scene_token'] for r in records]),
                    ('source', [str(r[0]) for r in records]),
                    ('group', [self.key(r[1]) for r in records]),
                ):
                    index.create_dataset(name, data=values, dtype=h5py.string_dtype('utf-8'))
                index.create_dataset('frame', data=[r[1]['frame_index'] for r in records])
                index.create_dataset('dataset_index', data=selected)
            self.handle.attrs['complete'] = False
            self.handle.flush()
        except BaseException:
            self.handle.close()
            raise
        return self

    @staticmethod
    def key(expected):
        scene = expected['scene_token']
        if not scene or '/' in scene or scene in ('.', '..'):
            raise ValueError(f'Invalid scene token: {scene}')
        return f"scenes/{scene}/frames/{int(expected['frame_index']):06d}"

    def verify(self, index, expected):
        group = self.handle.get(self.key(expected))
        if group is None or not group.attrs.get('complete') or group.attrs.get('index') != index:
            raise ValueError(f'Missing or incomplete archived prediction: {self.key(expected)}')
        if 'points' not in group or 'fields' not in group:
            raise ValueError('Archived prediction lacks points or dense fields')

    def write(self, index, expected, source, points, fields):
        key = self.key(expected)
        if key in self.handle:
            # An export may have been flushed just before its JSONL record was interrupted.
            del self.handle[key]
        group = self.handle.create_group(key)
        group.attrs.update(complete=False, index=index, source=str(source),
                           metadata_json=json.dumps(expected))
        arrays = {'points': points}
        names = ('point_density', 'rcs', 'doppler')
        arrays.update({f'fields/{name}': value for name, value in
                       (fields.items() if isinstance(fields, dict) else zip(names, fields))})
        for name, value in arrays.items():
            value = np.asarray(value)
            # Preserve original dtype and values, including empty point clouds.
            options = dict(compression='lzf', shuffle=True, fletcher32=True) if value.size else {}
            group.create_dataset(name, data=value, **options)
        self.handle.flush()
        group.attrs['complete'] = True
        self.handle.flush()

    def finish(self, count):
        self.handle.attrs['complete'] = count == self.provenance['sample_count']
        self.handle.attrs['evaluated_samples'] = count
        self.handle.flush()

    def __exit__(self, *exc):
        self.handle.close()
