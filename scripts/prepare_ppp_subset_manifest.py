#!/usr/bin/env python3
"""Bind a runtime subset to a verified full manifest without rescanning HDF5.

Only scene selection and subset_indices may differ. All inventory/completion checks remain active.
The source is never modified and destination must not already exist.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import pyrallis
    from diffusion.utils.config import SanaConfig
    from radargen.training.ppp_manifest import (
        dataset_contract, _read_json, _validate_document, _check_completions,
        _canonical, _digest)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    config = pyrallis.parse(SanaConfig, args=['--config_path', args.config_path])
    contract = dataset_contract(config)
    if not contract['subset_indices']:
        raise ValueError('An explicit runtime subset is required')
    full_contract = dict(contract, subset_indices=None, scene_tokens=None)
    document = _read_json(Path(args.source))
    body = _validate_document(document, full_contract)
    _check_completions(body)
    derived = copy.deepcopy(document)
    selected = contract['scene_tokens']
    if selected is not None:
        derived['body']['scenes'] = [r for r in body['scenes'] if r['scene_token'] in selected]
        derived['body']['samples'] = [r for r in body['samples'] if r['expected']['scene_token'] in selected]
        derived['body']['sample_count'] = len(derived['body']['samples'])
    derived['body']['contract']['scene_tokens'] = selected
    derived['body']['contract']['subset_indices'] = contract['subset_indices']
    derived['identity'] = _digest(derived['body'])
    _validate_document(derived, contract)
    _check_completions(derived["body"])
    destination = Path(config.data.ppp_manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        stream.write(_canonical(derived) + b'\n')
    print(f'Prepared {destination}: runtime indices {contract["subset_indices"]}')


if __name__ == '__main__':
    main()
