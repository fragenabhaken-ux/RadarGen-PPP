#!/usr/bin/env python
"""Validate annotated TruckScenes validation keyframes for PPP or RadarGen."""
import argparse
import logging
from pathlib import Path
import sys
import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import yaml
    from evaluation.validation import run_validation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', help='Override the local trained checkpoint')
    parser.add_argument('--output-dir', help='Override output directory (must be empty)')
    parser.add_argument('--max-samples', type=int, help='Optional bounded smoke run; default is all annotated validation keyframes')
    parser.add_argument('--resume', action='store_true', help='Continue an existing --output-dir with identical config/checkpoint/inventory')
    parser.add_argument('--dataset-index', help='Override persistent validation dataset index path')
    parser.add_argument('--rebuild-dataset-index', action='store_true',
                        help='Repeat full metadata validation and replace the saved dataset index')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    if args.dataset_index:
        config['dataset_index_path'] = args.dataset_index
    if args.rebuild_dataset_index:
        config['rebuild_dataset_index'] = True
    if args.checkpoint:
        config['checkpoint_path'] = args.checkpoint
    if not config.get('checkpoint_path'):
        parser.error('Specify a local checkpoint with --checkpoint or checkpoint_path in the config')
    def stop_on_term(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_on_term)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run_validation(config, output_dir=args.output_dir, max_samples=args.max_samples, resume=args.resume)


if __name__ == '__main__':
    main()
