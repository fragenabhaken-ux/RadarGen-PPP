#!/usr/bin/env python
"""Validate all preprocessed TruckScenes val pairs for PPP or RadarGen."""
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
    parser.add_argument('--max-samples', type=int, help='Optional bounded smoke run; default is the entire validation split')
    parser.add_argument('--resume', action='store_true', help='Continue an existing --output-dir with identical config/checkpoint/inventory')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
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
