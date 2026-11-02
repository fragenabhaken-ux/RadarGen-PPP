#!/usr/bin/env python
"""Export a scene once, then render arbitrary saved model/mark combinations."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    predict = commands.add_parser('predict', help='Run inference and save reusable scene HDF5 files')
    predict.add_argument('--config', required=True)
    predict.add_argument('--models', nargs='+', help='Model IDs from config; default: all configured models')
    render = commands.add_parser('render', help='Render saved exports without inference')
    render.add_argument('--scene-dir', required=True)
    render.add_argument('--models', nargs='+', required=True, help='Saved model IDs, in column order after GT')
    render.add_argument('--output-dir', required=True)
    render.add_argument('--marks', nargs='+', choices=['doppler', 'rcs', 'heatmap'], default=['doppler'])
    render.add_argument('--formats', nargs='+', choices=['tar', 'png', 'mp4'], default=['tar'])
    render.add_argument('--fps', type=float)
    render.add_argument('--dpi', type=int, default=110)
    render.add_argument('--gamma', type=float, default=.2, help='Heatmap display exponent only')
    render.add_argument('--doppler-limit', type=float, default=30.)
    render.add_argument('--rcs-limits', nargs=2, type=float, default=[-20., 66.])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    from evaluation.scene_comparison import predict_scene, render_scene
    if args.command == 'predict':
        predict_scene(args.config, args.models)
    else:
        if args.rcs_limits[0] >= args.rcs_limits[1]:
            parser.error('--rcs-limits must be increasing')
        render_scene(args.scene_dir, args.models, args.output_dir, marks=args.marks,
                     formats=args.formats, fps=args.fps, dpi=args.dpi, gamma=args.gamma,
                     doppler_limit=args.doppler_limit, rcs_limits=args.rcs_limits)


if __name__ == '__main__':
    main()
