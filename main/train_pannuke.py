r"""PanNuke training script — fold1=train, fold2=val, fold3=test.

Thin wrapper around the RayCastED pipeline CLI with PanNuke-specific config.
Uses pannuke.yaml which has split_map: fold1→train, fold2→val, fold3→test.

Usage:
    # Full pipeline (ingest → transform → train)
    uv run python main/train_pannuke.py --output output/pannuke --epochs 100

    # Individual stages
    uv run python main/train_pannuke.py --output output/pannuke --stage ingest
    uv run python main/train_pannuke.py --output output/pannuke --stage transform
    uv run python main/train_pannuke.py --output output/pannuke --stage train --epochs 50

    # Override training params
    uv run python main/train_pannuke.py --output output/pannuke --batch 8 --imgsz 640 --device 0
"""

import sys
from pathlib import Path

CONFIG_PATH = str(Path(__file__).parent / 'pannuke.yaml')


def main():
    """Inject PanNuke config defaults and delegate to RayCastPipeline CLI."""
    from raycasted.pipeline import main as pipeline_main

    # Inject --config pannuke.yaml before user args so user can still override
    user_args = sys.argv[1:]
    sys.argv = [sys.argv[0], '--config', CONFIG_PATH] + user_args

    pipeline_main()


if __name__ == '__main__':
    main()
