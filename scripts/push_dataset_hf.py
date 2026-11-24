"""Push the transformed PanNuke tiles (optionally the raw parquets) to HuggingFace.

Uploads the ETL-facing artifacts so another machine (e.g. the Jetson Orin Nano)
can evaluate without re-running the pipeline:

    train/ val/ test/<tile>.npz   (output/v2/transformed, 7,901 tiles, ~1.3 GB)
    data/fold{1,2,3}-00000-of-00001.parquet  (only with --with-parquet)

Run on the training machine once:

    HF_REPO_ID=<user>/<repo> uv run python scripts/push_dataset_hf.py

Then evaluate remotely with ``eval_lsp_deploy --hf-dataset <repo>`` (the test/
split is what the evaluator downloads) or keep tiles locally via
``--data-dir <repo>/test``.

The HF token is picked up from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` or an
existing `huggingface-cli login` cache.
"""

import argparse
import os
from pathlib import Path


def main():
    """Upload the transformed splits (and optionally parquets) to a HF dataset repo."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default=os.environ.get('HF_REPO_ID'), help='HF dataset repo id, e.g. <user>/<repo>')
    ap.add_argument(
        '--transformed-dir',
        default='output/v2/transformed',
        help='Root holding the transformed split directories (default: output/v2/transformed)',
    )
    ap.add_argument('--splits', default='train,val,test', help='Comma-separated splits to upload (default: all)')
    ap.add_argument(
        '--with-parquet',
        action='store_true',
        help='Also upload the raw fold parquets to data/ (needed for local ETL generation)',
    )
    ap.add_argument(
        '--parquet-dir',
        default='raycasted/data/dataset/PanNuke/data',
        help='Directory holding fold*-00000-of-00001.parquet (used with --with-parquet)',
    )
    ap.add_argument('--private', action='store_true', help='Create the repo as private')
    args = ap.parse_args()
    if not args.repo:
        ap.error('--repo or HF_REPO_ID is required')

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(args.repo, repo_type='dataset', private=args.private, exist_ok=True)

    root = Path(args.transformed_dir)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    if not splits:
        ap.error('--splits is empty')

    for split in splits:
        split_dir = root / split
        n_tiles = len(list(split_dir.glob('*.npz')))
        if n_tiles == 0:
            raise SystemExit(f'no .npz tiles in {split_dir}')
        print(f'Uploading {n_tiles} tiles {split_dir} -> {args.repo}:{split}/ ...', flush=True)
        api.upload_folder(
            repo_id=args.repo,
            repo_type='dataset',
            folder_path=str(split_dir),
            path_in_repo=split,
            commit_message=f'Add {n_tiles} transformed {split} tiles',
        )

    if args.with_parquet:
        for parquet in sorted(Path(args.parquet_dir).glob('fold*-00000-of-00001.parquet')):
            print(f'Uploading {parquet} -> {args.repo}:data/{parquet.name} ...', flush=True)
            api.upload_file(
                repo_id=args.repo,
                repo_type='dataset',
                path_or_fileobj=str(parquet),
                path_in_repo=f'data/{parquet.name}',
                commit_message=f'Add dense GT parquet {parquet.name}',
            )

    print(f'Done. Evaluate with: eval_lsp_deploy --hf-dataset {args.repo} ...')


if __name__ == '__main__':
    main()
