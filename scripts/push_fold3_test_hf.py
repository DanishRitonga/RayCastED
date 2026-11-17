"""Push the transformed held-out test set to a HuggingFace dataset repo.

Uploads the eval-facing artifacts the Jetson needs so ``eval_lsp_deploy
--hf-dataset <repo>`` can fetch them directly:
    test/<tile>.npz          (output/v2/transformed/test, 2,722 tiles, ~444 MB)
    fold3-00000-of-00001.parquet  (dense GT, ~289 MB)

Run on the training machine once:

    HF_REPO_ID=<user>/<repo> uv run python scripts/push_fold3_test_hf.py

The HF token is picked up from `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` or an
existing `huggingface-cli login` cache.
"""

import argparse
import os
from pathlib import Path


def main():
    """Upload the test tiles and fold parquet to a HuggingFace dataset repo."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default=os.environ.get('HF_REPO_ID'), help='HF dataset repo id, e.g. <user>/<repo>')
    ap.add_argument(
        '--test-dir',
        default='output/v2/transformed/test',
        help='Directory of transformed test .npz tiles',
    )
    ap.add_argument('--parquet', default='raycasted/data/dataset/PanNuke/data/fold3-00000-of-00001.parquet')
    ap.add_argument('--private', action='store_true', help='Create the repo as private')
    args = ap.parse_args()
    if not args.repo:
        ap.error('--repo or HF_REPO_ID is required')

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(args.repo, repo_type='dataset', private=args.private, exist_ok=True)

    test_dir = Path(args.test_dir)
    n_tiles = len(list(test_dir.glob('*.npz')))
    if n_tiles == 0:
        raise SystemExit(f'no .npz tiles in {test_dir}')
    print(f'Uploading {n_tiles} tiles from {test_dir} -> {args.repo}:test/ ...')
    api.upload_folder(
        repo_id=args.repo,
        repo_type='dataset',
        folder_path=str(test_dir),
        path_in_repo='test',
        commit_message=f'Add {n_tiles} transformed test tiles',
    )

    parquet = Path(args.parquet)
    if not parquet.is_file():
        raise SystemExit(f'parquet not found: {parquet}')
    print(f'Uploading {parquet} -> {args.repo}:{parquet.name} ...')
    api.upload_file(
        repo_id=args.repo,
        repo_type='dataset',
        path_or_fileobj=str(parquet),
        path_in_repo=parquet.name,
        commit_message='Add dense GT parquet',
    )
    print(f'Done. Fetch on the Jetson with: eval_lsp_deploy --hf-dataset {args.repo} ...')


if __name__ == '__main__':
    main()
