# %%
from pathlib import Path

import pandas as pd
from utils import _decode_image_bytes, _get_yolo_bbox, _load_parquet_as_df

script_dir = Path(__file__).parent
data_dir = script_dir.parent / 'data' / 'PanNuke' / 'data'
config_path = script_dir / 'pannuke_utils' / 'config.json'


def _get_gt_df(df: pd.DataFrame) -> pd.DataFrame:
    gt_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        gts = []
        for ins, cat in zip(row['instances'], row['categories']):
            mask = _decode_image_bytes(ins['bytes'])
            bbox = _get_yolo_bbox(mask)
            if bbox is not None:
                gt = (cat, *bbox)
                gts.append(gt)
        gt_list.append(gts)

    df['yolo_gt'] = gt_list

    return df


# %%
dfs = _load_parquet_as_df(data_dir)
df = _get_gt_df(dfs)

# %%
