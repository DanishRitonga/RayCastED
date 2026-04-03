# %%
import json  # noqa: F401
from pathlib import Path

import geopandas as gpd  # noqa: F401

# local
from puma_utils import load_puma_geojson, view_puma_roi

# %%
data_dir = "../data"
PUMA_DIR = f"{data_dir}/PUMA"

PUMA_NUCLEI_DIR = "01_training_dataset_geojson_nuclei"
PUMA_TISSUE_DIR = "01_training_dataset_geojson_tissue"
PUMA_ROI_CONTEXT_DIR = "01_training_dataset_tif_context_ROIs"
PUMA_ROI_DIR = "01_training_dataset_tif_ROIs"
# %%
id = "001"
geojson_path = Path(
    f"{PUMA_DIR}/{PUMA_NUCLEI_DIR}/training_set_metastatic_roi_{id}_nuclei.geojson"
)
image_path = Path(f"{PUMA_DIR}/{PUMA_ROI_DIR}/training_set_metastatic_roi_{id}.tif")


# %%
gdf = load_puma_geojson(geojson_path)

# %%
view_puma_roi(geojson_path, image_path)

# %%
