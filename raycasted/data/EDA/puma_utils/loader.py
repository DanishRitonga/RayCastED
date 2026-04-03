import json

import geopandas as gpd
from shapely.geometry import shape


def load_puma_geojson(file_path):
    with open(file_path) as f:
        raw_data = json.load(f)

    extracted_features = []

    # Loop through every annotation in the file
    for feature in raw_data["features"]:
        geom = shape(feature["geometry"])

        props = feature.get("properties", {})
        classification = props.get("classification", {})
        class_name = classification.get("name", "unlabeled")
        color = classification.get("color", {})

        extracted_features.append(
            {
                "class_name": class_name,
                "geometry": geom,
                "color": color,
            }
        )

    gdf = gpd.GeoDataFrame(extracted_features, geometry="geometry")

    return gdf
