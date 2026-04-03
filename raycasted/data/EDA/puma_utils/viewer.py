import matplotlib.pyplot as plt
from tiatoolbox.annotation.storage import SQLiteStore
from tiatoolbox.wsicore.wsireader import WSIReader


def view_puma_roi(geojson_path, image_path):
    store = SQLiteStore.from_geojson(geojson_path)

    reader = WSIReader.open(image_path)
    width, height = reader.slide_dimensions(resolution=0, units="level")
    print(f"Image dimensions: {width} x {height}")
    image_array = reader.read_bounds([0, 0, width, height], resolution=0, units="level")

    fig, ax = plt.subplots(figsize=(12, 12))  # Widened slightly to fit the legend
    ax.imshow(image_array)

    color_map = {
        "nuclei_tumor": "red",
        "nuclei_lymphocyte": "lime",
        "nuclei_apoptosis": "blue",
        "nuclei_stroma": "cyan",
        "nuclei_endothelium": "orange",
    }

    for ann in store.values():
        geom = ann.geometry
        props = ann.properties
        class_name = props.get("classification", {}).get("name", "unknown")
        color = color_map.get(class_name, "yellow")

        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            ax.plot(x, y, color=color, linewidth=1.5)
        elif geom.geom_type == "Point":
            ax.plot(geom.x, geom.y, marker="o", color=color, markersize=3)

    plt.axis("off")
    plt.tight_layout()  # Ensures the legend isn't cut off when saving or viewing
    plt.show()
