import orjson


class GeoJSONHandler:
    """Stateless toolkit for GeoJSON file I/O."""

    @staticmethod
    def load_json(path: str) -> dict:
        """Load and parse a GeoJSON file using orjson."""
        with open(path, 'rb') as f:
            return orjson.loads(f.read())
