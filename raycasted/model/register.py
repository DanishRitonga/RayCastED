"""RayCastED — Ultralytics Registration Utility.

Registers PolygonDetect and RayRefinementBlock with the Ultralytics
module resolution system so YAML model configs can reference them.

Usage:
    from raycasted.model import register_polygon_head
    register_polygon_head()  # call before loading any YAML config

Note:
    Full parse_model() frozenset integration requires monkey-patching.
    For Phase 4, only namespace patching is done — direct construction
    is used for testing. Full YAML integration will be added in Phase 7.
"""

_REGISTERED = False


def register_polygon_head() -> None:
    """Patch PolygonDetect into ultralytics module namespaces.

    Makes PolygonDetect resolvable by parse_model's globals() lookup.
    Must be called before loading any YAML config that references PolygonDetect.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks

    from raycasted.model.head import PolygonDetect, RayRefinementBlock

    # Inject into ultralytics.nn.modules namespace
    setattr(modules, 'PolygonDetect', PolygonDetect)
    setattr(modules, 'RayRefinementBlock', RayRefinementBlock)

    # Inject into ultralytics.nn.tasks namespace (for globals() resolution in parse_model)
    setattr(tasks, 'PolygonDetect', PolygonDetect)
    setattr(tasks, 'RayRefinementBlock', RayRefinementBlock)

    _REGISTERED = True
