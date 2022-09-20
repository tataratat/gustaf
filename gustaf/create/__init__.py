from gustaf.create import vertices

try:
    from gustaf.create import spline
except BaseException:
    pass

__all__ = [
        "vertices",
        "spline",
]
