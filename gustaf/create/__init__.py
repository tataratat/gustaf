from gustaf.create import vertices, faces, volumes

try:
    from gustaf.create import spline
except ImportError as err:
    # overwrites  all modules which depend on the `splinepy` library
    # with an object which will throw an error as soon
    # as it is used the first time. This means that any non spline based
    # functionality works as before, but as soon as these are used a
    # comprehensive exception will be raised which is understandable in
    # contrast to the possible multitude of errors previously possible
    from gustaf.helpers.raise_if import ModuleImportRaiser
    spline = ModuleImportRaiser("splinepy", err)

__all__ = [
        "vertices",
        "faces",
        "volumes",
        "spline",
]
