"""gustaf/gustaf/helpers/raise_if.py

Collection of wrapper functions/classes that raises Error with certain behavior
"""


def invalid_inherited_attr(func, qualname, property_=False):
    """
    Returns a function that would behave the same as given function,
    but would raise AttributeError. This needs to be defined in class level.

    Parameters
    -----------
    func: function
    cls: class
    property_: bool
      is this function a property?

    Returns
    --------
    raiser: function
       behaves same as func if property_ is correctly defined
    """
    def raiser(self):
        raise AttributeError(
            f"{func.__name__} is not supported from {qualname} "
            "and its subclasses thereof."
        )

    if property_:
        return property(raiser)

    else:
        return raiser
