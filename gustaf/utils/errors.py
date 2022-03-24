"""gustaf/gustaf/utils/errors.py

States all custom errors for gustaf.
"""

class InvalidSetterCallError(Exception):
    """
    Raised when setter is called, but it is inappropriate to call the setter.
    Probably because there are enough properties set, that setting more
    properties will result in undefined behavior.
    """
    def __init__(self, current_obj, message=""):
        self.message = (
            "Unalbe to set more properties, since it can cause inconsistency "
            + f"in other properties of current obj: {type(current_obj)}"
            + ", which already has following properties: "
            + f"{str(current_obj._properties.keys())[10:-1]}"
            + message
        )
        super().__init__(self.message)

    def __str__(self):
        return self.message
