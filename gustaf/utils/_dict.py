"""gustaf/gustaf/utils/dict_check.py

Functions for checking dict and log at the same time.
"""

from gustaf.utils import log


def _get_property(property_dict, key, log_head):
    """
    Checks if property exist in given dict and returns.
    If not, returns None.
    Adds log.

    Parameters
    -----------
    property_dict: dict
    key: str
    log_head: str

    Returns
    --------
    property: obj or None
    """
    if key in property_dict:
        log._debug(
            log_head,
            "- returning property `" + key + "`.",
        )
        return property_dict[key]

    else:
        log._debug(
            log_head,
            "- property `" + key + "` does not exist.",
            "Retuning `None`."
        )
        return None


def _update_property(property_dict, key, value, log_head):
    """
    Updates dictionary key with given value.
    If None is given as value and given key exists, it pops the key.
    Adds log.

    Parameters
    -----------
    property_dict: dict
    key: str
    value: obj
    class_name: str

    Returns
    --------
    None
    """
    if value is None:
        if key in property_dict:
            log._debug(
                log_head,
                "- `None` is given to update property.",
                "`pop`-ing the key:",
                key
            )
            property_dict.pop(key)

        else:
            log._debug(
                log_head,
                "- `None` is given to update property",
                f"`{key}`,",
                "which doesn't exist.",
                "Doing nothing.",
            )

        return None

    if key in property_dict:
        second_phrase = "- updating existing property `" + key + "`"

    else:
        second_phrase = "- adding new property `" + key + "`"

    log._debug(
        log_head,
        second_phrase,
        f"with {type(value)} obj at {hex(id(value))}.",
    )

    property_dict[key] = value


def _get_cached(cached_dict, key, log_head):
    """
    Checks if key exist in given dict and returns.
    If not, returns None.
    Adds log.

    Parameters
    -----------
    cached_dict: dict
    key: str
    log_head: str

    Returns
    --------
    property: obj or None
    """
    if key in cached_dict:
        log._debug(
            log_head,
            "- returning cached `" + key + "`.",
        )
        return cached_dict[key]

    else:
        log._debug(
            log_head,
            "- `" + key + "` is not one of cached properties.",
            "Retuning `None`."
        )
        return None


def _update_cached(cached_dict, key, value, log_head):
    """
    Updates dictionary key with given value.
    If None is given as value and given key exists, it pops the key.
    Adds log.

    Parameters
    -----------
    cached_dict: dict
    key: str
    value: obj
    class_name: str

    Returns
    --------
    None
    """
    if value is None:
        if key in cached_dict:
            log._debug(
                log_head,
                "- `None` is given to update cached properties.",
                "`pop`-ing the key:",
                key
            )
            property_dict.pop(key)

        else:
            log._debug(
                log_head,
                "- `None` is given to update cached properties",
                f"`{key}`,",
                "which doesn't exist.",
                "Doing nothing.",
            )

        return None

    if key in cached_dict:
        second_phrase = "- updating existing cached property `" + key + "`"

    else:
        second_phrase = "- adding `" + key + "` to cached properties"

    log._debug(
        log_head,
        second_phrase,
        f"with {type(value)} obj at {hex(id(value))}.",
    )

    cached_dict[key] = value
