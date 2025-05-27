"""gustaf/gustaf/utils/log.py.

Thin logging wrapper.
"""

import logging
from functools import partial, partialmethod


def configure(debug=False, logfile=None):
    """Logging configurator.

    Parameters
    -----------
    debug: bool
    logfile: str

    Returns
    --------
    None
    """
    # logger
    logger = logging.getLogger("gustaf")

    # level
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # format
    formatter = logging.Formatter(fmt="%(name)s [%(levelname)s] %(message)s")

    # apply format using stream handler
    # let's use only one stream handler so that calling configure multiple
    # times won't duplicate printing.
    new_handlers = []
    for _i, h in enumerate(logger.handlers):
        # we skip all the stream handler.
        if isinstance(h, logging.StreamHandler):
            continue

        # blindly keep other ones
        else:
            new_handlers.append(h)

    # add new stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    new_handlers.append(stream_handler)

    logger.handlers = new_handlers

    # output logs
    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)
        logger.addHandler(file_logger_handler)


def debug(*log):
    """Debug logger.

    Parameters
    -----------
    *log: Tuple[str]

    Returns
    --------
    None
    """
    logger = logging.getLogger("gustaf")
    logger.debug(" ".join(map(str, log)))


def info(*log):
    """Info logger.

    Parameters
    -----------
    *log: Tuple[str]

    Returns
    --------
    None
    """
    logger = logging.getLogger("gustaf")
    logger.info(" ".join(map(str, log)))


def warning(*log):
    """warning logger.

    Parameters
    -----------
    *log: Tuple[str]

    Returns
    --------
    None
    """
    logger = logging.getLogger("gustaf")
    logger.warning(" ".join(map(str, log)))


def prepended_log(message, log_func, as_method=True):
    """
    Prepend message before a logging function.

    Parameters
    ----------
    message: str
    log_func: function
      one of the following - {info, debug, warning}
    as_method: bool
      If True, uses partialmethod, else partial.
      Default is True.

    Returns
    -------
    prepended: function
    """
    p = partialmethod if as_method else partial
    return p(log_func, message)
