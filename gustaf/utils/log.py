"""gustaf/gustaf/utils/log.py.

Thin logging wrapper.
"""

import logging as _logging
from functools import partial as _partial


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
    logger = _logging.getLogger("gustaf")

    # level
    level = _logging.DEBUG if debug else _logging.INFO
    logger.setLevel(level)

    # format
    formatter = _logging.Formatter(fmt="%(name)s [%(levelname)s] %(message)s")

    # apply format using stream handler
    # let's use only one stream handler so that calling configure multiple
    # times won't duplicate printing.
    new_handlers = []
    for _i, h in enumerate(logger.handlers):
        # we skip all the stream handler.
        if isinstance(h, _logging.StreamHandler):
            continue

        # blindly keep other ones
        else:
            new_handlers.append(h)

    # add new stream handler
    stream_handler = _logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    new_handlers.append(stream_handler)

    logger.handlers = new_handlers

    # output logs
    if logfile is not None:
        file_logger_handler = _logging.FileHandler(logfile)
        logger.addHandler(file_logger_handler)


def debug(*log: str) -> None:
    """Debug logger.

    Parameters
    -----------
    *log: Tuple[str]

    Returns
    --------
    None
    """
    logger = _logging.getLogger("gustaf")
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
    logger = _logging.getLogger("gustaf")
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
    logger = _logging.getLogger("gustaf")
    logger.warning(" ".join(map(str, log)))


def prepended_log(message, log_func):
    """
    Prepend message before a logging function.

    Parameters
    ----------
    message: str
    log_func: function
      one of the following - {info, debug, warning}

    Returns
    -------
    prepended: function
    """
    return _partial(log_func, message)
