"""gustav/gustav/utils/log.py

Thin logging wrapper.
"""

import logging

def configure(debug=False, logfile=None):
    """
    Logging configurator.

    Parameters
    -----------
    debug: bool
    logfile: str

    Returns
    --------
    None
    """
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)

    else:
        logger.setLevel(logging.INFO)

    if logfile is not None:
        file_logger_handler = logging.FileHandler(logfile)

def _debug(*log):
    """
    Debug logger.

    Parameters
    -----------
    *log: *str

    Returns
    --------
    None
    """
    logs = [l for l in log]
    logging.debug(" ".join(map(str, logs)))

def _info(*log):
    """
    Info logger.

    Parameters
    -----------
    *log: *str

    Returns
    --------
    None
    """
    logs = [l for l in log]
    logging.info(" ".join(map(str, logs)))

def _warning(*log):
    """
    warning logger.

    Parameters
    -----------
    *log: *str

    Returns
    --------
    None
    """
    logs = [l for l in log]
    logging.warning(" ".join(map(str, logs)))

