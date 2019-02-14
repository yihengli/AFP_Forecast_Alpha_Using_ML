import logging
import os

import numpy as np
import yaml
from tqdm import tqdm, tqdm_notebook

import colorlog


def load_search_cv_config(path_config):
    """
    Load the configurations recorded in the config file.

    Numpy related functions will be translated and executed.

    Parameters
    ----------
    path_config : str
        The abs path to the `yaml` config file

    Returns
    -------
    dict
        The dict of kwargs that can be directly used by SearchCV objects.
    """
    if path_config is None:
        return path_config

    with open(path_config, 'r') as config:
        settings = yaml.load(config)
        for k, v in settings['search_spaces'].items():
            if isinstance(v, dict) and 'np' in v.keys():
                settings['search_spaces'][k] = getattr(
                    np, list(v['np'].keys())[-1])(**list(v['np'].values())[-1])
        return settings


def in_ipynb():
    """
    Check if the current environment is IPython Notebook

    Note, Spyder terminal is also using ZMQShell but cannot render Widget.

    Returns
    -------
    bool
        True if the current env is Jupyter notebook
    """
    try:
        zmq_status = str(type(get_ipython())) == "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"  # noqa E501
        spyder_status = any('SPYDER' in name for name in os.environ)
        return zmq_status and not spyder_status

    except NameError:
        return False


def get_tqdm():
    """
    Get proper tqdm progress bar instance based on if the current env is
    Jupyter notebook

    Note, Windows system doesn't supprot default progress bar effects

    Returns
    -------
    type, bool
        either tqdm or tqdm_notebook, the second arg will be ascii option
    """
    ascii = True if os.name == 'nt' else False

    if in_ipynb():
        # Jupyter notebook can always handle ascii correctly
        return tqdm_notebook, False
    else:
        return tqdm, ascii


class TqdmHandler(logging.StreamHandler):
    """
    A logging output handler that will use TQDM write method to live with
    tqdm progress bar
    """

    def __init__(self):
        colorlog.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


def get_logger(name='afp'):
    """
    Get a proper logger object for this project. The logger has been formatted
    with colors and should be able to use together with TQDM progressbars

    Parameters
    ----------
    name : str, optional
        The name of the logger (the default is 'afp')

    Returns
    -------
    logging.RootLogger
        The logger object for this project
    """

    logger = colorlog.getLogger(name)

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = TqdmHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(log_color)s %(levelname)-6s%(reset)s %(white)s%(message)s'))  # noqa E501
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
