import logging


def get_logger(level=logging.INFO,
               filename=None,
               formatting="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"):
    """
    Returns a basic logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatting)

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
