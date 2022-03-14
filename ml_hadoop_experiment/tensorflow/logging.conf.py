import logging


def with_file_handler(filename: str) -> None:
    _logger = logging.getLogger()
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")
    fh.setFormatter(formatter)
    _logger.addHandler(fh)
