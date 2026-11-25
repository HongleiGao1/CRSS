import logging

def make_logger(path):
    logger = logging.getLogger(str(path)); logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in [logging.FileHandler(path, encoding='utf-8'), logging.StreamHandler()]:
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger
