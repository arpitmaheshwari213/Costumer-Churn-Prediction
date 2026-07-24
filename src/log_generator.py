import os
import logging


def make_logger(name: str, log_file: str = None) -> logging.Logger:
    # logger = logging.getLogger(f"{__name__}.{name}")
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = logging.INFO

        # Add File Handler if configured (for local debugging/MLflow artifacts)
        if log_file:
            file_handler = logging.FileHandler(log_file, mode="a")
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        logger.setLevel(level)

    return logger
