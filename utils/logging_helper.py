import logging
import os
import time


import sys

def get_default_logger(name, output, use_timestamp=False, stdout_level=logging.INFO, file_level=logging.INFO):
    name = name.replace(".", "_")

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger_init_by_name(logger, name, output_path=output, use_timestamp=use_timestamp, stdout_level=stdout_level, file_level=file_level)
    return logger

def logger_init_by_name(logger, name, output_path="logs", level=logging.INFO, format='[%(asctime)s %(levelname)s]:%(message)s', stdout_level=logging.ERROR, file_level=logging.INFO, use_timestamp=False):
    timestamp_str = time.time_ns() if use_timestamp else ""
    name = name.replace(".", "_")
    logger.setLevel(level)
    formatter = logging.Formatter(format)
    os.makedirs(output_path, exist_ok=True)
    log_name = os.path.join(output_path, f"{name}_{timestamp_str}.log") if use_timestamp else os.path.join(output_path, f"{name}.log")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(stdout_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_name)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

def logger_init(logger, log_file_path, level=logging.INFO, format='[%(asctime)s %(levelname)s]:%(message)s', stdout_level=logging.INFO):
    logger.setLevel(level)
    formatter = logging.Formatter(format)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(stdout_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)


def logging_init(name, timestamp=True, desc_mode="always", level=logging.DEBUG, output='logs/', filemode='w', datefmt='%Y-%m-%d%I:%M:%S %p'):
    timestamp_str = time.time_ns() if timestamp else ""
    if desc_mode == "full":
        format = '[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]'
    elif desc_mode == "always":
        format = '[%(asctime)s-%(levelname)s:%(message)s]'
    else:
        raise RuntimeError("Not support format")
    
    log_name = f"{name}_{timestamp_str}.log"
    logging.basicConfig(filename=os.path.join(output, log_name),
                        format=format,
                        level=level, filemode=filemode, datefmt=datefmt)


def default_logger_init(name,output='logs/', filemode='w', datefmt='%I:%M:%S %p'):
    logging_init(name, datefmt=datefmt, output=output, level=logging.INFO, filemode=filemode)