import logging
import tensorflow as tf


def set_loggers(path_log=None, logging_level=0, b_stream=False, b_debug=False):
    """set loggers"""

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)  # level INFO

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    if path_log:  # run_paths['path_logs_train'] 'xxx\experiments\run_...\logs\run.log'
        file_handler = logging.FileHandler(path_log)
        logger.addHandler(file_handler)  # logged to the file run.log
        logger_tf.addHandler(file_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)
