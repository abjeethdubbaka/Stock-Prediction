import logging
import logging.handlers
import os
import json


def setup_logging(config_path='config/config.json'):
    """ Configure logging system"""
    with open(config_path) as f:
        config = json.load(f)['logging']

    # create log directory
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    # Main logger configuration
    logger = logging.getLogger('StockPrediction')
    logger.setLevel(config['level'])

    # Formatter
    formatter = logging.Formatter(config['format'], datefmt=config['date_format'])

    # File Handler (rotation)
    file_handler = logging.handlers.RotatingFileHandler(filename=os.path.join(config['log_dir'], config['main_log']),
                                                        maxBytes=10 * 1024 * 1024, backupCount=5)  # 10 MB
    file_handler.setFormatter(formatter)

    # Error Handling
    error_handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(config['log_dir'], config['error_log']),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger

