import logging


def configure_logger(name=__name__, log_file='app.log', level=logging.INFO) -> logging.Logger:
    '''
    Configures the logger with console and file handlers.

    Args:
        name (str): The name of the logger.
        log_file (str): The file to log messages to.
        level (int): The logging level.
    
    Return:
        The logger object.
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False # do not pass logs to the default logger

    # Check if the logger already has handlers to prevent adding multiple handlers
    if not logger.handlers:
        # Create the formatter object for the logger
        file_formatter = logging.Formatter('%(asctime)s \t %(filename)s \t %(levelname)s \t %(message)s')
        stdout_formatter = logging.Formatter('%(levelname)s \t %(message)s')

        # Create the console handler and setting its level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Create the file handler and setting its level
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        # Add the formatter to the handlers
        console_handler.setFormatter(stdout_formatter)
        file_handler.setFormatter(file_formatter)

        # Add the handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger
