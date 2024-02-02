import logging


def get_logger(file_name):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler in write mode (w)
    handler = logging.FileHandler(file_name, 'w')
    handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter('%(message)s')

    # Add the formatter to the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def logging_args(args, logging):
    args_dict = vars(args)
    logging.info("[argument setting]")
    for key in args_dict:
        logging.info(f"{key}: {args_dict[key]}")
    logging.info("")
