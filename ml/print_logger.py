import logging


def print(*log_line_vals):
    """
        Prints and logs a string with an arbitrary set of values
    """
    log_line = ''
    for arg in log_line_vals:
        log_line = log_line + str(arg) + ' '

    sys.stdout.write(log_line.strip() + '\n')
    logging.info(log_line.strip())


def initialise_print_logger(file_location):
    """ Setup logging config to allow all print statements to be logged
    """
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d/%m/%Y %I:%M:%S %p',
        level=logging.DEBUG,
        filename=file_location
    )
