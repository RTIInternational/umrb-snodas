#!/usr/bin/python
##############################################################################
#
#  United States Department of Commerce
#  NOAA (National Oceanic and Atmospheric Administration)
#  National Weather Service
#  Office of Water Prediction
#
#  Author:
#      Anders Nilsson, UCAR (created)
#
##############################################################################
""" This contains methods to set up logging and command-line parsing
"""
# Global system imports
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import logging
import logging.handlers
import socket
import sys

# Global default constants

__version__ = '1.0'

# Format for logged messages
DEFAULT_LOG_FORMAT = '%(asctime)s ' + socket.gethostname() + \
                     ' %(filename)s[%(process)d-%(threadName)s]:' + \
                     ' %(levelname)s: %(message)s'
DEFAULT_SYSLOG_FORMAT = ' %(filename)s[%(process)d-%(threadName)s]:' + \
                        ' %(levelname)s: %(message)s'

# Format for the dates used in logged messages
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S %Z'

#  Level of messages that are written to the log
DEFAULT_LEVEL = logging.INFO

############################################################################
#
#  setup_arguments
#
############################################################################
def setup_arguments(program_description,
                    epilogue,
                    add_date=False,
                    add_target=False):
    """ Configure command line parser

    This method configures the command line argument parser and help message.

    Args:
        program_description (str): A description of the program that occurs
            before the command line options.
        epilogue (str): The part of the help documentation that falls after
            the command line arguments
        add_date (bool): Whether to add a date specifying argument
        add_target (bool): Whether to add a target configuration file
                           argument

    Returns:
        A namespace containing all set configuration options

    Exceptions:
        None
    """
    # Get command line arguments
    parser = ArgumentParser(description=program_description,
                            epilog=epilogue,
                            formatter_class=RawDescriptionHelpFormatter)

    # Get global configuration file location from the command line
    parser.add_argument('config_pathname',
                        action='store',
                        type=str,
                        help='Global configuration file. ' + \
                             'This is required.')

    # Specify a target task configuration file that specifies a remote
    # directory location.
    if add_target:
        parser.add_argument('-t',
                            '--target_config_pathname',
                            dest='target_task_config',
                            action='store',
                            type=str,
                            help='specify a target task configuration file '
                                 'that contains a remote directory location',
                            metavar='FILE')

    # Date option
    if add_date:
        parser.add_argument('-d',
                            '--date',
                            dest='date',
                            action='store',
                            type=str,
                            help='calculate tasks for specified '
                                 'date, rather than the current date',
                            metavar='DATE')

    # Unique option
    parser.add_argument('-p',
                        '--pidfile',
                        action='store',
                        type=str,
                        dest='pidfile',
                        help='Enforce unique processing using '
                             ' lock file PIDFILE',
                        metavar='PIDFILE')

    # Get version information
    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s ' + str(__version__),
                        help='Display version number')

    # Log message filtering by level
    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        dest='verbose',
                        help='Display all info messages')

    parser.add_argument('-q',
                        '--quiet',
                        action='store_true',
                        dest='quiet',
                        help='Display only error messages')

    # Additional logging options
    parser.add_argument('-s',
                        '--syslog',
                        dest='log_facility',
                        action='store',
                        type=str,
                        help='Route logging messages to syslog facility FACILITY',
                        metavar='FACILITY')
    parser.add_argument('-l',
                        '--log',
                        dest='log_file',
                        action='store',
                        type=str,
                        help='Route logging messages to log file LOGFILE',
                        metavar='LOGFILE')
    parser.add_argument('-o',
                        '--stdout',
                        action='store_true',
                        dest='stdout',
                        help='Route logging messages to stdout')
    parser.add_argument('-e',
                        '--stderr',
                        action='store_true',
                        dest='stderr',
                        help='Route logging messages to stderr')

    # This can exit if the help option is requested
    return parser.parse_args()

############################################################################
#
#  init_logging
#
############################################################################
def init_logging():
    """ Initialize logging setup

    This method intializes logging for the program.

    Args:
        None

    Returns:
        None

    Exceptions:
        None
    """
    # Set up logging
    logging.basicConfig(format=DEFAULT_LOG_FORMAT,
                        datefmt=DEFAULT_DATE_FORMAT)

############################################################################
#
#  set_logging_options
#
############################################################################
def set_logging_options(options):
    """ Set logging options

    This method toggles any logging options set from the command line.

    Args:
        options (namespace argparse return) A namespace containing set
            variables from the command line. This is returned from the
            parse_args() method.

    Returns:
        None

    Exceptions:
        None
    """

    # Get logger
    logger = logging.getLogger()

    # Verbose logging?
    if options.verbose:
        log_level = logging.DEBUG
    elif options.quiet:
        log_level = logging.WARNING
    else:
        log_level = DEFAULT_LEVEL

    logger.setLevel(log_level)

    # Set up other logging destinations
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT,
                                  datefmt=DEFAULT_DATE_FORMAT)

    # Mark existing default handler to be removed if another handler
    # is specified
    if logger.handlers:
        default_handler = logger.handlers[0]
    else:
        default_handler = None

    # Log to standard out (and remove any default handler)
    if options.stdout:
        syshandler = logging.StreamHandler(stream=sys.stdout)
        syshandler.setFormatter(formatter)
        if default_handler:
            logger.removeHandler(default_handler)
            default_handler = None
        logger.addHandler(syshandler)

    # Log to standard error (and remove any default handler)
    if options.stderr:
        syshandler = logging.StreamHandler(stream=sys.stderr)
        syshandler.setFormatter(formatter)
        if default_handler:
            logger.removeHandler(default_handler)
            default_handler = None
        logger.addHandler(syshandler)

    # Log to syslog (and remove any default handler)
    if options.log_facility:
        sysformatter = logging.Formatter(DEFAULT_SYSLOG_FORMAT,
                                         datefmt=DEFAULT_DATE_FORMAT)
        syshandler = logging.handlers.SysLogHandler(address='/dev/log',
                                                    facility=options.log_facility)
        syshandler.setFormatter(sysformatter)
        if default_handler:
            logger.removeHandler(default_handler)
            default_handler = None
        logger.addHandler(syshandler)

    # Log to file (and remove any default handler)
    if options.log_file:
        filehandler = logging.FileHandler(options.log_file)
        filehandler.setFormatter(formatter)
        if default_handler:
            logger.removeHandler(default_handler)
            default_handler = None
        logger.addHandler(filehandler)
