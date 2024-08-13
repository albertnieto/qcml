import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
import sys

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    slack_available = True
except ImportError:
    slack_available = False

# Constants
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
DEFAULT_JSON_FORMAT = json.dumps({
    'time': '%(asctime)s',
    'name': '%(name)s',
    'level': '%(levelname)s',
    'message': '%(message)s',
    'filename': '%(filename)s',
    'line': '%(lineno)d'
})
DEFAULT_LOG_FILENAME = f"qcml-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

class ContextFilter(logging.Filter):
    def __init__(self, context_info):
        super().__init__()
        self.context_info = context_info

    def filter(self, record):
        for key, value in self.context_info.items():
            setattr(record, key, value)
        return True

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[92m',  # Green
        'INFO': '\033[94m',   # Blue
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m' # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        colored_levelname = f"{self.COLORS.get(record.levelname, self.RESET)}{record.levelname}{self.RESET}"
        record.levelname = colored_levelname
        return super().format(record)

class SlackHandler(logging.Handler):
    def __init__(self, slack_client, channel):
        super().__init__(level=logging.INFO)
        self.slack_client = slack_client
        self.channel = channel

    def emit(self, record):
        log_entry = self.format(record)
        try:
            self.slack_client.chat_postMessage(channel=self.channel, text=f"```{log_entry}```")
        except SlackApiError as e:
            logging.error(f"Failed to send log to Slack: {e.response['error']}")

def setup_formatter(use_json, use_color, log_format):
    if use_json:
        return logging.Formatter(fmt=DEFAULT_JSON_FORMAT)
    elif use_color:
        return ColoredFormatter(log_format)
    else:
        return logging.Formatter(log_format)

def setup_handlers(output, logs_path, log_filename, max_bytes, backup_count, level, formatter):
    handlers = []

    if output in ("terminal", "both"):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    if output in ("file", "both"):
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        file_handler = RotatingFileHandler(os.path.join(logs_path, log_filename), maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    return handlers

def setup_logging_filters(logger, keyword_filters, context_info):
    if keyword_filters:
        def filter_keywords(record):
            return any(keyword in record.msg for keyword in keyword_filters)
        keyword_filter = logging.Filter()
        keyword_filter.filter = filter_keywords
        logger.addFilter(keyword_filter)

    if context_info:
        context_filter = ContextFilter(context_info)
        logger.addFilter(context_filter)

def setup_async_logging(logger):
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

def setup_slack_handler(slack_credentials):
    if not slack_credentials or len(slack_credentials) != 2:
        raise ValueError("Slack credentials must be a list containing the token and channel.")
    slack_token, slack_channel = slack_credentials
    slack_client = WebClient(token=slack_token)
    
    # Check Slack connection
    try:
        response = slack_client.auth_test()
        if not response['ok']:
            raise SlackApiError("Slack authentication failed", response)
        logging.info(f"Slack connected successfully: {response['url']}")
    except SlackApiError as e:
        logging.error(f"Failed to authenticate Slack: {e.response['error']}")
        raise

    return SlackHandler(slack_client, slack_channel)

def log_setup(level="ERROR", hide_logs=None, output="both", logs_path="logs/",
              log_filename=DEFAULT_LOG_FILENAME, max_bytes=1048576, backup_count=3, 
              terminal_level=None, file_level=None, 
              log_format=DEFAULT_LOG_FORMAT, use_json=False, keyword_filters=None, 
              use_color=True, asynchronous=True, add_context=False, context_info=None,
              slack_notify=False, slack_credentials=None):
    """
    Set up logging to display logs in a Jupyter notebook and/or save to a file.

    Parameters:
    level (str): Default logging level as a string (e.g., 'DEBUG', 'INFO').
    hide_logs (list): List of libraries whose logs should be hidden (set to ERROR level).
    output (str): Where to output logs. Options are 'terminal', 'file', or 'both'. Default is 'both'.
    logs_path (str): Directory path where log files should be saved. Default is 'logs/'.
    log_filename (str): Custom log file name. Default is 'qcml-{timestamp}.log'.
    max_bytes (int): Maximum file size in bytes before rotating. Default is 1MB.
    backup_count (int): Number of backup files to keep. Default is 3.
    terminal_level (str): Logging level for terminal output. Default is the 'level' parameter.
    file_level (str): Logging level for file output. Default is the 'level' parameter.
    log_format (str): Log message format. Default is a detailed format.
    use_json (bool): Whether to use JSON format for logs. Default is False.
    keyword_filters (list): List of keywords to filter logs. Only logs containing these keywords will be shown. Default is None.
    use_color (bool): Whether to use color-coded logs in the terminal. Default is True.
    asynchronous (bool): Whether to use asynchronous logging. Default is True.
    add_context (bool): Whether to add contextual information to logs. Default is False.
    context_info (dict): Contextual information to be added to logs. Default is None.
    slack_notify (bool): Whether to send notifications to Slack. Default is False.
    slack_credentials (list): Slack API token and channel as a list. Required if slack_notify is True.
    """

    # Default terminal and file levels to the main logging level
    terminal_level = terminal_level or level
    file_level = file_level or level

    # Setup logging levels
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.ERROR))

    # Remove existing handlers
    logger.handlers.clear()

    # Setup formatter
    formatter = setup_formatter(use_json, use_color, log_format)

    # Setup and add handlers
    handlers = setup_handlers(output, logs_path, log_filename, max_bytes, backup_count, 
                              getattr(logging, terminal_level.upper(), logging.ERROR), 
                              formatter)
    
    for handler in handlers:
        logger.addHandler(handler)

    # Log whether Slack SDK is available
    if slack_notify and slack_available:
        try:
            slack_handler = setup_slack_handler(slack_credentials)
            slack_handler.setFormatter(formatter)
            logger.addHandler(slack_handler)
        except (ValueError, SlackApiError) as e:
            logger.error(f"Slack setup failed: {str(e)}")
    elif slack_notify and not slack_available:
        logger.warning("Slack notifications are enabled, but the Slack SDK is not installed. Please install it to use this feature.")

    # Set logging levels for specified libraries
    if hide_logs:
        for library in hide_logs:
            logging.getLogger(library).setLevel(logging.ERROR)
            logger.info(f"{library} logs are set to ERROR level.")

    # Setup filters and context
    setup_logging_filters(logger, keyword_filters, context_info if add_context else None)

    # Setup asynchronous logging
    if asynchronous:
        setup_async_logging(logger)

    logger.info("Logging is set up.")

# Example usage in main
if __name__ == "__main__":
    context_info = {
        'user_id': '12345',
        'session_id': 'abcde'
    }
    
    # Example of passing credentials securely
    slack_token = os.getenv("SLACK_API_TOKEN")
    slack_channel = "#logging-channel"
    slack_credentials = [slack_token, slack_channel]

    log_setup(level="DEBUG", output="both", logs_path="my_logs", 
              use_json=False, keyword_filters=["error", "critical"],
              use_color=True, asynchronous=True, add_context=True, context_info=context_info,
              slack_notify=True, slack_credentials=slack_credentials)
    
    # Example logging
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
