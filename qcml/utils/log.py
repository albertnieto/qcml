# Copyright 2024 Albert Nieto

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json
import dask.config
import sys

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    slack_available = True
except ImportError:
    slack_available = False

# Constants
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
)
DEFAULT_JSON_FORMAT = json.dumps(
    {
        "time": "%(asctime)s",
        "name": "%(name)s",
        "level": "%(levelname)s",
        "message": "%(message)s",
        "filename": "%(filename)s",
        "line": "%(lineno)d",
    }
)
DEFAULT_LOG_FILENAME = f"qcml-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
DEFAULT_LOG_PATH = "logs/"
DEFAULT_LOG_FILE_MAX_BYTES = 209715200

def setup_evaluate_model_logging(logger_name):
    # Extract log file name and other settings from environment variables
    log_filename = os.getenv('QCML_DASK_LOG_FILENAME')
    logs_path = os.getenv('QCML_DASK_LOGS_PATH', 'logs/')
    log_format = os.getenv('QCML_DASK_LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')
    log_level = os.getenv('QCML_DASK_LEVEL', 'INFO')
    terminal_level = os.getenv('QCML_DASK_TERMINAL_LEVEL', log_level)
    use_color = os.getenv('QCML_DASK_USE_COLOR', 'True') == 'True'

    # Set up the logger with the specific name
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Set up formatter
    if use_color:
        formatter = ColoredFormatter(log_format)
    else:
        formatter = logging.Formatter(log_format)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, terminal_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Setup file handler if log file name is specified
    if log_filename:
        file_handler = logging.FileHandler(os.path.join(logs_path, log_filename))
        file_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def set_dask_environment_variables(
    level=None,
    hide_logs=None,
    output=None,
    logs_path=None,
    log_filename=None,
    max_bytes=None,
    backup_count=None,
    terminal_level=None,
    file_level=None,
    log_format=None,
    use_json=False,
    keyword_filters=None,
    use_color=True,
    asynchronous=False,
    add_context=False,
    context_info=None,
    slack_notify=False,
    slack_credentials=None
):
    """
    Sets environment variables for Dask logging configuration based on log setup parameters.
    Environment variables are prefixed with 'QCML_DASK_'.
    """
    os.environ['QCML_DASK_LEVEL'] = level or 'INFO'
    os.environ['QCML_DASK_OUTPUT'] = output or 'both'
    os.environ['QCML_DASK_LOGS_PATH'] = logs_path or 'logs/'
    os.environ['QCML_DASK_LOG_FILENAME'] = log_filename or f"qcml-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    os.environ['QCML_DASK_MAX_BYTES'] = str(max_bytes or 1048576)
    os.environ['QCML_DASK_BACKUP_COUNT'] = str(backup_count or 3)
    os.environ['QCML_DASK_TERMINAL_LEVEL'] = terminal_level or level or 'INFO'
    os.environ['QCML_DASK_FILE_LEVEL'] = file_level or level or 'DEBUG'
    os.environ['QCML_DASK_LOG_FORMAT'] = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    os.environ['QCML_DASK_USE_JSON'] = str(use_json)
    os.environ['QCML_DASK_USE_COLOR'] = str(use_color)
    os.environ['QCML_DASK_ASYNCHRONOUS'] = str(asynchronous)
    os.environ['QCML_DASK_ADD_CONTEXT'] = str(add_context)
    os.environ['QCML_DASK_SLACK_NOTIFY'] = str(slack_notify)

    if slack_credentials:
        os.environ['QCML_DASK_SLACK_TOKEN'] = slack_credentials[0]
        os.environ['QCML_DASK_SLACK_CHANNEL'] = slack_credentials[1]
    
    if hide_logs:
        os.environ['QCML_DASK_HIDE_LOGS'] = ','.join(hide_logs)
    
    if keyword_filters:
        os.environ['QCML_DASK_KEYWORD_FILTERS'] = ','.join(keyword_filters)
    
    if context_info:
        os.environ['QCML_DASK_CONTEXT_INFO'] = json.dumps(context_info)

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
        "DEBUG": "\033[92m",  # Green
        "INFO": "\033[94m",  # Blue
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
    }
    RESET = "\033[0m"

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
            self.slack_client.chat_postMessage(
                channel=self.channel, text=f"```{log_entry}```"
            )
        except SlackApiError as e:
            logging.error(f"Failed to send log to Slack: {e.response['error']}")

def setup_formatter(use_json, use_color, log_format):
    if use_json:
        return logging.Formatter(fmt=DEFAULT_JSON_FORMAT)
    elif use_color:
        return ColoredFormatter(log_format)
    else:
        return logging.Formatter(log_format)


def setup_handlers(
    output,
    logs_path,
    log_filename,
    max_bytes,
    backup_count,
    terminal_level,
    formatter,
    file_level,
):
    handlers = []

    if output in ("terminal", "both"):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(
            terminal_level
        )  # Use terminal_level for the stream handler
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)

    if output in ("file", "both"):
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        file_handler = RotatingFileHandler(
            os.path.join(logs_path, log_filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(file_level)  # Use file_level for the file handler
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
        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


def setup_slack_handler(slack_credentials):
    if not slack_credentials or len(slack_credentials) != 2:
        raise ValueError(
            "Slack credentials must be a list containing the token and channel."
        )
    slack_token, slack_channel = slack_credentials
    slack_client = WebClient(token=slack_token)

    # Check Slack connection
    try:
        response = slack_client.auth_test()
        if not response["ok"]:
            raise SlackApiError("Slack authentication failed", response)
        logging.info(f"Slack connected successfully: {response['url']}")
    except SlackApiError as e:
        logging.error(f"Failed to authenticate Slack: {e.response['error']}")
        raise

    return SlackHandler(slack_client, slack_channel)


def log_setup(
    level=None,
    hide_logs=None,
    output=None,
    logs_path=DEFAULT_LOG_PATH,
    log_filename=DEFAULT_LOG_FILENAME,
    max_bytes=DEFAULT_LOG_FILE_MAX_BYTES,
    backup_count=3,
    terminal_level=None,
    file_level=None,
    log_format=DEFAULT_LOG_FORMAT,
    use_json=False,
    keyword_filters=None,
    use_color=True,
    asynchronous=False,
    add_context=False,
    context_info=None,
    slack_notify=False,
    slack_credentials=None,
    config_dask=False,
):
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
    config_dask (bool): Whether to configure Dask logging settings. Default is False.
    """

    if output is None:
        output = "both"  # Default to both if output is not specified

    if level is None and terminal_level is None and file_level is None:
        raise ValueError(
            "You must specify at least one of 'level', 'terminal_level', or 'file_level'."
        )

    if level:
        terminal_level = terminal_level or level
        file_level = file_level or level

    # Create a single logger variable (the root logger)
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # Remove existing handlers from the root logger
    logger.handlers.clear()

    # Setup formatter
    formatter = setup_formatter(use_json, use_color, log_format)

    # Setup and add handlers
    terminal_handler = None if terminal_level is None else terminal_level.upper()
    file_handler = None if file_level is None else file_level.upper()
    handlers = setup_handlers(
        output,
        logs_path,
        log_filename,
        max_bytes,
        backup_count,
        terminal_handler,
        formatter,
        file_handler,
    )

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
        logger.warning(
            "Slack notifications are enabled, but the Slack SDK is not installed. Please install it to use this feature."
        )

    # Setup filters and context
    setup_logging_filters(
        logger, keyword_filters, context_info if add_context else None
    )

    # Setup asynchronous logging
    if asynchronous:
        setup_async_logging(logger)

    logger.info("Logging is set up.")

    # Configure Dask logging if requested
    if config_dask:
        set_dask_environment_variables(
            level=level,
            hide_logs=hide_logs,
            output=output,
            logs_path=logs_path,
            log_filename=log_filename,
            max_bytes=max_bytes,
            backup_count=backup_count,
            terminal_level=terminal_level,
            file_level=file_level,
            log_format=log_format,
            use_json=use_json,
            keyword_filters=keyword_filters,
            use_color=use_color,
            asynchronous=asynchronous,
            add_context=add_context,
            context_info=context_info,
            slack_notify=slack_notify,
            slack_credentials=slack_credentials
        )
        logger.info("Environment variables for Dask logging setup have been set.")

    # Set logging levels for specified libraries
    if hide_logs:
        for library in hide_logs:
            logging.getLogger(library).setLevel(logging.ERROR)
            logger.info(f"{library} logs are set to ERROR level.")


# Example usage in main
if __name__ == "__main__":
    context_info = {"user_id": "12345", "session_id": "abcde"}

    # Example of passing credentials securely
    slack_token = os.getenv("SLACK_API_TOKEN")
    slack_channel = "#logging-channel"
    slack_credentials = [slack_token, slack_channel]

    log_setup(
        level="DEBUG",
        output="both",
        logs_path="my_logs",
        use_json=False,
        keyword_filters=["error", "critical"],
        use_color=True,
        asynchronous=True,
        add_context=True,
        context_info=context_info,
        slack_notify=True,
        slack_credentials=slack_credentials,
    )

    # Example logging
    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")
