import logging
import os
import json

def dask_setup(worker):
    # Read environment variables
    log_level = os.getenv('QCML_DASK_LEVEL', 'INFO').upper()
    output = os.getenv('QCML_DASK_OUTPUT', 'both')
    logs_path = os.getenv('QCML_DASK_LOGS_PATH', 'logs/')
    log_filename = os.getenv('QCML_DASK_LOG_FILENAME', 'dask_worker.log')
    max_bytes = int(os.getenv('QCML_DASK_MAX_BYTES', '1048576'))
    backup_count = int(os.getenv('QCML_DASK_BACKUP_COUNT', '3'))
    terminal_level = os.getenv('QCML_DASK_TERMINAL_LEVEL', log_level).upper()
    file_level = os.getenv('QCML_DASK_FILE_LEVEL', log_level).upper()
    log_format = os.getenv('QCML_DASK_LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')
    use_json = os.getenv('QCML_DASK_USE_JSON', 'False') == 'True'
    use_color = os.getenv('QCML_DASK_USE_COLOR', 'True') == 'True'
    asynchronous = os.getenv('QCML_DASK_ASYNCHRONOUS', 'False') == 'True'
    add_context = os.getenv('QCML_DASK_ADD_CONTEXT', 'False') == 'True'
    slack_notify = os.getenv('QCML_DASK_SLACK_NOTIFY', 'False') == 'True'
    
    slack_token = os.getenv('QCML_DASK_SLACK_TOKEN')
    slack_channel = os.getenv('QCML_DASK_SLACK_CHANNEL')
    hide_logs = os.getenv('QCML_DASK_HIDE_LOGS')
    keyword_filters = os.getenv('QCML_DASK_KEYWORD_FILTERS')
    context_info = os.getenv('QCML_DASK_CONTEXT_INFO')

    if context_info:
        context_info = json.loads(context_info)
    
    if hide_logs:
        hide_logs = hide_logs.split(',')
    
    if keyword_filters:
        keyword_filters = keyword_filters.split(',')
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Setup formatter
    formatter = logging.Formatter(log_format)

    # Add handlers based on output configuration
    if output in ('terminal', 'both'):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, terminal_level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if output in ('file', 'both'):
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(logs_path, log_filename),
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, file_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Slack handler setup (if needed)
    if slack_notify and slack_token and slack_channel:
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
            
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

            slack_client = WebClient(token=slack_token)
            slack_handler = SlackHandler(slack_client, slack_channel)
            slack_handler.setFormatter(formatter)
            logger.addHandler(slack_handler)

        except ImportError:
            logging.error("Slack SDK not installed. Slack notifications will not work.")

    # Set logging levels for specified libraries
    for library in ["pennylane","jax","distributed","matplotlib"]:
        logging.getLogger(library).setLevel(logging.INFO)
        logger.info(f"{library} logs are set to INFO level.")

    
    logging.info("Dask worker logging setup complete.")
    worker.qml_devices = {}