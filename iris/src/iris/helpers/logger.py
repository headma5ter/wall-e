import logging
import pathlib
import sys


class CustomFormatter(logging.Formatter):
    def __init__(self):
        super(CustomFormatter, self).__init__()

    def format(self, record: logging.LogRecord) -> str:
        extra = getattr(record, "with", "")
        extra_text = ""
        if extra:
            extra_text = f": {extra}"

        return f"({self.formatTime(record, self.datefmt)}) [{record.levelname}] => {record.getMessage()}{extra_text}"


class LogHandler:
    def __init__(self, file_path: pathlib.Path = None, level: str = "DEBUG"):
        self._level = level
        self._logger = self._initiate_logger()
        self._handlers = dict()

        # Write to stdout and file (if applicable)
        self.set_handler("iris", handler_dest=sys.stdout)
        if file_path is not None:
            self.set_handler("iris_file", handler_dest=file_path)

    def _initiate_logger(self) -> logging.Logger:
        logger = logging.getLogger("IRIS")
        logger.setLevel(self._level)
        return logger

    def set_handler(
        self, handler_name: str, handler_dest: (pathlib.Path, sys.stdout)
    ) -> None:
        if handler_name in self._handlers:
            self.info("Replacing logger output")
            self._logger.removeHandler(self._handlers[handler_name])

        handler = logging.StreamHandler(handler_dest)
        if isinstance(handler_dest, pathlib.Path):
            handler = logging.FileHandler(handler_dest)
        handler.setFormatter(CustomFormatter())

        self._logger.addHandler(handler)
        self._handlers[handler_name] = handler

    def debug(self, message: str, *args, **kwargs) -> None:
        self._logger.debug(message, *args, extra={"with": kwargs})

    def info(self, message: str, *args, **kwargs) -> None:
        self._logger.info(message, *args, extra={"with": kwargs})

    def warn(self, message: str, *args, **kwargs) -> None:
        self._logger.warning(message, *args, extra={"with": kwargs})

    def error(self, message: str, *args, **kwargs) -> None:
        self._logger.error(message, *args, extra={"with": kwargs})
