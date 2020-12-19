from iris import logger


def log_function(func):
    def wrapper(*args, **kwargs):
        try:
            extras = next(
                str(kwargs.get(k))
                for k in ("file_path", "csv_path", "serial_path")
                if kwargs.get(k) is not None
            )
            logger.info(func.__name__.replace("_", " ").strip(), file=extras)
        except StopIteration:
            logger.info(func.__name__.replace("_", " ").strip())

        return func(*args, **kwargs)

    return wrapper
