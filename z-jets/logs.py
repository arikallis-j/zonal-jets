from loguru import logger
import sys

logger.add("z-jets/my_app.log")

class Runner:
    def __init__(self):
        logger.info("Hello, info!")
        logger.trace("Hello, trace!")
        logger.success("Hello, success")
        logger.warning("Hello, warning")
        logger.debug("Hello, debug")
        logger.error("Hello, error")
        logger.critical("Hello, critical")


app = Runner()
