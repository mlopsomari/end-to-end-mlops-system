import logging
import mlflow
import joblib
from pathlib import Path
import sys
import uuid
import pandas as pd
from typing import Union
import sqlite3
from datetime import datetime, timezone

from mako.runtime import capture


def configure_logging():
    """Configure logging handlers and return a logger instance."""
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )
