from dataclasses import dataclass
import typing as t
import os
from pathlib import Path

DIR_CONTAINING_FILE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    ROOT_DIR = Path(DIR_CONTAINING_FILE)


CONFIG = Config()
