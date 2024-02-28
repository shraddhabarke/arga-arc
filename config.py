from dataclasses import dataclass
import typing as t
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DIR_CONTAINING_FILE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    ROOT_DIR = Path(DIR_CONTAINING_FILE)
    OPENAI_SECRET_KEY: str = os.getenv("OPENAI_SECRET_KEY")
    OPENAI_ORGANIZATION: str = os.getenv("OPENAI_ORGANIZATION")


CONFIG = Config()
