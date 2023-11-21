from os.path import dirname, abspath
import datetime

PROJECT_ROOT = dirname(abspath(__file__))

def get_timestamp(micros=False):
    fmt = "%Y%m%dT%H%M%S"
    if micros:
        fmt += "m%f"
    return datetime.datetime.now().strftime(fmt)