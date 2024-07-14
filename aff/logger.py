# --------------------------------------------------------
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import time
from typing import Optional
import sys
import os

text_colors = {
    "logs": "\033[34m",  # 033 is the escape code and 34 is the color code
    "info": "\033[32m",
    "warning": "\033[33m",
    "debug": "\033[93m",
    "error": "\033[31m",
    "bold": "\033[1m",
    "end_color": "\033[0m",
    "light_red": "\033[36m",
}


def get_curr_time_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def error(message: str) -> None:
    time_stamp = get_curr_time_stamp()
    error_str = (
        text_colors["error"]
        + text_colors["bold"]
        + "ERROR  "
        + text_colors["end_color"]
    )

    # exiting with code -1 does not tell any information about the error (e.g., NaN encountered in the loss).
    # For more descriptive error messages, we replace exit(-1) with sys.exit(ERROR_MESSAGE).
    # This allows us to handle specific exceptions in the tests.

    # print("{} - {} - {}".format(time_stamp, error_str, message), flush=True)
    # print("{} - {} - {}".format(time_stamp, error_str, "Exiting!!!"), flush=True)
    # exit(-1)

    sys.exit("{} - {} - {}. Exiting!!!".format(time_stamp, error_str, message))

