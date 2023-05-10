# See https://stackoverflow.com/a/51028921 for why there is this file

import sys
import os

from pathlib import Path

module_path = str(Path(__file__).parents[1])
if module_path not in sys.path:
    sys.path.append(module_path)