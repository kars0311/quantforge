"""Make `src/` importable in tests without installing the package (e.g. `from quantforge...`)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
