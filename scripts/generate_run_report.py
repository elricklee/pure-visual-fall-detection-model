from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.runtime.generate_run_report import *  # noqa: F401,F403
from scripts.runtime.generate_run_report import generate_report, main


if __name__ == "__main__":
    main()
