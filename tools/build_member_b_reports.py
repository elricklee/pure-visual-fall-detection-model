from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.reports.build_member_b_reports import *  # noqa: F401,F403
from tools.reports.build_member_b_reports import main


if __name__ == "__main__":
    main()
