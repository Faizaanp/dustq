"""Bootstrap imports for the vendored upstream clone in third_party/dust.

Upstream dust.py does `from model import ...`, so third_party/dust must be first on
sys.path before `import dust`. The clone is pinned (see UPSTREAM_COMMIT) and never edited.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = REPO_ROOT / "third_party" / "dust"


def ensure_upstream_on_path() -> Path:
    if not UPSTREAM_DIR.is_dir():
        raise FileNotFoundError(
            f"Upstream clone missing at {UPSTREAM_DIR}. Run: "
            "git clone https://github.com/qlabs-eng/dust third_party/dust "
            "(then record the pinned commit in third_party/dust/UPSTREAM_COMMIT)."
        )
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    return UPSTREAM_DIR


ensure_upstream_on_path()
