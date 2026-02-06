from __future__ import annotations

import sys
from pathlib import Path


def ensure_open_instruct_importable() -> None:
    """
    Make sibling `rl/open-instruct/open_instruct` importable as `open_instruct`.

    This keeps open-instruct2 runnable without requiring a separate editable install.
    """
    try:
        import open_instruct  # noqa: F401

        return
    except ImportError:
        pass

    this_file = Path(__file__).resolve()
    rl_dir = this_file.parents[2]
    candidate = rl_dir / "open-instruct"
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        import open_instruct  # noqa: F401

        return

    raise ImportError(
        "Could not import `open_instruct`. Expected sibling path at "
        f"{candidate}. Either install open-instruct or set PYTHONPATH manually."
    )
