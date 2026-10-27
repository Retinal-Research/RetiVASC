"""Allow ``python -m automorph_v2`` to invoke the public CLI."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
