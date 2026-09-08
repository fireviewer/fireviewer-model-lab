"""Install the two pinned pure-Python readers, then run the ActiveFire audit."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "remotezip==0.12.6",
            "tifffile==2024.9.20",
        ],
        check=True,
    )
    from pointing_activefire_audit import main as audit_main

    audit_main()


if __name__ == "__main__":
    main()
