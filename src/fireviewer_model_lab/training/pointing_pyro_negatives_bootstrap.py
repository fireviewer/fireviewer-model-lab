"""Install the pinned image decoder before running the Pyro-SDIS audit."""

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
            "pillow==11.3.0",
        ],
        check=True,
    )
    subprocess.run(  # noqa: S603 - launcher supplies the fixed script and arguments
        [sys.executable, *sys.argv[1:]],
        check=True,
    )


if __name__ == "__main__":
    main()
