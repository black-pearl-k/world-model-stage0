"""Start TensorBoard from PyCharm without typing a terminal command.

In PyCharm: right-click this file -> Run.
After the browser opens, keep this run alive; press Enter in the Run console
to stop TensorBoard.
"""

from __future__ import annotations

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from train import PROJECT_DIR

PORT = 6006


def main() -> None:
    logdir = PROJECT_DIR / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    url = f"http://localhost:{PORT}"
    print(f"Starting TensorBoard: {url}")
    print(f"logdir: {logdir}")
    print("Press Enter in this Run console to stop TensorBoard.\n")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(logdir),
            "--port",
            str(PORT),
        ]
    )

    try:
        time.sleep(3)
        webbrowser.open(url)
        input()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("TensorBoard stopped.")


if __name__ == "__main__":
    main()
