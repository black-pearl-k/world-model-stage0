"""在 PyCharm 里启动“只站立”实验的 TensorBoard。

右键这个文件 -> Run。
浏览器打开后，保持这个 Run 存活；在控制台按 Enter 停止 TensorBoard。
"""

from __future__ import annotations  # 让类型注解延迟求值

import subprocess  # 启动 TensorBoard 子进程
import sys  # 当前 Python 解释器
import time  # 等待 TensorBoard 启动
import webbrowser  # 自动打开浏览器

from train import PROJECT_DIR  # 项目目录

PORT = 6010  # 和 6007 / 6008 / 6009 区分开


def main() -> None:
    logdir = PROJECT_DIR / "logs" / "halfcheetah_stand"
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
