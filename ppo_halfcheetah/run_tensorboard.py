"""Start HalfCheetah TensorBoard from PyCharm without a terminal command.

In PyCharm: right-click this file -> Run.
After the browser opens, keep this run alive; press Enter in the Run console
to stop TensorBoard.
"""

from __future__ import annotations  # 让类型注解延迟求值

import subprocess  # 启动 TensorBoard 子进程
import sys  # 获取当前 Python 解释器路径
import time  # 等待 TensorBoard 启动
import webbrowser  # 自动打开浏览器
from pathlib import Path  # 处理文件路径

from train import PROJECT_DIR  # 导入项目目录

PORT = 6007  # TensorBoard 端口，和 CartPole 的 6006 区分开


def main() -> None:  # 定义主函数
    logdir = PROJECT_DIR / "logs" / "halfcheetah"  # HalfCheetah 日志目录
    logdir.mkdir(parents=True, exist_ok=True)  # 如果日志目录不存在就创建

    url = f"http://localhost:{PORT}"  # TensorBoard 访问地址
    print(f"Starting TensorBoard: {url}")  # 打印访问地址
    print(f"logdir: {logdir}")  # 打印日志目录
    print("Press Enter in this Run console to stop TensorBoard.\n")  # 提示如何停止

    proc = subprocess.Popen(  # 启动 TensorBoard 子进程
        [
            sys.executable,  # 使用当前 Python 解释器
            "-m",  # 以模块方式运行
            "tensorboard.main",  # TensorBoard 主模块
            "--logdir",  # 指定日志目录
            str(logdir),  # 日志目录路径
            "--port",  # 指定端口
            str(PORT),  # 端口号
        ]
    )

    try:  # 尝试等待用户按 Enter
        time.sleep(3)  # 给 TensorBoard 3 秒启动时间
        webbrowser.open(url)  # 自动打开浏览器
        input()  # 等待用户在 Run 控制台按 Enter
    finally:  # 无论用户正常停止还是按了 Ctrl+C，都执行清理
        proc.terminate()  # 请求 TensorBoard 子进程退出
        try:  # 尝试优雅等待
            proc.wait(timeout=5)  # 最多等 5 秒
        except subprocess.TimeoutExpired:  # 如果 5 秒还没退出
            proc.kill()  # 强制结束子进程
        print("TensorBoard stopped.")  # 打印停止信息


if __name__ == "__main__":  # 只有直接运行这个文件时才执行
    main()  # 调用主函数
