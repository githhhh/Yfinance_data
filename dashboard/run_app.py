import argparse
import os
import signal
import socket
import subprocess
import sys
import webbrowser
from pathlib import Path


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local breakout pool dashboard.")
    default_csv = Path(__file__).resolve().parents[1] / "us" / "breakout_follow_pool.csv"
    parser.add_argument("--csv", default=str(default_csv))
    parser.add_argument("--server-port", default="8501")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without opening browser.")
    args = parser.parse_args()

    port = int(args.server_port)
    url = f"http://localhost:{port}"

    if is_port_in_use(port):
        if not args.headless:
            print(f"[INFO] Streamlit 服务已在端口 {port} 运行中，正在为您自动打开可视化面板: {url}")
            webbrowser.open(url)
        else:
            print(f"[INFO] Streamlit 服务已在端口 {port} 运行中: {url}")
        return 0

    app_dir = Path(__file__).resolve().parent
    app_path = app_dir / "app.py"
    csv_path = Path(args.csv).expanduser()
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path.name),
        "--server.port",
        str(args.server_port),
        "--server.headless",
        "true" if args.headless else "false",
        "--",
        "--csv",
        str(csv_path),
    ]

    if os.name == "posix":
        # 在 macOS/Linux 上使用 execvp 替换当前 Python 进程，不创建子进程。
        # 这样终端关闭(SIGHUP)或 Ctrl+C(SIGINT) 会直接发送给 Streamlit 进程，即时彻底关闭服务，绝不残留后台孤儿进程。
        os.chdir(app_dir)
        os.execvp(command[0], command)
    else:
        # 非 POSIX 系统 (Windows 等) 的兜底子进程管理与信号转发
        with subprocess.Popen(command, cwd=app_dir) as proc:
            def cleanup(signum: int, frame: object) -> None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                sys.exit(0)

            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, cleanup)
            return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
