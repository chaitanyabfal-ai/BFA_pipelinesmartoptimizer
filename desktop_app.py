"""Desktop launcher: runs the Streamlit app locally and shows it in a native window (pywebview).
Falls back to your default browser if pywebview is not installed.

Download fix: an embedded window silently drops browser-style downloads unless the webview
is told to allow them. This launcher (1) enables pywebview's ALLOW_DOWNLOADS (save-file dialog),
(2) keeps a persistent profile, and (3) sets BFA_DESKTOP=1 so the dashboard also shows
'Save to Downloads folder' buttons that write the file directly - works even if (1) is unsupported."""
import os, socket, subprocess, sys, time, urllib.request, webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]


def main():
    port = free_port()
    env = dict(os.environ, BFA_DESKTOP="1")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(HERE / "streamUIdashboard.py"),
                             "--server.headless", "true", "--server.port", str(port),
                             "--browser.gatherUsageStats", "false"], cwd=HERE, env=env)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(url + "/_stcore/health", timeout=1).read() == b"ok": break
            except Exception: time.sleep(0.5)
        try:
            import webview
            try: webview.settings["ALLOW_DOWNLOADS"] = True   # pywebview >= 5
            except Exception: pass
            webview.create_window("Bharat Flow Analytics — Pipeline Smart Optimizer", url, width=1500, height=950)
            storage = HERE / ".webview_profile"; storage.mkdir(exist_ok=True)
            try: webview.start(private_mode=False, storage_path=str(storage))
            except TypeError: webview.start()
        except ImportError:
            print("pywebview not installed - opening browser instead:", url)
            webbrowser.open(url); proc.wait()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
