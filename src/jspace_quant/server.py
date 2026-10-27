"""Start and stop jlens-server for one model."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from jlens_gguf.client import NativeClient

from jspace_quant import config

logger = logging.getLogger(__name__)


def _port_in_use(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


@contextmanager
def native_server(model_path: Path, *, port: int = config.SERVER_PORT, gpu: str = config.SERVER_GPU,
                  log_path: Path | None = None):
    """Starts jlens-server with all layers on `gpu` and yields a NativeClient."""
    if _port_in_use(port):
        raise RuntimeError(f"port {port} is in use — is a jlens-server already running?")
    if not config.JLENS_SERVER_BIN.exists():
        raise FileNotFoundError(f"{config.JLENS_SERVER_BIN} missing — see docs/en/03-setup.md")
    cmd = [str(config.JLENS_SERVER_BIN), "-m", str(model_path), "--port", str(port),
           "-c", str(config.SERVER_CTX), "-ngl", "99"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    log_file = open(log_path, "w") if log_path else None
    logger.info("starting jlens-server: %s (GPU %s)", model_path.name, gpu)
    proc = subprocess.Popen(cmd, env=env, stdout=log_file or subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        client = NativeClient(f"http://127.0.0.1:{port}")
        for _ in range(600):
            if proc.poll() is not None:
                raise RuntimeError(f"jlens-server exited with code {proc.returncode} (log: {log_path})")
            if client.health():
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("jlens-server did not become ready")
        served = Path(client.props().get("model_path", "")).resolve()
        if served != model_path.resolve():
            raise RuntimeError(f"server serves {served}, expected {model_path}")
        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_file:
            log_file.close()
