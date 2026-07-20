"""Serve the browser demo and its allowlisted v3.0 release models."""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPOSITORY_ROOT / "web"
WEIGHTS_ROOT = REPOSITORY_ROOT / "weights"
RELEASE_TAG = "v3.0"
MODEL_NAMES = frozenset(
    f"raco_aliked_lightglue_pipeline_k{keypoints}.onnx" for keypoints in (512, 1024, 1536, 2048, 2560, 3072, 3584)
)
RELEASE_BASE_URL = f"https://github.com/fabio-sim/LightGlue-ONNX/releases/download/{RELEASE_TAG}"


class DemoRequestHandler(SimpleHTTPRequestHandler):
    """Static handler with a tightly scoped same-origin release proxy."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        model_name = self._release_model_name()
        if model_name is None:
            super().do_GET()
            return
        self._serve_release_model(model_name, send_body=True)

    def do_HEAD(self) -> None:
        model_name = self._release_model_name()
        if model_name is None:
            super().do_HEAD()
            return
        self._serve_release_model(model_name, send_body=False)

    def end_headers(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/release/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _release_model_name(self) -> str | None:
        path = unquote(urlparse(self.path).path)
        prefix = f"/release/{RELEASE_TAG}/"
        if not path.startswith(prefix):
            return None
        name = path.removeprefix(prefix)
        if name not in MODEL_NAMES:
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown release model")
            return ""
        return name

    def _serve_release_model(self, model_name: str, *, send_body: bool) -> None:
        if not model_name:
            return
        local_model = WEIGHTS_ROOT / model_name
        if local_model.is_file():
            self._serve_local_model(local_model, send_body=send_body)
            return
        self._proxy_release_model(model_name, send_body=send_body)

    def _send_model_headers(self, model_name: str, content_length: str | int | None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.send_header("Content-Disposition", f'inline; filename="{model_name}"')
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()

    def _serve_local_model(self, path: Path, *, send_body: bool) -> None:
        self._send_model_headers(path.name, path.stat().st_size)
        if not send_body:
            return
        with contextlib.suppress(BrokenPipeError, ConnectionResetError), path.open("rb") as model_file:
            shutil.copyfileobj(model_file, self.wfile, length=1024 * 1024)

    def _proxy_release_model(self, model_name: str, *, send_body: bool) -> None:
        method = "GET" if send_body else "HEAD"
        request = urllib.request.Request(
            f"{RELEASE_BASE_URL}/{model_name}", method=method, headers={"User-Agent": "LightGlue-ONNX-browser-demo"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                self._send_model_headers(model_name, response.headers.get("Content-Length"))
                if send_body:
                    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                        shutil.copyfileobj(response, self.wfile, length=1024 * 1024)
        except urllib.error.HTTPError as error:
            self.send_error(error.code, f"Release download failed: {error.reason}")
        except urllib.error.URLError as error:
            self.send_error(HTTPStatus.BAD_GATEWAY, f"Unable to reach the GitHub release: {error.reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Address to bind (default: %(default)s)")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind (default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"Serving the browser demo at http://{args.host}:{args.port}")
    print("Release models are served from weights/ when present, otherwise from GitHub v3.0.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.exit(main())
