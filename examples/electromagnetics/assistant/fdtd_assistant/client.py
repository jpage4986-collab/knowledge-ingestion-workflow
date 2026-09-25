"""Small, dependency-free Python client for the knowledge runtime API."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class FDTDClientError(RuntimeError):
    """Raised when the runtime cannot return a valid API response."""


class FDTDClient:
    """Call the runtime API from another agent or a Python service.

    The client deliberately uses only the Python standard library so it can be
    embedded in simulation and drawing agents without adding a dependency.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, *, method: str = "GET", payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                status = getattr(response, "status", 200)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FDTDClientError(f"API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise FDTDClientError(f"API connection failed: {exc.reason}") from exc
        try:
            result = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FDTDClientError("API returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise FDTDClientError("API returned a non-object response")
        if status >= 400:
            raise FDTDClientError(f"API HTTP {status}: {result}")
        return result

    def health(self) -> dict:
        return self._request("/health")

    def answer(self, question: str, *, allow_web: bool = False, max_iterations: int = 2, include_full_page: bool = True) -> dict:
        return self._request(
            "/answer",
            method="POST",
            payload={"question": question, "allow_web": allow_web, "max_iterations": max_iterations, "include_full_page": include_full_page},
        )

    def validate_simulation(self, **spec: object) -> dict:
        return self._request("/simulation/validate", method="POST", payload=spec)

    def visualization_task(self, **spec: object) -> dict:
        return self._request("/visualization/task", method="POST", payload=spec)

    def lease_pdf(self, source: str, *, ttl_seconds: int = 900, filename: str | None = None) -> dict:
        """Create a temporary PDF lease; call ``release_pdf`` after reading it."""
        payload: dict[str, object] = {"source": source, "ttl_seconds": ttl_seconds}
        if filename is not None:
            payload["filename"] = filename
        return self._request("/pdf/lease", method="POST", payload=payload)

    def download_pdf(self, lease_id: str, destination: str | Path) -> Path:
        """Download a leased PDF, leaving lifecycle release to the caller."""
        request = Request(
            f"{self.base_url}/pdf/lease/{lease_id}",
            headers={"Accept": "application/pdf"},
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise FDTDClientError(f"PDF HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise FDTDClientError(f"PDF connection failed: {exc.reason}") from exc
        output = Path(destination)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
        return output

    def release_pdf(self, lease_id: str) -> dict:
        return self._request(f"/pdf/lease/{lease_id}", method="DELETE")

    def cleanup_pdf_leases(self) -> dict:
        return self._request("/pdf/cleanup", method="POST")

    @contextmanager
    def pdf_lease(
        self,
        source: str,
        *,
        destination: str | Path,
        ttl_seconds: int = 900,
        filename: str | None = None,
    ):
        """Download a PDF and release the server-side lease on exit."""
        lease = self.lease_pdf(source, ttl_seconds=ttl_seconds, filename=filename)
        lease_id = str(lease["lease_id"])
        output = Path(destination)
        if output.exists():
            self.release_pdf(lease_id)
            raise FileExistsError(f"refusing to overwrite existing PDF: {output}")
        try:
            yield self.download_pdf(lease_id, output)
        finally:
            self.release_pdf(lease_id)
            if output.is_file():
                output.unlink()
