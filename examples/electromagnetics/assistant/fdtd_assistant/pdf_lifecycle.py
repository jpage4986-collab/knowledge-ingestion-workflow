"""Safe staging and cleanup for temporary PDF supplements.

The literature index owns the original document.  This module only creates a
short-lived, controlled copy (or HTTPS download) for downstream agents and
removes that copy when the caller releases it or when its TTL expires.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


_SAFE_LEASE_ID = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class PdfLease:
    """A temporary PDF copy that remains valid until release or expiration."""

    lease_id: str
    path: Path
    source: str
    created_at: float
    expires_at: float
    filename: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "path": str(self.path),
            "source": self.source,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "cleanup": {
                "release_endpoint": f"/pdf/lease/{self.lease_id}",
                "expires_at": self.expires_at,
            },
        }


class PdfLeaseManager:
    """Manage bounded, temporary PDF copies without touching source files.

    Local paths and URLs are deny-by-default.  Configure ``allowed_roots`` for
    local literature directories and ``allowed_url_hosts`` for approved HTTPS
    hosts.  The manager never deletes an original source path.
    """

    def __init__(
        self,
        *,
        root: str | Path | None = None,
        allowed_roots: list[str | Path] | tuple[str | Path, ...] = (),
        allowed_url_hosts: list[str] | tuple[str, ...] = (),
        default_ttl_seconds: int = 900,
        max_bytes: int = 200 * 1024 * 1024,
        request_timeout_seconds: float = 30.0,
    ) -> None:
        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.root = Path(root or (Path(tempfile.gettempdir()) / "em-knowledge-answer-assistant-pdf-leases")).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = tuple(Path(item).resolve() for item in allowed_roots)
        self.allowed_url_hosts = tuple(host.lower().strip() for host in allowed_url_hosts if host.strip())
        self.default_ttl_seconds = default_ttl_seconds
        self.max_bytes = max_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self._lock = threading.RLock()

    def acquire(
        self,
        source: str | Path,
        *,
        ttl_seconds: int | None = None,
        filename: str | None = None,
    ) -> PdfLease:
        """Create a temporary PDF lease from an allowed path or HTTPS URL."""

        ttl = self.default_ttl_seconds if ttl_seconds is None else int(ttl_seconds)
        if ttl <= 0:
            raise ValueError("ttl_seconds must be positive")
        source_value = str(source)
        parsed = urlparse(source_value)
        lease_id = uuid.uuid4().hex
        lease_dir = self._lease_dir(lease_id)
        lease_dir.mkdir(parents=False, exist_ok=False)
        try:
            chosen_name = self._safe_filename(filename or self._name_from_source(source_value))
            payload_path = lease_dir / chosen_name
            if parsed.scheme in {"http", "https"}:
                self._download_url(source_value, payload_path, parsed)
            elif parsed.scheme == "file":
                source_path = Path(unquote(parsed.path))
                self._copy_local(source_path, payload_path)
            else:
                self._copy_local(Path(source_value), payload_path)
            self._validate_pdf(payload_path)
            now = time.time()
            lease = PdfLease(
                lease_id=lease_id,
                path=payload_path,
                source=source_value,
                created_at=now,
                expires_at=now + ttl,
                filename=payload_path.name,
                size_bytes=payload_path.stat().st_size,
            )
            (lease_dir / "lease.json").write_text(
                json.dumps(lease.as_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return lease
        except Exception:
            self._remove_lease_dir(lease_dir)
            raise

    @contextmanager
    def lease(
        self,
        source: str | Path,
        *,
        ttl_seconds: int | None = None,
        filename: str | None = None,
    ) -> Iterator[PdfLease]:
        """Yield a lease and release it even when downstream processing fails."""

        current = self.acquire(source, ttl_seconds=ttl_seconds, filename=filename)
        try:
            yield current
        finally:
            self.release(current.lease_id)

    def release(self, lease_id: str) -> bool:
        """Delete exactly one staged lease; return whether it existed."""

        lease_dir = self._lease_dir(lease_id)
        with self._lock:
            if not lease_dir.is_dir():
                return False
            metadata_path = lease_dir / "lease.json"
            if not metadata_path.is_file():
                raise ValueError("refusing to remove an unrecognized lease directory")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("lease_id") != lease_id:
                raise ValueError("lease metadata does not match lease id")
            self._remove_lease_dir(lease_dir)
            return True

    def get(self, lease_id: str) -> PdfLease:
        """Resolve a lease without exposing arbitrary filesystem paths."""

        lease_dir = self._lease_dir(lease_id)
        metadata_path = lease_dir / "lease.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("PDF lease not found or already released")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("lease_id") != lease_id:
            raise ValueError("lease metadata does not match lease id")
        payload_path = lease_dir / str(metadata["filename"])
        if not payload_path.is_file():
            raise FileNotFoundError("PDF payload is missing")
        return PdfLease(
            lease_id=lease_id,
            path=payload_path,
            source=str(metadata.get("source", "")),
            created_at=float(metadata["created_at"]),
            expires_at=float(metadata["expires_at"]),
            filename=payload_path.name,
            size_bytes=payload_path.stat().st_size,
        )

    def cleanup_expired(self, *, now: float | None = None) -> int:
        """Remove expired leases and return the number removed."""

        current_time = time.time() if now is None else float(now)
        removed = 0
        with self._lock:
            for candidate in tuple(self.root.iterdir()):
                if not candidate.is_dir() or not _SAFE_LEASE_ID.fullmatch(candidate.name):
                    continue
                metadata_path = candidate / "lease.json"
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    if float(metadata["expires_at"]) <= current_time and self.release(candidate.name):
                        removed += 1
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    # A partial/interrupted lease is safe to leave for the next
                    # operator inspection rather than deleting unknown data.
                    continue
        return removed

    def _copy_local(self, source_path: Path, destination: Path) -> None:
        resolved = source_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"PDF source does not exist: {resolved}")
        if self.allowed_roots and not self._under_allowed_root(resolved):
            raise PermissionError(f"PDF source is outside allowed roots: {resolved}")
        if not self.allowed_roots:
            raise PermissionError("local PDF sources require configured allowed_roots")
        if resolved.stat().st_size > self.max_bytes:
            raise ValueError(f"PDF exceeds max_bytes={self.max_bytes}")
        shutil.copyfile(resolved, destination)

    def _download_url(self, source: str, destination: Path, parsed) -> None:
        if parsed.scheme != "https":
            raise PermissionError("only HTTPS PDF downloads are allowed")
        host = (parsed.hostname or "").lower()
        if not self.allowed_url_hosts or host not in self.allowed_url_hosts:
            raise PermissionError(f"PDF URL host is not allowlisted: {host}")
        request = Request(source, headers={"User-Agent": "em-knowledge-answer-assistant/1.0"})
        total = 0
        with urlopen(request, timeout=self.request_timeout_seconds) as response, destination.open("wb") as handle:
            declared_length = response.headers.get("Content-Length")
            if declared_length and int(declared_length) > self.max_bytes:
                raise ValueError(f"PDF exceeds max_bytes={self.max_bytes}")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.max_bytes:
                    raise ValueError(f"PDF exceeds max_bytes={self.max_bytes}")
                handle.write(chunk)

    @staticmethod
    def _validate_pdf(path: Path) -> None:
        if path.suffix.lower() != ".pdf":
            raise ValueError("PDF supplement must use a .pdf filename")
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise ValueError("staged file is not recognized as a PDF")

    def _lease_dir(self, lease_id: str) -> Path:
        if not _SAFE_LEASE_ID.fullmatch(str(lease_id)):
            raise ValueError("invalid lease id")
        candidate = (self.root / str(lease_id)).resolve()
        if self.root not in candidate.parents:
            raise ValueError("lease path escaped manager root")
        return candidate

    def _under_allowed_root(self, path: Path) -> bool:
        return any(root == path or root in path.parents for root in self.allowed_roots)

    @staticmethod
    def _safe_filename(value: str) -> str:
        candidate = Path(value).name
        candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
        if not candidate:
            candidate = "supplement.pdf"
        if not candidate.lower().endswith(".pdf"):
            candidate += ".pdf"
        return candidate

    @staticmethod
    def _name_from_source(source: str) -> str:
        parsed = urlparse(source)
        if parsed.scheme in {"http", "https", "file"}:
            return Path(unquote(parsed.path)).name or "supplement.pdf"
        return Path(source).name or "supplement.pdf"

    def _remove_lease_dir(self, lease_dir: Path) -> None:
        resolved = lease_dir.resolve()
        if self.root not in resolved.parents or resolved == self.root:
            raise ValueError("refusing to remove outside lease root")
        shutil.rmtree(resolved)
