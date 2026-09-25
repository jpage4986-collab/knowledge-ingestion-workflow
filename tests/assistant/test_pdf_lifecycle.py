from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fdtd_assistant.pdf_lifecycle import PdfLeaseManager


class PdfLifecycleTests(unittest.TestCase):
    def test_api_downloads_and_releases_pdf_lease(self) -> None:
        try:
            from fastapi.testclient import TestClient
            from fdtd_assistant.api import app, configure_pdf_manager
        except ImportError:
            self.skipTest("API extra is not installed")
        if app is None:
            self.skipTest("API extra is not installed")

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.pdf"
            source.write_bytes(b"%PDF-1.7\napi\n")
            manager = PdfLeaseManager(root=base / "leases", allowed_roots=[base])
            configure_pdf_manager(manager)
            try:
                client = TestClient(app)
                created = client.post("/pdf/lease", json={"source": str(source), "ttl_seconds": 60})
                self.assertEqual(created.status_code, 200)
                lease_id = created.json()["lease_id"]
                downloaded = client.get(f"/pdf/lease/{lease_id}")
                self.assertEqual(downloaded.status_code, 200)
                self.assertEqual(downloaded.content, source.read_bytes())
                released = client.delete(f"/pdf/lease/{lease_id}")
                self.assertEqual(released.status_code, 200)
                self.assertFalse(released.json()["released"])
                self.assertFalse((manager.root / lease_id).exists())
            finally:
                configure_pdf_manager(PdfLeaseManager())

    def test_context_manager_deletes_staged_copy_but_not_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.pdf"
            source.write_bytes(b"%PDF-1.7\nsource\n")
            manager = PdfLeaseManager(root=base / "leases", allowed_roots=[base])

            with manager.lease(source, ttl_seconds=60) as lease:
                self.assertTrue(lease.path.is_file())
                self.assertNotEqual(lease.path, source)
                self.assertEqual(lease.path.read_bytes(), source.read_bytes())
            self.assertFalse(lease.path.exists())
            self.assertTrue(source.exists())

    def test_release_is_idempotent_and_never_accepts_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.pdf"
            source.write_bytes(b"%PDF-1.7\nsource\n")
            manager = PdfLeaseManager(root=base / "leases", allowed_roots=[base])
            lease = manager.acquire(source)
            self.assertTrue(manager.release(lease.lease_id))
            self.assertFalse(manager.release(lease.lease_id))
            with self.assertRaises(ValueError):
                manager.release("../source.pdf")

    def test_expired_cleanup_removes_only_expired_lease(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / "source.pdf"
            source.write_bytes(b"%PDF-1.7\nsource\n")
            manager = PdfLeaseManager(root=base / "leases", allowed_roots=[base])
            expired = manager.acquire(source, ttl_seconds=30)
            live = manager.acquire(source, ttl_seconds=300)
            removed = manager.cleanup_expired(now=expired.expires_at + 1)
            self.assertEqual(removed, 1)
            self.assertFalse(expired.path.exists())
            self.assertTrue(live.path.exists())
            manager.release(live.lease_id)

    def test_local_sources_are_deny_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.pdf"
            source.write_bytes(b"%PDF-1.7\nsource\n")
            manager = PdfLeaseManager(root=Path(temp) / "leases")
            with self.assertRaises(PermissionError):
                manager.acquire(source)


if __name__ == "__main__":
    unittest.main()
