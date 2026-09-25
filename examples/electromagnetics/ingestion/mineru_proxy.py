"""Local serialized gateway for the MinerU API used by RAGFlow.

RAGFlow submits page ranges as separate synchronous /file_parse requests.
This gateway keeps those requests serialized, waits for the upstream MinerU
task slot to be free, and preserves the upstream response body for diagnosis.
It is intentionally local-only and has no external network destinations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


UPSTREAM = os.environ.get("MINERU_PROXY_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
HOST = os.environ.get("MINERU_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("MINERU_PROXY_PORT", "8001"))
LOG_PATH = Path(os.environ.get("MINERU_PROXY_LOG", "mineru_proxy.log"))
MAX_RETRIES = int(os.environ.get("MINERU_PROXY_MAX_RETRIES", "3"))
RETRY_SECONDS = float(os.environ.get("MINERU_PROXY_RETRY_SECONDS", "8"))
SLOT_WAIT_SECONDS = float(os.environ.get("MINERU_PROXY_SLOT_WAIT_SECONDS", "5"))
SLOT_WAIT_LIMIT = int(os.environ.get("MINERU_PROXY_SLOT_WAIT_LIMIT", "120"))

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("mineru-proxy")

app = FastAPI(title="Local MinerU serialized proxy", version="1.0")
request_lock = asyncio.Lock()


def _headers(request: Request) -> dict[str, str]:
    blocked = {"host", "content-length", "connection", "transfer-encoding"}
    return {k: v for k, v in request.headers.items() if k.lower() not in blocked}


async def _upstream_health(client: httpx.AsyncClient) -> dict | None:
    try:
        response = await client.get(f"{UPSTREAM}/health", timeout=10)
        if response.status_code != 200:
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception as exc:  # pragma: no cover - diagnostic path
        logger.warning("health check failed: %s", exc)
        return None


async def _wait_for_slot(client: httpx.AsyncClient) -> None:
    for _ in range(SLOT_WAIT_LIMIT):
        health = await _upstream_health(client)
        processing = int((health or {}).get("processing_tasks", 0))
        queued = int((health or {}).get("queued_tasks", 0))
        if health is not None and processing == 0 and queued == 0:
            return
        await asyncio.sleep(SLOT_WAIT_SECONDS)
    logger.warning("upstream slot remained busy after %ss; forwarding anyway", SLOT_WAIT_LIMIT * SLOT_WAIT_SECONDS)


@app.get("/health")
async def health() -> JSONResponse:
    async with httpx.AsyncClient() as client:
        upstream = await _upstream_health(client)
    payload = {
        "status": "healthy" if upstream else "degraded",
        "proxy": {"upstream": UPSTREAM, "port": PORT, "serialized": True},
        "upstream": upstream,
    }
    return JSONResponse(payload, status_code=200 if upstream else 503)


@app.get("/openapi.json")
async def openapi_proxy() -> Response:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{UPSTREAM}/openapi.json", timeout=30)
    return Response(content=response.content, status_code=response.status_code, media_type="application/json")


@app.post("/file_parse")
async def file_parse(request: Request) -> Response:
    body = await request.body()
    headers = _headers(request)
    async with request_lock:
        started = time.monotonic()
        async with httpx.AsyncClient(timeout=httpx.Timeout(1900.0, connect=30.0)) as client:
            await _wait_for_slot(client)
            last_status = 502
            last_body = b""
            last_headers: dict[str, str] = {}
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    response = await client.post(
                        f"{UPSTREAM}/file_parse",
                        content=body,
                        headers=headers,
                    )
                    last_status = response.status_code
                    last_body = response.content
                    last_headers = dict(response.headers)
                    if response.status_code != 409:
                        break
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text[:2000]
                    logger.warning("upstream 409 attempt=%s detail=%s", attempt, json.dumps(detail, ensure_ascii=False))
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_SECONDS)
                except Exception as exc:  # pragma: no cover - network diagnostic path
                    last_status = 502
                    last_body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
                    logger.exception("upstream request failed attempt=%s", attempt)
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_SECONDS)
            elapsed = time.monotonic() - started
            logger.info("file_parse status=%s bytes=%s elapsed=%.1fs", last_status, len(body), elapsed)
            response_headers = {
                k: v
                for k, v in last_headers.items()
                if k.lower() not in {"content-length", "transfer-encoding", "connection"}
            }
            return Response(content=last_body, status_code=last_status, headers=response_headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_config=None)
