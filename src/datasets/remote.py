"""Remote-file helpers: public Google Drive folder listing/download and a seekable HTTP range reader.

The range reader lets `zipfile` read only the members we need from a very large remote archive (UA-DETRAC images are a
single ~9.8 GB zip; the disk budget does not allow downloading it whole). Every download is checksummed so it can be
recorded in the evidence manifest.
"""

from __future__ import annotations

import hashlib
import html
import io
import logging
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

USER_AGENT = "dsci41-research-client/0.1"
DRIVE_FOLDER_URL = "https://drive.google.com/embeddedfolderview?id={folder_id}"
DRIVE_DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"

_ENTRY_RE = re.compile(r'id="entry-(?P<id>[A-Za-z0-9_-]+)".*?flip-entry-title">(?P<title>[^<]*)<', re.DOTALL)


@dataclass(frozen=True)
class DriveEntry:
    file_id: str
    name: str

    @property
    def download_url(self) -> str:
        return DRIVE_DOWNLOAD_URL.format(file_id=self.file_id)


def _open(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0):  # noqa: ANN202
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - fixed https hosts


def parse_drive_folder_html(page: str) -> list[DriveEntry]:
    """Parse the public `embeddedfolderview` page into (file id, name) entries."""
    return [DriveEntry(m.group("id"), html.unescape(m.group("title")).strip()) for m in _ENTRY_RE.finditer(page)]


def list_drive_folder(folder_id: str) -> list[DriveEntry]:
    with _open(DRIVE_FOLDER_URL.format(folder_id=folder_id)) as response:
        page = response.read().decode("utf-8", errors="replace")
    entries = parse_drive_folder_html(page)
    if not entries:
        raise RuntimeError(f"no entries found in Drive folder {folder_id} (private, empty or page format changed)")
    return entries


def download_file(
    url: str, destination: Path, expected_size: int | None = None, retries: int = 3, chunk_size: int = 1 << 20
) -> str:
    """Stream `url` to `destination` atomically (via a .part file) and return its SHA-256."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        digest = hashlib.sha256()
        try:
            with _open(url, timeout=120) as response, tmp.open("wb") as handle:
                content_type = response.headers.get("Content-Type", "")
                if content_type.startswith("text/html"):
                    raise RuntimeError(f"expected binary content, got HTML from {url} (quota or permission page?)")
                while chunk := response.read(chunk_size):
                    handle.write(chunk)
                    digest.update(chunk)
            size = tmp.stat().st_size
            if expected_size is not None and size != expected_size:
                raise RuntimeError(f"size mismatch for {url}: {size} != {expected_size}")
            tmp.replace(destination)
            return digest.hexdigest()
        except (OSError, RuntimeError) as exc:
            logger.warning("download attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(2.0 * attempt)
    raise AssertionError("unreachable")


RangeFetcher = Callable[[int, int], bytes]  # (start, end_inclusive) -> bytes


def http_range_fetcher(url: str) -> tuple[RangeFetcher, int]:
    """Return a fetcher for byte ranges of `url` and the total size (server must answer 206)."""
    with _open(url, headers={"Range": "bytes=0-0"}) as response:
        if response.status != 206:
            raise RuntimeError(f"server does not support range requests for {url} (status {response.status})")
        content_range = response.headers["Content-Range"]  # e.g. "bytes 0-0/9849591986"
        total = int(content_range.rsplit("/", 1)[1])

    def fetch(start: int, end: int) -> bytes:
        for attempt in range(1, 4):
            try:
                with _open(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120) as resp:
                    data = resp.read()
                if len(data) != end - start + 1:
                    raise RuntimeError(f"short range read {len(data)} != {end - start + 1}")
                return data
            except (OSError, RuntimeError) as exc:
                if attempt == 3:
                    raise
                logger.warning("range fetch %d-%d failed (%s); retrying", start, end, exc)
                time.sleep(2.0 * attempt)
        raise AssertionError("unreachable")

    return fetch, total


class RangeFile(io.RawIOBase):
    """Read-only, seekable file backed by a range fetcher, with a single read-ahead block cache."""

    def __init__(self, fetch: RangeFetcher, size: int, block_size: int = 4 << 20) -> None:
        super().__init__()
        self._fetch = fetch
        self._size = size
        self._pos = 0
        self._block_size = block_size
        self._cache_start = -1
        self._cache = b""
        self.bytes_fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self._size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = new
        return new

    def _load_block(self, position: int, minimum: int) -> None:
        length = max(self._block_size, minimum)
        end = min(position + length, self._size) - 1
        self._cache = self._fetch(position, end)
        self._cache_start = position
        self.bytes_fetched += len(self._cache)

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        cache_end = self._cache_start + len(self._cache)
        if not (self._cache_start <= self._pos and self._pos + size <= cache_end):
            self._load_block(self._pos, size)
        offset = self._pos - self._cache_start
        data = self._cache[offset : offset + size]
        self._pos += len(data)
        return data

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
