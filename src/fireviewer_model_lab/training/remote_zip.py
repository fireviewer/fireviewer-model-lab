"""Read selected ZIP members over HTTP Range requests without downloading archives."""

from __future__ import annotations

import binascii
import struct
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass

EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


def require_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("remote ZIP URL must use HTTP or HTTPS with a hostname")
    return url


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    compressed_size: int
    uncompressed_size: int
    compression_method: int
    crc32: int
    local_header_offset: int


class RemoteZip:
    def __init__(self, url: str, *, timeout_seconds: float = 120.0) -> None:
        self.url = require_http_url(url)
        self.timeout_seconds = timeout_seconds
        request = urllib.request.Request(self.url, method="HEAD")  # noqa: S310
        with urllib.request.urlopen(  # noqa: S310 - URL validated above
            request, timeout=timeout_seconds
        ) as response:
            self.size = int(response.headers["Content-Length"])
            if response.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise ValueError(f"remote ZIP does not advertise byte ranges: {url}")

    def _range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.size:
            raise ValueError(f"invalid HTTP range {start}-{end} for {self.size}")
        request = urllib.request.Request(  # noqa: S310 - URL validated at construction
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "FireViewer/1.0"},
        )
        with urllib.request.urlopen(  # noqa: S310 - URL validated at construction
            request, timeout=self.timeout_seconds
        ) as response:
            payload = response.read()
            if int(response.status) != 206:
                raise OSError(f"remote server ignored ZIP byte range: {self.url}")
        expected = end - start + 1
        if len(payload) != expected:
            raise OSError(f"incomplete ZIP byte range: {len(payload)} != {expected}")
        return payload

    def _central_location(self) -> tuple[int, int]:
        tail_size = min(self.size, 1024 * 1024)
        tail_start = self.size - tail_size
        tail = self._range(tail_start, self.size - 1)
        eocd_position = tail.rfind(EOCD_SIGNATURE)
        if eocd_position < 0:
            raise ValueError("remote ZIP end-of-central-directory record not found")
        eocd = struct.unpack_from("<4s4H2LH", tail, eocd_position)
        central_size, central_offset = int(eocd[5]), int(eocd[6])
        if central_offset != 0xFFFFFFFF and central_size != 0xFFFFFFFF:
            return central_offset, central_size

        locator_position = tail.rfind(ZIP64_LOCATOR_SIGNATURE, 0, eocd_position)
        if locator_position < 0:
            raise ValueError("remote ZIP64 locator not found")
        locator = struct.unpack_from("<4sLQL", tail, locator_position)
        zip64_offset = int(locator[2])
        zip64 = self._range(zip64_offset, zip64_offset + 55)
        record = struct.unpack_from("<4sQ2H2L4Q", zip64, 0)
        if record[0] != ZIP64_EOCD_SIGNATURE:
            raise ValueError("invalid remote ZIP64 record")
        return int(record[9]), int(record[8])

    @staticmethod
    def _zip64_values(
        extra: bytes,
        *,
        uncompressed_size: int,
        compressed_size: int,
        local_header_offset: int,
    ) -> tuple[int, int, int]:
        position = 0
        values: list[int] | None = None
        while position + 4 <= len(extra):
            field_id, field_size = struct.unpack_from("<HH", extra, position)
            position += 4
            field = extra[position : position + field_size]
            position += field_size
            if field_id == 0x0001:
                values = [
                    struct.unpack_from("<Q", field, offset)[0]
                    for offset in range(0, len(field) - 7, 8)
                ]
                break
        required = sum(
            value == 0xFFFFFFFF
            for value in (uncompressed_size, compressed_size, local_header_offset)
        )
        if required and (values is None or len(values) < required):
            raise ValueError("incomplete ZIP64 extended information")
        iterator = iter(values or [])
        if uncompressed_size == 0xFFFFFFFF:
            uncompressed_size = int(next(iterator))
        if compressed_size == 0xFFFFFFFF:
            compressed_size = int(next(iterator))
        if local_header_offset == 0xFFFFFFFF:
            local_header_offset = int(next(iterator))
        return uncompressed_size, compressed_size, local_header_offset

    def entries(self) -> list[RemoteZipEntry]:
        central_offset, central_size = self._central_location()
        central = self._range(central_offset, central_offset + central_size - 1)
        entries: list[RemoteZipEntry] = []
        position = 0
        while position + 46 <= len(central):
            values = struct.unpack_from("<4s6H3L5H2L", central, position)
            if values[0] != CENTRAL_SIGNATURE:
                break
            flags = int(values[3])
            method = int(values[4])
            crc32 = int(values[7])
            compressed_size = int(values[8])
            uncompressed_size = int(values[9])
            name_length, extra_length, comment_length = map(int, values[10:13])
            local_offset = int(values[16])
            name_start = position + 46
            name_bytes = central[name_start : name_start + name_length]
            extra = central[name_start + name_length : name_start + name_length + extra_length]
            uncompressed_size, compressed_size, local_offset = self._zip64_values(
                extra,
                uncompressed_size=uncompressed_size,
                compressed_size=compressed_size,
                local_header_offset=local_offset,
            )
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = name_bytes.decode(encoding, errors="replace")
            if not name.endswith("/"):
                entries.append(
                    RemoteZipEntry(
                        name=name,
                        compressed_size=compressed_size,
                        uncompressed_size=uncompressed_size,
                        compression_method=method,
                        crc32=crc32,
                        local_header_offset=local_offset,
                    )
                )
            position += 46 + name_length + extra_length + comment_length
        if not entries:
            raise ValueError(f"remote ZIP central directory has no files: {self.url}")
        return entries

    def read(self, entry: RemoteZipEntry) -> bytes:
        header = self._range(entry.local_header_offset, entry.local_header_offset + 29)
        values = struct.unpack_from("<4s5H3L2H", header, 0)
        if values[0] != LOCAL_SIGNATURE:
            raise ValueError(f"invalid local ZIP header for {entry.name}")
        name_length, extra_length = int(values[9]), int(values[10])
        data_start = entry.local_header_offset + 30 + name_length + extra_length
        compressed = self._range(data_start, data_start + entry.compressed_size - 1)
        if entry.compression_method == 0:
            payload = compressed
        elif entry.compression_method == 8:
            payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
        else:
            raise ValueError(
                f"unsupported ZIP compression method {entry.compression_method}: {entry.name}"
            )
        if len(payload) != entry.uncompressed_size:
            raise OSError(f"remote ZIP member size mismatch: {entry.name}")
        if binascii.crc32(payload) & 0xFFFFFFFF != entry.crc32:
            raise OSError(f"remote ZIP member CRC mismatch: {entry.name}")
        return payload
