"""List or extract selected entries from a single HTTP ZIP using positive ranges."""

from __future__ import annotations

import argparse
import binascii
import json
import shutil
import struct
import subprocess
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import requests


EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = 0x02014B50
LOCAL_SIGNATURE = 0x04034B50
RANGE_CHUNK_SIZE = 128 * 1024
CURL_RANGE_CHUNK_SIZE = 32 * 1024


@dataclass(frozen=True)
class Entry:
    name: str
    offset: int
    compression: int
    compressed_size: int
    size: int
    crc32: int


def request_range(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
) -> bytes:
    expected = end - start + 1
    error = None
    for attempt in range(5):
        try:
            response = session.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                allow_redirects=True,
                timeout=120,
            )
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(f"Expected HTTP 206, got {response.status_code}")
            if len(response.content) != expected:
                raise RuntimeError(
                    f"Range length mismatch {len(response.content)} != {expected}"
                )
            return response.content
        except (requests.RequestException, RuntimeError) as exc:
            error = exc
            if attempt == 4:
                break
            time.sleep(2**attempt)
    raise RuntimeError(f"Range request failed after 5 attempts: {start}-{end}") from error


def request_span(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
) -> bytes:
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + RANGE_CHUNK_SIZE - 1, end)
        chunks.append(request_range(session, url, cursor, chunk_end))
        cursor = chunk_end + 1
    return b"".join(chunks)


def request_range_curl(url: str, start: int, end: int) -> bytes:
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if executable is None:
        raise FileNotFoundError("curl executable not found")
    expected = end - start + 1
    error = None
    for attempt in range(5):
        result = subprocess.run(
            [
                executable,
                "--silent",
                "--show-error",
                "--fail",
                "--noproxy",
                "*",
                "--range",
                f"{start}-{end}",
                "--max-time",
                "60",
                "--location",
                url,
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0 and len(result.stdout) == expected:
            return result.stdout
        message = result.stderr.decode(errors="replace").strip()
        error = RuntimeError(
            f"curl returned code {result.returncode}, "
            f"length {len(result.stdout)} != {expected}: {message}"
        )
        if attempt < 4:
            time.sleep(2**attempt)
    raise RuntimeError(f"curl range request failed after 5 attempts: {start}-{end}") from error


def request_span_curl(url: str, start: int, end: int) -> bytes:
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + CURL_RANGE_CHUNK_SIZE - 1, end)
        chunks.append(request_range_curl(url, cursor, chunk_end))
        cursor = chunk_end + 1
    return b"".join(chunks)


def archive_size(session: requests.Session, url: str) -> int:
    error = None
    for attempt in range(5):
        response = None
        try:
            response = session.get(
                url,
                headers={"Range": "bytes=0-0"},
                allow_redirects=True,
                timeout=120,
                stream=True,
            )
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            if response.status_code != 206 or "/" not in content_range:
                raise RuntimeError("Server did not return a usable Content-Range")
            return int(content_range.rsplit("/", 1)[1])
        except (requests.RequestException, RuntimeError) as exc:
            error = exc
            if attempt == 4:
                break
            time.sleep(2**attempt)
        finally:
            if response is not None:
                response.close()
    raise RuntimeError("Archive-size request failed after 5 attempts") from error


def read_entries(session: requests.Session, url: str) -> list[Entry]:
    size = archive_size(session, url)
    tail_size = min(size, 65557)
    tail_start = size - tail_size
    tail = request_span(session, url, tail_start, size - 1)
    eocd_at = tail.rfind(EOCD_SIGNATURE)
    if eocd_at < 0:
        raise ValueError("End-of-central-directory record not found")

    fields = struct.unpack_from("<IHHHHIIH", tail, eocd_at)
    _, disk, central_disk, disk_entries, total, central_size, central_offset, _ = fields
    if disk != 0 or central_disk != 0 or disk_entries != total:
        raise ValueError("Split ZIP archives are not supported")
    if total == 0xFFFF or central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        raise ValueError("ZIP64 central directories are not supported")

    central = request_span(
        session,
        url,
        central_offset,
        central_offset + central_size - 1,
    )
    return parse_entries(central)


def parse_entries(central: bytes) -> list[Entry]:
    entries = []
    cursor = 0
    central_size = len(central)
    while cursor < central_size:
        header = struct.unpack_from("<IHHHHHHIIIHHHHHII", central, cursor)
        if header[0] != CENTRAL_SIGNATURE:
            raise ValueError(f"Invalid central-directory signature at {cursor}")
        (
            _, _, _, flags, compression, _, _, crc32, compressed_size, raw_size,
            name_length, extra_length, comment_length, disk_number, _, _, offset,
        ) = header
        if disk_number != 0:
            raise ValueError("Split ZIP entry encountered")
        start = cursor + 46
        raw_name = central[start:start + name_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = raw_name.decode(encoding)
        entries.append(
            Entry(name, offset, compression, compressed_size, raw_size, crc32)
        )
        cursor = start + name_length + extra_length + comment_length
    if cursor != central_size:
        raise ValueError(f"Central-directory size mismatch {cursor} != {central_size}")
    return entries


def safe_output(output_root: Path, name: str, strip_components: int) -> Path:
    parts = PurePosixPath(name).parts[strip_components:]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe or empty archive path {name}")
    output = output_root.joinpath(*parts)
    resolved_root = output_root.resolve()
    resolved_output = output.resolve()
    if resolved_root != resolved_output and resolved_root not in resolved_output.parents:
        raise ValueError(f"Archive path escapes output root {name}")
    return output


def extract_one(
    entry: Entry,
    url: str,
    output_root: Path,
    strip_components: int,
    curl_direct: bool,
) -> str:
    session = None if curl_direct else requests.Session()
    request = (
        lambda start, end: request_span_curl(url, start, end)
        if curl_direct
        else request_span(session, url, start, end)
    )
    header = request(entry.offset, entry.offset + 29)
    local = struct.unpack("<IHHHHHIIIHH", header)
    if local[0] != LOCAL_SIGNATURE:
        raise ValueError(f"Invalid local header for {entry.name}")
    name_length, extra_length = local[-2:]
    start = entry.offset + 30 + name_length + extra_length
    payload = request(
        start,
        start + entry.compressed_size - 1,
    )
    if entry.compression == 0:
        data = payload
    elif entry.compression == 8:
        data = zlib.decompress(payload, -15)
    else:
        raise ValueError(
            f"Unsupported ZIP method {entry.compression} for {entry.name}"
        )
    crc = binascii.crc32(data) & 0xFFFFFFFF
    if len(data) != entry.size or crc != entry.crc32:
        raise ValueError(f"Size or CRC mismatch for {entry.name}")

    output = safe_output(output_root, entry.name, strip_components)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return entry.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", type=Path)
    parser.add_argument("archive")
    parser.add_argument("output", type=Path)
    parser.add_argument("--entry", action="append", default=[])
    parser.add_argument("--prefix", default="")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--central-file", type=Path)
    parser.add_argument("--show-offsets", action="store_true")
    parser.add_argument("--curl-direct", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--strip-components", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    record = next(
        (item for item in metadata if item["filename"] == args.archive),
        None,
    )
    if record is None:
        raise KeyError(f"Archive not found in metadata {args.archive}")
    url = record["cdn"]

    session = requests.Session()
    entries = (
        parse_entries(args.central_file.read_bytes())
        if args.central_file is not None
        else read_entries(session, url)
    )
    print(f"Found {len(entries)} entries in {args.archive}")

    if args.list_only:
        selected = [entry for entry in entries if entry.name.startswith(args.prefix)]
        for entry in selected[:args.limit]:
            fields = [entry.name, str(entry.size)]
            if args.show_offsets:
                fields.extend(
                    [str(entry.offset), str(entry.compressed_size), str(entry.compression)]
                )
            print("\t".join(fields))
        print(f"Matched {len(selected)} entries")
        return

    catalog = {entry.name: entry for entry in entries}
    selected = []
    if args.entry:
        for name in args.entry:
            if name not in catalog:
                raise KeyError(f"ZIP entry not found {name}")
            selected.append(catalog[name])
    elif args.prefix:
        selected = [entry for entry in entries if entry.name.startswith(args.prefix)]
    else:
        raise ValueError("Provide --entry or --prefix")
    selected = [entry for entry in selected if not entry.name.endswith("/")]
    print(
        f"Extracting {len(selected)} files, "
        f"{sum(entry.size for entry in selected) / 2**20:.1f} MiB uncompressed"
    )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                extract_one,
                entry,
                url,
                args.output,
                args.strip_components,
                args.curl_direct,
            )
            for entry in selected
        ]
        for future in as_completed(futures):
            print(f"Extracted {future.result()}", flush=True)


if __name__ == "__main__":
    main()
