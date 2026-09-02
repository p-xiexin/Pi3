"""Extract selected files from an HTTP-hosted split ZIP archive."""

from __future__ import annotations

import argparse
import binascii
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import requests


EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = 0x02014B50
LOCAL_SIGNATURE = 0x04034B50


@dataclass(frozen=True)
class Entry:
    name: str
    disk: int
    offset: int
    compression: int
    compressed_size: int
    size: int
    crc32: int


def read_entries(descriptor: Path) -> dict[str, Entry]:
    with descriptor.open("rb") as handle:
        handle.seek(max(0, descriptor.stat().st_size - 65557))
        tail = handle.read()
        eocd_at = tail.rfind(EOCD_SIGNATURE)
        if eocd_at < 0:
            raise ValueError(f"End-of-central-directory record not found in {descriptor}")
        eocd = struct.unpack_from("<IHHHHIIH", tail, eocd_at)
        _, _, central_disk, _, total, central_size, central_offset, _ = eocd
        handle.seek(central_offset)
        central = handle.read(central_size)

    entries: dict[str, Entry] = {}
    cursor = 0
    while cursor < central_size:
        fields = struct.unpack_from("<IHHHHHHIIIHHHHHII", central, cursor)
        if fields[0] != CENTRAL_SIGNATURE:
            raise ValueError(f"Invalid central-directory signature at {cursor}")
        (
            _, _, _, _, compression, _, _, crc32, compressed_size, size,
            name_length, extra_length, comment_length, disk, _, _, offset,
        ) = fields
        start = cursor + 46
        name = central[start:start + name_length].decode("utf-8")
        entries[name] = Entry(
            name=name,
            disk=disk,
            offset=offset,
            compression=compression,
            compressed_size=compressed_size,
            size=size,
            crc32=crc32,
        )
        cursor = start + name_length + extra_length + comment_length
    if cursor != central_size:
        raise ValueError(
            f"Central-directory size mismatch {cursor} != {central_size} "
            f"on disk {central_disk + 1}"
        )
    return entries


def request_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    response = session.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        allow_redirects=True,
        timeout=120,
    )
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Expected HTTP 206 from {url}, got {response.status_code}")
    return response.content


def extract_one(
    entry: Entry,
    urls: dict[int, str],
    output_root: Path,
    strip_components: int,
) -> str:
    session = requests.Session()
    url = urls[entry.disk]
    header = request_range(session, url, entry.offset, entry.offset + 29)
    local = struct.unpack("<IHHHHHIIIHH", header)
    if local[0] != LOCAL_SIGNATURE:
        raise ValueError(f"Invalid local header for {entry.name}")
    name_length, extra_length = local[-2:]
    start = entry.offset + 30 + name_length + extra_length
    payload = request_range(
        session,
        url,
        start,
        start + entry.compressed_size - 1,
    )
    if entry.compression == 0:
        data = payload
    elif entry.compression == 8:
        data = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"Unsupported ZIP method {entry.compression} for {entry.name}")
    if len(data) != entry.size or (binascii.crc32(data) & 0xFFFFFFFF) != entry.crc32:
        raise ValueError(f"Size or CRC mismatch for {entry.name}")

    parts = PurePosixPath(entry.name).parts[strip_components:]
    if not parts:
        raise ValueError(f"strip_components removes full path {entry.name}")
    output = output_root.joinpath(*parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    return entry.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("base_url")
    parser.add_argument("output", type=Path)
    parser.add_argument("entries", nargs="+")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--strip-components", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    catalog = read_entries(args.descriptor)
    selected = []
    for name in args.entries:
        if name not in catalog:
            raise KeyError(f"ZIP entry not found: {name}")
        selected.append(catalog[name])

    disks = sorted({entry.disk for entry in selected})
    urls = {
        disk: (
            f"{args.base_url}/{args.stem}.zip"
            if disk == 15
            else f"{args.base_url}/{args.stem}.z{disk + 1:02d}"
        )
        for disk in disks
    }
    print(f"Selected {len(selected)} entries from disks {[disk + 1 for disk in disks]}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                extract_one,
                entry,
                urls,
                args.output,
                args.strip_components,
            )
            for entry in selected
        ]
        for future in as_completed(futures):
            print(f"Extracted {future.result()}", flush=True)


if __name__ == "__main__":
    main()
