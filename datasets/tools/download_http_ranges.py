"""Download a large HTTP file with resumable parallel byte ranges."""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def remote_size(url: str, timeout: int) -> int:
    headers = {"Accept-Encoding": "identity"}
    response = requests.head(
        url,
        headers=headers,
        allow_redirects=True,
        timeout=timeout,
    )
    response.raise_for_status()
    size = int(response.headers["content-length"])
    if response.headers.get("accept-ranges", "").lower() != "bytes":
        probe = requests.get(
            url,
            headers={"Range": "bytes=0-0", **headers},
            allow_redirects=True,
            timeout=timeout,
        )
        if probe.status_code != 206 or "content-range" not in probe.headers:
            raise RuntimeError(f"Server does not support byte ranges: {url}")
    return size


def download_part(
    url: str,
    path: Path,
    start: int,
    end: int,
    timeout: int,
    retries: int,
) -> tuple[int, int]:
    expected = end - start + 1
    if path.is_file() and path.stat().st_size == expected:
        return start, expected

    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(
                url,
                headers={
                    "Range": f"bytes={start}-{end}",
                    "Accept-Encoding": "identity",
                },
                allow_redirects=True,
                stream=True,
                timeout=timeout,
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Expected HTTP 206 for bytes={start}-{end}, "
                        f"got {response.status_code}"
                    )
                with temporary.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            handle.write(block)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"Range bytes={start}-{end} has {temporary.stat().st_size} "
                    f"bytes, expected {expected}"
                )
            temporary.replace(path)
            return start, expected
        except Exception:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 10))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    size = remote_size(args.url, args.timeout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file() and args.output.stat().st_size == size:
        print(f"Already complete: {args.output} ({size} bytes)", flush=True)
        return
    if args.output.exists():
        raise FileExistsError(
            f"Incomplete output already exists: {args.output}. Move it aside first."
        )

    part_dir = args.output.with_name(args.output.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = args.chunk_mib * 1024 * 1024
    ranges = [
        (start, min(start + chunk - 1, size - 1))
        for start in range(0, size, chunk)
    ]
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_part,
                args.url,
                part_dir / f"{index:06d}.part",
                start,
                end,
                args.timeout,
                args.retries,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        for future in as_completed(futures):
            _, count = future.result()
            completed += count
            print(
                f"Downloaded {completed / 1024**2:.1f}/{size / 1024**2:.1f} MiB",
                flush=True,
            )

    with args.output.open("wb") as destination:
        for index in range(len(ranges)):
            with (part_dir / f"{index:06d}.part").open("rb") as source:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
    if args.output.stat().st_size != size:
        raise RuntimeError(f"Assembled size mismatch: {args.output}")
    print(f"Saved {args.output} ({size} bytes)", flush=True)


if __name__ == "__main__":
    main()
