"""Execution engine shared by the dataset command-line interface."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < 1024 or unit == "PiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PiB"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def proxy_environment(proxy: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if proxy:
        normalized = proxy if "://" in proxy else f"http://{proxy}"
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[key] = normalized
    return env


def download_file(
    url: str,
    output: Path,
    proxy: str | None,
    workers: int,
    dry_run: bool,
) -> None:
    if output.exists() and output.stat().st_size > 0:
        print(f"reuse archive  {output}")
        return
    print(f"download       {url}")
    print(f"destination    {output}")
    if dry_run:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TOOLS_ROOT / "download_http_ranges.py"),
        url,
        str(output),
        "--workers",
        str(workers),
    ]
    subprocess.run(command, check=True, env=proxy_environment(proxy))


def _safe_member(output: Path, member_name: str, strip_components: int) -> Path | None:
    parts = PurePosixPath(member_name.replace("\\", "/")).parts[strip_components:]
    if not parts:
        return None
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe archive path {member_name}")
    destination = output.joinpath(*parts).resolve()
    root = output.resolve()
    if root != destination and root not in destination.parents:
        raise ValueError(f"Archive path escapes output root {member_name}")
    return destination


def extract_archive(
    archive: Path,
    output: Path,
    strip_components: int = 0,
    dry_run: bool = False,
) -> None:
    print(f"extract        {archive}")
    print(f"into           {output}")
    if dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                destination = _safe_member(output, member.filename, strip_components)
                if destination is None:
                    continue
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                destination = _safe_member(output, member.name, strip_components)
                if destination is None or member.isdir():
                    if destination is not None:
                        destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                source = handle.extractfile(member)
                if source is None:
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        return
    raise ValueError(f"Unsupported archive format {archive}")


def run_command(command: list[str], proxy: str | None, dry_run: bool) -> None:
    print("command        " + subprocess.list2cmdline(command), flush=True)
    if dry_run:
        return
    if len(command) >= 4 and command[:2] == ["git", "clone"]:
        target = Path(command[-1])
        if target.exists():
            print(f"reuse checkout {target}")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=proxy_environment(proxy))


def _format_command(parts: Iterable[str], values: dict[str, str]) -> list[str]:
    return [part.format(**values) for part in parts]


def execute_recipe(
    dataset_id: str,
    profile: dict[str, Any],
    output: Path,
    cache: Path,
    input_path: Path | None,
    proxy: str | None,
    workers: int,
    dry_run: bool,
) -> Path:
    recipe = profile["recipe"]
    recipe_type = recipe["type"]
    if not dry_run:
        cache.mkdir(parents=True, exist_ok=True)
    for file_record in recipe.get("files", []):
        destination = output / file_record.get("path", file_record["filename"])
        download_file(file_record["url"], destination, proxy, workers, dry_run)
    if recipe_type == "archives":
        for artifact in recipe["artifacts"]:
            archive = cache / artifact["filename"]
            download_file(artifact["url"], archive, proxy, workers, dry_run)
            target = output / artifact.get("extract_to", "")
            extract_archive(archive, target, artifact.get("strip_components", 0), dry_run)
        return output
    if recipe_type == "remote-zip":
        import json

        metadata = cache / "remote_zip_urls.json"
        records = [
            {"filename": artifact["filename"], "cdn": artifact["url"]}
            for artifact in recipe["artifacts"]
        ]
        if not dry_run:
            metadata.write_text(json.dumps(records, indent=2), encoding="utf-8")
        for artifact in recipe["artifacts"]:
            artifact_output = output / artifact.get("extract_to", "")
            command = [
                sys.executable,
                str(TOOLS_ROOT / "extract_remote_zip_entries.py"),
                str(metadata),
                artifact["filename"],
                str(artifact_output),
                "--workers",
                str(workers),
                "--strip-components",
                str(artifact.get("strip_components", 0)),
            ]
            for entry in artifact["entries"]:
                command.extend(["--entry", entry])
            run_command(command, proxy, dry_run)
        return output
    if recipe_type == "split-zip-sample":
        descriptor = cache / recipe["descriptor_filename"]
        download_file(recipe["descriptor_url"], descriptor, proxy, workers, dry_run)
        command = [
            sys.executable,
            str(TOOLS_ROOT / "extract_split_zip_entries.py"),
            str(descriptor),
            recipe["base_url"],
            str(output),
            *recipe["entries"],
            "--stem",
            recipe["stem"],
            "--strip-components",
            str(recipe.get("strip_components", 0)),
            "--workers",
            str(workers),
        ]
        run_command(command, proxy, dry_run)
        return output
    if recipe_type == "waymo":
        if input_path is None:
            raise ValueError("Waymo needs --input pointing to one TFRecord or a TFRecord directory")
        records = [input_path] if input_path.is_file() else sorted(input_path.glob("*.tfrecord*"))
        if not records:
            raise FileNotFoundError(f"No TFRecord files found under {input_path}")
        limit = 1 if recipe.get("one_record") else None
        for record in records[:limit]:
            command = [
                sys.executable,
                str(TOOLS_ROOT / "prepare_waymo_sample.py"),
                str(record),
                str(output),
            ]
            if recipe.get("max_frames"):
                command.extend(["--max-frames", str(recipe["max_frames"])])
            run_command(command, proxy, dry_run)
        return output
    if recipe_type == "official-command":
        values = {
            "output": str(output),
            "cache": str(cache),
            "python": sys.executable,
            "input": str(input_path) if input_path else "",
            "workers": str(workers),
        }
        if input_path is not None and not recipe.get("commands"):
            print(f"using input    {input_path}")
            return input_path
        print("official tool  this recipe follows the upstream downloader")
        for artifact in recipe.get("artifacts", []):
            download_file(
                artifact["url"],
                cache / artifact["filename"],
                proxy,
                workers,
                dry_run,
            )
        for command in recipe.get("commands", []):
            run_command(_format_command(command, values), proxy, dry_run)
        if not recipe.get("commands"):
            raise ValueError("This dataset needs --input pointing to data acquired with the official tool")
        return output
    if recipe_type == "authorized-input":
        if input_path is None:
            raise ValueError("This dataset needs --input pointing to an authorized official download")
        print(f"using input    {input_path}")
        return input_path
    raise ValueError(f"Unsupported recipe type {recipe_type} for {dataset_id}")


def execute_postprocess(
    profile: dict[str, Any],
    output: Path,
    cache: Path,
    proxy: str | None,
    workers: int,
    dry_run: bool,
) -> None:
    values = {
        "output": str(output),
        "cache": str(cache),
        "repo": str(REPO_ROOT),
        "python": sys.executable,
        "workers": str(workers),
    }
    for command in profile.get("postprocess", []):
        run_command(_format_command(command, values), proxy, dry_run)


def verify_rules(root: Path, rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for rule in rules:
        pattern = rule["glob"]
        matches = [path for path in root.glob(pattern) if path.is_file()]
        minimum = int(rule.get("min_count", 1))
        results.append(
            {
                "glob": pattern,
                "count": len(matches),
                "minimum": minimum,
                "ok": len(matches) >= minimum,
            }
        )
    return results


def check_url(url: str, proxy: str | None) -> tuple[bool, str]:
    handlers = []
    if proxy:
        normalized = proxy if "://" in proxy else f"http://{proxy}"
        handlers.append(urllib.request.ProxyHandler({"http": normalized, "https": normalized}))
    opener = urllib.request.build_opener(*handlers)
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with opener.open(request, timeout=20) as response:
            return True, str(response.status)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, str(exc)
