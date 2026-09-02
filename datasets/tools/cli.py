"""Unified command-line interface for acquiring Pi3 datasets."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .engine import (
    REPO_ROOT,
    check_url,
    execute_postprocess,
    execute_recipe,
    human_bytes,
    resolve_path,
    verify_rules,
)
from .registry import DatasetCatalog, load_catalog


MODE_LABELS = {
    "automatic": "automatic",
    "official-tool": "official tool",
    "authorized-input": "authorized input",
}


def _profile(catalog: DatasetCatalog, dataset_id: str, name: str) -> tuple[dict[str, Any], str]:
    try:
        return catalog.profile(dataset_id, name)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def command_list(catalog: DatasetCatalog, args: argparse.Namespace) -> int:
    print(f"{'dataset':18} {'minimal':18} {'train':18} default minimal output")
    for dataset_id, dataset in sorted(catalog.datasets.items()):
        minimal = MODE_LABELS[dataset["profiles"]["minimal"]["mode"]]
        train = MODE_LABELS[dataset["profiles"]["train"]["mode"]]
        output = dataset["profiles"]["minimal"]["default_output"]
        print(f"{dataset_id:18} {minimal:18} {train:18} {output}")
    return 0


def command_info(catalog: DatasetCatalog, args: argparse.Namespace) -> int:
    try:
        dataset = catalog.get(args.dataset)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc
    print(dataset["name"])
    print(f"id             {args.dataset.lower()}")
    print(f"homepage       {dataset['homepage']}")
    print(f"license        {dataset['license_url']}")
    print(f"loader         {dataset.get('loader', 'n/a')}")
    for profile_name in ("minimal", "train"):
        profile = dataset["profiles"][profile_name]
        print()
        print(f"{profile_name} profile")
        print(f"mode           {MODE_LABELS[profile['mode']]}")
        print(f"purpose        {profile['description']}")
        print(f"download       {human_bytes(profile.get('download_bytes'))}")
        print(f"disk           {human_bytes(profile.get('disk_bytes'))}")
        print(f"output         {profile['default_output']}")
        for prerequisite in profile.get("prerequisites", []):
            print(f"prerequisite   {prerequisite}")
    return 0


def command_doctor(catalog: DatasetCatalog, args: argparse.Namespace) -> int:
    profile, normalized = _profile(catalog, args.dataset, args.profile)
    dataset = catalog.get(args.dataset)
    output = resolve_path(args.output or profile["default_output"])
    usage = shutil.disk_usage(output.parent if output.parent.exists() else REPO_ROOT)
    needed = profile.get("disk_bytes")
    enough = needed is None or usage.free >= needed
    print(f"dataset        {dataset['name']}")
    print(f"profile        {normalized}")
    print(f"mode           {MODE_LABELS[profile['mode']]}")
    print(f"python         {sys.version.split()[0]}")
    print(f"output         {output}")
    print(f"free disk      {human_bytes(usage.free)}")
    print(f"estimated disk {human_bytes(needed)}")
    print(f"disk check     {'ok' if enough else 'insufficient'}")
    if profile.get("prerequisites"):
        print("prerequisites")
        for item in profile["prerequisites"]:
            print(f"  {item}")
    if args.network:
        urls = [
            artifact["url"]
            for artifact in profile["recipe"].get("artifacts", [])
            if artifact.get("url")
        ]
        urls.extend(
            file_record["url"]
            for file_record in profile["recipe"].get("files", [])
            if file_record.get("url")
        )
        for url in urls:
            ok, detail = check_url(url, args.proxy)
            print(f"network        {'ok' if ok else 'failed'}  {detail}  {url}")
            enough = enough and ok
    return 0 if enough else 1


def command_fetch(catalog: DatasetCatalog, args: argparse.Namespace) -> int:
    profile, normalized = _profile(catalog, args.dataset, args.profile)
    dataset = catalog.get(args.dataset)
    output = resolve_path(args.output or profile["default_output"])
    cache = resolve_path(args.cache_dir or f"data/.dataset_tools_cache/{args.dataset}/{normalized}")
    input_path = resolve_path(args.input) if args.input else None
    print(f"dataset        {dataset['name']}")
    print(f"profile        {normalized}")
    print(f"mode           {MODE_LABELS[profile['mode']]}")
    print(f"output         {output}")
    if profile.get("license_accept_required") and not args.accept_license:
        print(f"license        {dataset['license_url']}")
        raise SystemExit("Read the dataset license, then rerun with --accept-license")
    if normalized == "minimal" and input_path is None and not args.force and output.exists():
        existing = verify_rules(output, dataset.get("verify", []))
        if existing and all(result["ok"] for result in existing):
            print("result         existing minimal dataset already passes all layout checks")
            print("hint           add --force to download and prepare it again")
            return 0
    actual_root = execute_recipe(
        args.dataset,
        profile,
        output,
        cache,
        input_path,
        args.proxy,
        args.workers,
        args.dry_run,
    )
    execute_postprocess(
        profile,
        actual_root,
        cache,
        args.proxy,
        args.workers,
        args.dry_run,
    )
    if args.dry_run:
        print("result         dry run completed without filesystem changes")
        return 0
    results = verify_rules(actual_root, dataset.get("verify", []))
    failed = [result for result in results if not result["ok"]]
    for result in results:
        status = "ok" if result["ok"] else "missing"
        print(
            f"verify         {status:7} {result['count']:6} files  "
            f"need {result['minimum']:3}  {result['glob']}"
        )
    if failed:
        raise SystemExit(f"Downloaded data failed {len(failed)} layout checks")
    print(f"result         ready at {actual_root}")
    print(f"next           {profile.get('next_command', 'python -m datasets.tools verify ' + args.dataset)}")
    return 0


def _verify_one(
    catalog: DatasetCatalog,
    dataset_id: str,
    profile_name: str,
    root_override: str | None,
) -> bool:
    profile, normalized = _profile(catalog, dataset_id, profile_name)
    dataset = catalog.get(dataset_id)
    root = resolve_path(root_override or profile["default_output"])
    results = verify_rules(root, dataset["verify"])
    ok = all(result["ok"] for result in results)
    print(f"{dataset_id:18} {normalized:7} {'ok' if ok else 'failed':7} {root}")
    for result in results:
        if not result["ok"]:
            print(
                f"  missing {result['glob']}  found {result['count']}  need {result['minimum']}"
            )
    return ok


def command_verify(catalog: DatasetCatalog, args: argparse.Namespace) -> int:
    if args.all:
        if args.root:
            raise SystemExit("--root cannot be combined with --all")
        results = [
            _verify_one(catalog, dataset_id, args.profile, None)
            for dataset_id in sorted(catalog.datasets)
        ]
        print(f"summary        {sum(results)}/{len(results)} datasets passed")
        return 0 if all(results) else 1
    if not args.dataset:
        raise SystemExit("Provide a dataset id or use --all")
    return 0 if _verify_one(catalog, args.dataset, args.profile, args.root) else 1


def command_catalog(catalog: DatasetCatalog, args: argparse.Namespace) -> int:
    if args.json:
        print(json.dumps(catalog.datasets, indent=2, ensure_ascii=False))
    else:
        print(catalog.path)
    return 0


def command_index(catalog: DatasetCatalog, args: argparse.Namespace) -> int:
    from .build_index import BUILDERS, build_index

    if args.dataset not in BUILDERS:
        supported = ", ".join(sorted(BUILDERS))
        raise SystemExit(f"Index generation is supported for: {supported}")
    profile, normalized = _profile(catalog, args.dataset, args.profile)
    root = resolve_path(args.root or profile["default_output"])
    print(f"dataset        {args.dataset}")
    print(f"profile        {normalized}")
    print(f"data root      {root}")
    build_index(args.dataset, root, Path(args.output), args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m datasets.tools",
        description="Download minimal test samples or full training datasets for Pi3",
    )
    parser.add_argument("--catalog", type=Path, help="use another catalog.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="show supported datasets and acquisition modes")
    list_parser.set_defaults(handler=command_list)

    info = subparsers.add_parser("info", help="show sizes, license, output layout, and prerequisites")
    info.add_argument("dataset")
    info.set_defaults(handler=command_info)

    doctor = subparsers.add_parser("doctor", help="check disk space, prerequisites, and optional network access")
    doctor.add_argument("dataset")
    doctor.add_argument("--profile", default="minimal", choices=["minimal", "test", "train", "full"])
    doctor.add_argument("--output")
    doctor.add_argument("--proxy")
    doctor.add_argument("--network", action="store_true")
    doctor.set_defaults(handler=command_doctor)

    fetch = subparsers.add_parser("fetch", help="download, extract, process, and validate a dataset")
    fetch.add_argument("dataset")
    fetch.add_argument("--profile", default="minimal", choices=["minimal", "test", "train", "full"])
    fetch.add_argument("--output")
    fetch.add_argument("--cache-dir")
    fetch.add_argument("--input", help="official TFRecord, archive, or prepared root for authorized datasets")
    fetch.add_argument("--proxy", help="HTTP proxy such as localhost:7890")
    fetch.add_argument("--workers", type=int, default=8)
    fetch.add_argument("--accept-license", action="store_true")
    fetch.add_argument("--dry-run", action="store_true")
    fetch.add_argument("--force", action="store_true", help="rerun an already valid minimal recipe")
    fetch.set_defaults(handler=command_fetch)

    verify = subparsers.add_parser("verify", help="check the loader-facing directory contract")
    verify.add_argument("dataset", nargs="?")
    verify.add_argument("--profile", default="minimal", choices=["minimal", "test", "train", "full"])
    verify.add_argument("--root")
    verify.add_argument("--all", action="store_true")
    verify.set_defaults(handler=command_verify)

    index = subparsers.add_parser(
        "index", help="build a relocatable training-sampling index"
    )
    index.add_argument("dataset")
    index.add_argument(
        "--profile", default="minimal", choices=["minimal", "test", "train", "full"]
    )
    index.add_argument("--root", help="override the profile data root")
    index.add_argument("--output", default="pi3_index.npy")
    index.add_argument("--rgb-depth-tolerance", type=float, default=0.02)
    index.add_argument("--pose-tolerance", type=float, default=0.02)
    index.add_argument("--render-pass", default="final")
    index.add_argument("--camera", action="append")
    index.set_defaults(handler=command_index)

    catalog_parser = subparsers.add_parser("catalog", help="print the editable dataset catalog path")
    catalog_parser.add_argument("--json", action="store_true")
    catalog_parser.set_defaults(handler=command_catalog)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog = load_catalog(args.catalog)
    try:
        code = args.handler(catalog, args)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"error: {exc}\n")
    raise SystemExit(code)
