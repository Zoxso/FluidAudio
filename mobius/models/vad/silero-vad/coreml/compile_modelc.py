#!/usr/bin/env python3
"""Compile unified Silero VAD CoreML packages into .mlmodelc bundles."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

DEFAULT_BASE_MODELS_DIR = Path("silero-vad-coreml")
DEFAULT_QUANTIZED_DIR = Path("quantized_models")
DEFAULT_OUTPUT_DIR = Path("compiled")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-models-dir",
        type=Path,
        default=DEFAULT_BASE_MODELS_DIR,
        help="Directory that contains the standard unified CoreML packages.",
    )
    parser.add_argument(
        "--quantized-dir",
        type=Path,
        default=DEFAULT_QUANTIZED_DIR,
        help="Directory that contains quantized unified CoreML packages.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for compiled .mlmodelc bundles.",
    )
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=True,
        help="Overwrite any existing compiled bundles (default).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip recompilation when a compiled bundle already exists.",
    )
    parser.add_argument(
        "--xcrun-binary",
        default="xcrun",
        help="Path to the xcrun executable (defaults to the system xcrun).",
    )
    return parser.parse_args(list(argv))


def find_unified_packages(*directories: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for directory in directories:
        if directory is None:
            continue
        directory = directory.expanduser().resolve()
        if not directory.exists():
            continue
        for package in directory.rglob("*.mlpackage"):
            if "unified" not in package.stem:
                continue
            key = package.stem
            if key in discovered:
                # Prefer the shortest path (usually top-level replicas) and skip duplicates.
                existing = discovered[key]
                if len(str(package)) < len(str(existing)):
                    discovered[key] = package
                continue
            discovered[key] = package
    return dict(sorted(discovered.items()))


def compile_package(
    package_path: Path,
    output_dir: Path,
    overwrite: bool,
    xcrun_binary: str,
) -> Tuple[Path, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled_path = output_dir / f"{package_path.stem}.mlmodelc"

    if compiled_path.exists():
        if overwrite:
            shutil.rmtree(compiled_path)
        else:
            return compiled_path, False

    cmd = [
        xcrun_binary,
        "coremlcompiler",
        "compile",
        str(package_path),
        str(output_dir),
    ]
    subprocess.run(cmd, check=True)

    if not compiled_path.exists():
        raise RuntimeError(
            f"coremlcompiler reported success but '{compiled_path}' was not created"
        )

    return compiled_path, True


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    packages = find_unified_packages(args.base_models_dir, args.quantized_dir)

    if not packages:
        print("No unified CoreML packages were found to compile.")
        return 1

    compiled = []
    skipped = []
    for name, package in packages.items():
        try:
            compiled_path, updated = compile_package(
                package_path=package,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                xcrun_binary=args.xcrun_binary,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Failed to compile {package}: {exc}")
            return exc.returncode or 1
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Unexpected error while compiling {package}: {exc}")
            return 1

        if updated:
            compiled.append(compiled_path)
        else:
            skipped.append(compiled_path)

    print("Compiled the following unified models to .mlmodelc:")
    for path in compiled:
        print(f"  ✓ {path}")
    if skipped:
        print("Skipped models that already had compiled outputs (use --overwrite to rebuild):")
        for path in skipped:
            print(f"  → {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
