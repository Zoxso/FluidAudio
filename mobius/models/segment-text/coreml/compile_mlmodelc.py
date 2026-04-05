from __future__ import annotations

import shutil
import subprocess
import sys
import typer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "compiled"

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def ensure_coremlcompiler() -> None:
    """Ensure ``xcrun coremlcompiler`` is available for the active Xcode."""
    xcrun_path = shutil.which("xcrun")
    if xcrun_path is None:
        print("Error: 'xcrun' not found on PATH. Install Xcode command line tools.", file=sys.stderr)
        sys.exit(1)

    try:
        subprocess.run([
            xcrun_path,
            "--find",
            "coremlcompiler",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Error: 'coremlcompiler' not found via xcrun. Check your Xcode installation.", file=sys.stderr)
        sys.exit(1)


def gather_packages(dir: str) -> list[Path]:
    """Return a list of all ``*.mlpackage`` bundles under the source dirs."""
    packages: list[Path] = []
    source = BASE_DIR / dir
    if not source.exists():
        print(f"Warning: {source.relative_to(BASE_DIR)} does not exist; skipping", file=sys.stderr)
        return packages
    packages.extend(source.rglob("*.mlpackage"))
    return packages


def compile_package(package: Path, output_dir: Path) -> None:
    """Compile a single ``.mlpackage`` bundle using ``xcrun coremlcompiler``."""
    relative_pkg = package.relative_to(BASE_DIR)
    resolved_output_dir = output_dir if output_dir.is_absolute() else BASE_DIR / output_dir
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = resolved_output_dir / f"{package.stem}.mlmodelc"

    if output_path.exists():
        shutil.rmtree(output_path)

    cmd = [
        "xcrun",
        "coremlcompiler",
        "compile",
        str(package),
        str(resolved_output_dir),
    ]

    try:
        relative_output = output_path.relative_to(BASE_DIR)
    except ValueError:
        relative_output = output_path

    print(f"Compiling {relative_pkg} -> {relative_output}")
    subprocess.run(cmd, check=True)


@app.command()
def compile(
    coreml_dir: Path = typer.Option(
        Path("sat_coreml"),
        help="Directory where the mlpackage is",
    ),
    output_dir: Path = typer.Option(
        Path("compiled"),
        help="Directory where the compiled model is written",
    ),
):
    ensure_coremlcompiler()
    packages = gather_packages(coreml_dir)

    if not packages:
        print("No .mlpackage bundles found to compile.")
        return

    for package in packages:
        try:
            compile_package(package, output_dir)
        except subprocess.CalledProcessError as exc:
            print(f"Failed to compile {package}: {exc}", file=sys.stderr)
            sys.exit(exc.returncode)

    print(f"Finished compiling {len(packages)} package(s) into {OUTPUT_ROOT.relative_to(BASE_DIR)}.")


if __name__ == "__main__":
    app()
