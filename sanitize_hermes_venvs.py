#!/usr/bin/env python3
"""Sanitize incompatible persistent Python virtual environments and packages.

Hermes Agent's dependency manager (PM) creates virtual environments under:
  $HERMES_HOME/installs/<install_id>/environments/<env_hash>/venv

If the persistent volume (/data/.hermes) was previously mounted by a different
Python version (e.g. Python 3.14 on a host system or earlier container build),
those virtualenvs contain compiled C extensions (.cpython-314-*.so) such as
`pydantic_core._pydantic_core`.

When Python 3.12 boots, hermes_bootstrap activates the committed PM environment
onto sys.path. Python 3.12 cannot load C extensions built for Python 3.14,
causing `ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'`.

This script runs at container startup before config migration and before server.py.
It verifies any cached venvs under:
  1. $HERMES_HOME/installs/
  2. $HERMES_HOME/profiles/*/installs/
  3. $HERMES_HOME/lazy-packages/

Any venv or lazy package tree built for a Python version differing from the
running interpreter's major.minor version is safely pruned so Hermes can cleanly
re-create or re-install them for the current Python interpreter.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def get_current_pyver() -> tuple[int, int]:
    return (sys.version_info.major, sys.version_info.minor)


def check_and_clean_installs_dir(installs_dir: Path, target_pyver: tuple[int, int]) -> list[str]:
    """Inspect each install record in installs_dir and remove incompatible environments."""
    if not installs_dir.is_dir():
        return []

    target_tag = f"python{target_pyver[0]}.{target_pyver[1]}"
    pruned: list[str] = []

    for install_entry in list(installs_dir.iterdir()):
        if not install_entry.is_dir():
            continue

        incompatible = False
        reason = ""

        # Check environments under install_entry
        envs_dir = install_entry / "environments"
        if envs_dir.is_dir():
            for env in list(envs_dir.iterdir()):
                # 1. Check pyvenv.cfg
                pyvenv_cfg = env / "venv" / "pyvenv.cfg"
                if pyvenv_cfg.is_file():
                    try:
                        content = pyvenv_cfg.read_text(encoding="utf-8", errors="ignore")
                        for line in content.splitlines():
                            line_strip = line.strip()
                            if line_strip.startswith("version_info"):
                                # e.g. version_info = 3.14.7 or version = 3.14.7
                                _, _, val = line_strip.partition("=")
                                parts = val.strip().split(".")
                                if len(parts) >= 2:
                                    try:
                                        maj, min_ = int(parts[0]), int(parts[1])
                                        if (maj, min_) != target_pyver:
                                            incompatible = True
                                            reason = f"pyvenv.cfg has version {maj}.{min_} != {target_pyver[0]}.{target_pyver[1]}"
                                            break
                                    except ValueError:
                                        pass
                            elif line_strip.startswith("home"):
                                # e.g. home = .../python-3.14.7...
                                for py_maj in range(3, 4):
                                    for py_min in range(8, 20):
                                        tag = f"python{py_maj}.{py_min}"
                                        if tag in line_strip and tag != target_tag:
                                            incompatible = True
                                            reason = f"pyvenv.cfg home points to {tag} != {target_tag}"
                                            break
                    except OSError:
                        pass
                if incompatible:
                    break

                # 2. Check lib/pythonX.Y directory
                lib_dir = env / "venv" / "lib"
                if lib_dir.is_dir():
                    try:
                        for child in lib_dir.iterdir():
                            if child.is_dir() and child.name.startswith("python"):
                                if child.name != target_tag:
                                    incompatible = True
                                    reason = f"venv lib has {child.name} != {target_tag}"
                                    break
                    except OSError:
                        pass
                if incompatible:
                    break

        # 3. Check facts.json for foreign Python references
        facts_file = install_entry / "facts.json"
        if not incompatible and facts_file.is_file():
            try:
                content = facts_file.read_text(encoding="utf-8", errors="ignore")
                for py_maj in range(3, 4):
                    for py_min in range(8, 20):
                        tag = f"python{py_maj}.{py_min}"
                        if tag in content and tag != target_tag:
                            incompatible = True
                            reason = f"facts.json references {tag} != {target_tag}"
                            break
                    if incompatible:
                        break
            except OSError:
                pass

        if incompatible:
            print(f"[sanitize-venvs] Pruning incompatible environment at {install_entry} ({reason})", flush=True)
            shutil.rmtree(install_entry, ignore_errors=True)
            pruned.append(str(install_entry))

    return pruned


def check_and_clean_lazy_packages(lazy_dir: Path, target_pyver: tuple[int, int]) -> bool:
    """Validate ABI stamp of lazy-packages directory and prune if mismatched."""
    if not lazy_dir.is_dir():
        return False

    target_prefix = f"{target_pyver[0]}.{target_pyver[1]}"
    stamp_file = lazy_dir / ".python-abi"
    mismatched = False

    if stamp_file.is_file():
        try:
            val = stamp_file.read_text(encoding="utf-8", errors="ignore").strip()
            if val and not val.startswith(target_prefix):
                mismatched = True
        except OSError:
            pass
    else:
        # Check if there are any .so or compiled files for foreign Python versions
        for root, _, files in os.walk(lazy_dir):
            for file in files:
                if file.endswith(".so"):
                    for py_min in range(8, 20):
                        tag = f".cpython-3{py_min}"
                        if tag in file and py_min != target_pyver[1]:
                            mismatched = True
                            break
                if mismatched:
                    break
            if mismatched:
                break

    if mismatched:
        print(f"[sanitize-venvs] Pruning mismatched lazy-packages at {lazy_dir}", flush=True)
        for child in list(lazy_dir.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
        return True

    return False


def sanitize_all(hermes_home: Path, target_pyver: tuple[int, int] | None = None) -> list[str]:
    if target_pyver is None:
        target_pyver = get_current_pyver()

    cleaned: list[str] = []

    # 1. Root installs
    cleaned.extend(check_and_clean_installs_dir(hermes_home / "installs", target_pyver))

    # 2. Named profiles installs
    profiles_dir = hermes_home / "profiles"
    if profiles_dir.is_dir():
        try:
            for profile in profiles_dir.iterdir():
                if profile.is_dir():
                    cleaned.extend(check_and_clean_installs_dir(profile / "installs", target_pyver))
        except OSError:
            pass

    # 3. Durable lazy-packages
    lazy_dir = hermes_home / "lazy-packages"
    if check_and_clean_lazy_packages(lazy_dir, target_pyver):
        cleaned.append(str(lazy_dir))

    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanitize incompatible Hermes Python virtual environments.")
    parser.add_argument(
        "--hermes-home",
        type=Path,
        default=Path(os.environ.get("HERMES_HOME", "/data/.hermes")),
        help="Path to HERMES_HOME",
    )
    args = parser.parse_args()

    hermes_home = args.hermes_home.resolve()
    target_pyver = get_current_pyver()

    cleaned = sanitize_all(hermes_home, target_pyver)
    if cleaned:
        print(f"[sanitize-venvs] Successfully cleaned {len(cleaned)} incompatible environment(s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
