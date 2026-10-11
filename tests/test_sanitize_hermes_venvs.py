"""Unit tests for sanitize_hermes_venvs.py."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from sanitize_hermes_venvs import (
    check_and_clean_installs_dir,
    check_and_clean_lazy_packages,
    sanitize_all,
)


class SanitizeHermesVenvsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="test-sanitize-venvs-")
        self.addCleanup(self.tmp.cleanup)
        self.hermes_home = Path(self.tmp.name) / ".hermes"
        self.hermes_home.mkdir(parents=True)

    def test_prunes_incompatible_pyvenv_cfg_version(self):
        install_dir = self.hermes_home / "installs" / "ffad2f5543956651"
        env_dir = install_dir / "environments" / "1f66ccbc0e3a496dba9ebc6b2288b5e4" / "venv"
        env_dir.mkdir(parents=True)
        (env_dir / "pyvenv.cfg").write_text("version_info = 3.14.7\n", encoding="utf-8")

        cleaned = check_and_clean_installs_dir(self.hermes_home / "installs", (3, 12))
        self.assertEqual(len(cleaned), 1)
        self.assertFalse(install_dir.exists())

    def test_prunes_incompatible_lib_directory(self):
        install_dir = self.hermes_home / "installs" / "ffad2f5543956651"
        lib_dir = install_dir / "environments" / "1f66ccbc0e3a496dba9ebc6b2288b5e4" / "venv" / "lib" / "python3.14"
        lib_dir.mkdir(parents=True)

        cleaned = check_and_clean_installs_dir(self.hermes_home / "installs", (3, 12))
        self.assertEqual(len(cleaned), 1)
        self.assertFalse(install_dir.exists())

    def test_prunes_incompatible_facts_json_references(self):
        install_dir = self.hermes_home / "installs" / "ffad2f5543956651"
        install_dir.mkdir(parents=True)
        (install_dir / "facts.json").write_text(
            '{"environment": ".../venv/lib/python3.14/site-packages"}',
            encoding="utf-8",
        )

        cleaned = check_and_clean_installs_dir(self.hermes_home / "installs", (3, 12))
        self.assertEqual(len(cleaned), 1)
        self.assertFalse(install_dir.exists())

    def test_preserves_compatible_environments(self):
        install_dir = self.hermes_home / "installs" / "ffad2f5543956651"
        env_dir = install_dir / "environments" / "1f66ccbc0e3a496dba9ebc6b2288b5e4" / "venv"
        lib_dir = env_dir / "lib" / "python3.12"
        lib_dir.mkdir(parents=True)
        (env_dir / "pyvenv.cfg").write_text("version_info = 3.12.12\n", encoding="utf-8")
        (install_dir / "facts.json").write_text(
            '{"environment": ".../venv/lib/python3.12/site-packages"}',
            encoding="utf-8",
        )

        cleaned = check_and_clean_installs_dir(self.hermes_home / "installs", (3, 12))
        self.assertEqual(len(cleaned), 0)
        self.assertTrue(install_dir.exists())

    def test_prunes_mismatched_lazy_packages_abi(self):
        lazy_dir = self.hermes_home / "lazy-packages"
        lazy_dir.mkdir(parents=True)
        (lazy_dir / ".python-abi").write_text("3.14:cpython-314-x86_64-linux-gnu", encoding="utf-8")
        pkg_dir = lazy_dir / "pydantic"
        pkg_dir.mkdir()

        pruned = check_and_clean_lazy_packages(lazy_dir, (3, 12))
        self.assertTrue(pruned)
        self.assertFalse(pkg_dir.exists())

    def test_sanitize_all_cleans_root_and_profiles(self):
        # Profile install
        profile_install = self.hermes_home / "profiles" / "work" / "installs" / "ffad2f5543956651"
        (profile_install / "environments" / "abc" / "venv" / "lib" / "python3.14").mkdir(parents=True)

        cleaned = sanitize_all(self.hermes_home, (3, 12))
        self.assertIn(str(profile_install), cleaned)
        self.assertFalse(profile_install.exists())


if __name__ == "__main__":
    unittest.main()
