"""Prepare launch files and syntax-check them without executing the launcher."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
import shutil
from unittest.mock import patch


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("display_prepare", HERE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class BundleTests(unittest.TestCase):
    def test_explicit_prefill_chunk_and_invalid_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "recipe"
            fixture.mkdir()
            shutil.copytree(prepare.ROOT / "scripts", fixture / "scripts")
            for name in ("start.sh", "stop.sh", "status.sh", "verify.sh", "AGENTS.md"):
                shutil.copy2(prepare.ROOT / name, fixture / name)
            with patch.object(prepare, "ROOT", fixture):
                for setting, expected in (("", "512"), ("DISPLAY_PREFILL_CHUNK=4096 # explicit\n", "4096")):
                    with self.subTest(setting=setting):
                        (fixture / ".env").write_text("IMAGE=base:latest\nPREFILL_CHUNK=2048\n" + setting)
                        destination = Path(directory) / ("bundle" + expected)
                        prepare.prepare(destination)
                        self.assertIn("PREFILL_CHUNK=" + expected + "\n", (destination / ".env").read_text())
                for setting in ("0", "-1", "abc", "1.5", "1\nDISPLAY_PREFILL_CHUNK=2"):
                    with self.subTest(invalid=setting):
                        (fixture / ".env").write_text("IMAGE=base:latest\nDISPLAY_PREFILL_CHUNK=" + setting + "\n")
                        destination = Path(directory) / "invalid"
                        with self.assertRaises(ValueError):
                            prepare.prepare(destination)
                        self.assertFalse(destination.exists())

    def test_bundle_is_separate_and_refuses_overwrite(self):
        active = {name: (prepare.ROOT / name).read_bytes() for name in (".env", "start.sh")}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            prepare.prepare(destination)
            start = (destination / "start.sh").read_text()
            config = (destination / ".env").read_text()
            self.assertIn("IMAGE=" + prepare.IMAGE, config)
            root_config = (prepare.ROOT / ".env").read_text()
            spill = next(line for line in root_config.splitlines() if line.startswith("SPILL_GIB="))
            self.assertIn(spill + "\n", config)
            self.assertIn("PREFILL_CHUNK=512\n", config)
            self.assertIn("--device /dev/dri", start)
            self.assertIn("-e TF_DS_DISPLAY_KV_MIB=1792", start)
            self.assertNotIn("docker rm -f tfv41-", start)
            self.assertLess(start.index("REFUSING experimental startup"), start.index('if [ "${PREFLIGHT'))
            subprocess.run(["bash", "-n", str(destination / "start.sh")], check=True)
            for name in ("stop.sh", "status.sh", "verify.sh"):
                subprocess.run(["bash", "-n", str(destination / name)], check=True)
            with self.assertRaises(ValueError):
                prepare.prepare(destination)
        for name, original in active.items():
            self.assertEqual((prepare.ROOT / name).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
