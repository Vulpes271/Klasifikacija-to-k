"""Run the Linux helper with fake commands, so no Docker/GPU is required."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BASH = shutil.which("bash")
if BASH is None and os.name == "nt":
    candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    if candidate.exists():
        BASH = str(candidate)


@unittest.skipUnless(BASH, "Bash is needed to exercise the Linux helper")
class VisionScriptTests(unittest.TestCase):
    def test_helper_pulls_selected_models_and_forwards_config_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            source = Path(__file__).resolve().parents[1] / "scripts/test_vision_pipeline.sh"
            script = root / "scripts/test_vision_pipeline.sh"
            script.write_bytes(source.read_bytes())
            binary_dir = root / "bin"
            binary_dir.mkdir()
            docker = binary_dir / "docker"
            docker.write_text('''#!/usr/bin/env bash
printf '<%s>' "$@" >> "$MOCK_LOG"
printf '\\n' >> "$MOCK_LOG"
if [[ " $* " == *" --print-models "* ]]; then
    printf 'selected-vision:test\\nselected-reason:test\\n'
elif [[ " $* " == *" -u -m llm_object_classifier.vision_pipeline "* ]]; then
    mkdir -p results/test-run
    printf '{"ok": true}\\n' > results/test-run/summary.json
fi
''', encoding="utf-8")
            nvidia = binary_dir / "nvidia-smi"
            nvidia.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            for command in (docker, nvidia):
                command.chmod(0o755)
            log = root / "commands.log"
            env = {**os.environ, "PATH": str(binary_dir) + os.pathsep + os.environ["PATH"],
                   "MOCK_LOG": log.as_posix()}
            completed = subprocess.run([BASH, str(script), "test-run", "--config", "configs/custom config.json",
                                        "--caption-detail", "detailed", "--temperatures", "0.2", "0.8"],
                                       cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            commands = log.read_text(encoding="utf-8")
            for model in ("selected-vision:test", "selected-reason:test"):
                self.assertIn(f"<ollama><pull><{model}>", commands)
                self.assertIn(f"<--model><{model}>", commands)
            self.assertNotIn("qwen3.5:4b", commands)
            inference = next(line for line in commands.splitlines() if "<-u>" in line)
            self.assertIn("<--config><configs/custom config.json>", inference)
            self.assertIn("<--caption-detail><detailed>", inference)
            self.assertIn("<--temperatures><0.2><0.8>", inference)
            self.assertIn("<--output-dir><results/test-run><--require-gpu>", inference)


if __name__ == "__main__":
    unittest.main()
