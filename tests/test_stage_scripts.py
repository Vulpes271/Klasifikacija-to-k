"""Check stage-specific Docker launchers without a Docker daemon or GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which("bash"), "Bash required")
class StageScriptTests(unittest.TestCase):
    def test_stage1_prepares_local_models_and_forwards_input(self):
        self.run_helper("stage1_describe", True)

    def test_stage2_api_only_does_not_start_ollama_or_require_host_gpu(self):
        self.run_helper("stage2_reason", False)

    def run_helper(self, stage, local):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts, binaries = root / "scripts", root / "bin"
            scripts.mkdir(); binaries.mkdir()
            source = Path(__file__).resolve().parents[1] / "scripts"
            for name in ("run_stage.sh", f"{stage}.sh"):
                shutil.copy2(source / name, scripts / name)
            docker = binaries / "docker"
            docker.write_text('''#!/usr/bin/env bash
printf '<%s>' "$@" >> "$MOCK_LOG"
printf '\\n' >> "$MOCK_LOG"
if [[ " $* " == *" --list-local-models "* ]] && [[ "$MOCK_LOCAL" == 1 ]]; then
  printf 'model-a\\nmodel-b\\n'
fi
''')
            smi = binaries / "nvidia-smi"
            smi.write_text('#!/usr/bin/env bash\necho GPU_CHECK >> "$MOCK_LOG"\n')
            for p in (docker, smi): p.chmod(0o755)
            log = root / "calls.log"
            env = {**os.environ, "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                   "MOCK_LOG": str(log), "MOCK_LOCAL": "1" if local else "0"}
            flag = "--input" if local else "--captions"
            p = subprocess.run(["bash", str(scripts / f"{stage}.sh"), "sample", flag, "/datasets/RGB folder/data.jsonl"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            calls = log.read_text()
            self.assertIn("<" + flag + "></datasets/RGB folder/data.jsonl>", calls)
            self.assertIn(f"<llm_object_classifier.{stage}>", calls)
            if local:
                self.assertIn("<ollama><pull><model-a>", calls)
                self.assertIn("<ollama><stop><model-b>", calls)
                self.assertIn("GPU_CHECK", calls)
            else:
                self.assertNotIn("GPU_CHECK", calls)
                self.assertNotIn("<up>", calls)
                self.assertNotIn("<ollama><pull>", calls)
