import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from llm_object_classifier.cli import main


class QueueHandler(BaseHTTPRequestHandler):
    responses: list[str] = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        content = self.responses.pop(0)
        body = {
            "choices": [
                {
                    "message": {
                        "content": content,
                    }
                }
            ]
        }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, *_args):
        return


class ClientCliTests(unittest.TestCase):
    def run_server(self):
        server = HTTPServer(("127.0.0.1", 0), QueueHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def test_run_records_success_bad_json_and_missing_id(self):
        QueueHandler.responses = [
            json.dumps(
                {
                    "objects": [
                        {"id": 1, "classification": "movable", "reason": "Freestanding chair."},
                        {"id": 2, "classification": "fixed", "reason": "Structural wall."},
                    ]
                }
            ),
            "{bad json",
            json.dumps({"objects": [{"id": 1, "classification": "fixed", "reason": "Only one."}]}),
        ]
        server = self.run_server()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_path = root / "input.jsonl"
                input_path.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "frame_id": "frame-1",
                                    "objects": [{"id": 1, "label": "chair"}, {"id": 2, "label": "wall"}],
                                }
                            ),
                            json.dumps({"frame_id": "frame-2", "objects": [{"id": 1, "label": "box"}]}),
                            json.dumps(
                                {
                                    "frame_id": "frame-3",
                                    "objects": [{"id": 1, "label": "chair"}, {"id": 2, "label": "wall"}],
                                }
                            ),
                        ]
                    ),
                    encoding="utf-8",
                )
                config_path = root / "config.json"
                config_path.write_text(
                    json.dumps(
                        {
                            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                            "model": "fake",
                            "temperature": 0,
                            "max_tokens": 128,
                            "timeout_seconds": 10,
                        }
                    ),
                    encoding="utf-8",
                )
                out_dir = root / "results" / "run"
                code = main(
                    [
                        "run",
                        "--input",
                        str(input_path),
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(out_dir),
                        "--resume",
                    ]
                )
                self.assertEqual(code, 0)
                records = [
                    json.loads(line)
                    for line in (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual([item["status"] for item in records], ["ok", "error", "error"])
                self.assertIn("raw_response", records[1])
                self.assertIn("Missing classifications", records[2]["error"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()

