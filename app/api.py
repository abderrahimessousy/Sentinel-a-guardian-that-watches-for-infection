"""Tiny HTTP API (standard library only).

POST /diagnose  with JSON body = patient features
-> {"probability": 0.87, "diagnostic": "Infecté"}
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def make_handler(predictor):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/diagnose":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                prob = predictor.probability(data)
                self._send(200, {"probability": round(prob, 4),
                                 "diagnostic": predictor.diagnose(data)})
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": str(e)})

        def log_message(self, *args):
            pass

    return Handler


def run_server(predictor, host="127.0.0.1", port=8000):
    server = HTTPServer((host, port), make_handler(predictor))
    print(f"API running on http://{host}:{port}  (POST /diagnose)")
    server.serve_forever()
