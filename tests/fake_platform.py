"""An in-process stand-in for the Infino data API, for unit tests.

It keeps rows in memory and answers the routes the plugin calls, with the
same status codes the platform documents (409 with Retry-After on a busy
table, 400 for a hybrid search the deployment cannot embed, 403 for an
embedding column an account cannot have). Search is a naive keyword match:
these tests check the plugin's logic, not ranking. The real platform is
exercised by the end-to-end test on a Hermes install.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakePlatform:
    def __init__(self) -> None:
        self.databases: dict[str, dict[str, list[dict]]] = {}
        self.schemas: dict[tuple[str, str], list[dict]] = {}
        self.requests: list[tuple[str, str]] = []
        self.busy_appends = 0          # next N appends answer 409
        self.failed_appends = 0        # next N appends answer 500 (not retried)
        self.trials = 0                # free accounts handed out
        self.hybrid_refused = False    # hybrid search answers 400 (no embedder)
        self.hybrid_statuses: list[int] = []  # the next hybrid searches answer these statuses, in order
        self.failed_deletes = 0        # next N deletes answer 500
        self.no_embedder = False       # create_table refuses embedding columns with 403
        self.database_limit = None     # like a free plan: creates past this count answer 403
        self.trial_status = None       # /v1/trial answers this status (501: no free accounts; 409: one per network)
        self.trial_without_key = False  # /v1/trial answers 200 with no api_key
        self.valid_keys = None         # a set: any other bearer key answers 401
        self.keys_seen: list[str] = []  # the bearer key of every request
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()

    def rows(self, db: str, table: str) -> list[dict]:
        return list(self.databases.get(db, {}).get(table, []))

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # quiet
                pass

            def _send(self, status: int, body=None, headers=None):
                raw = json.dumps(body if body is not None else {}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _refused_key(self) -> bool:
                """Record the request's key; answer 401 when the platform does not know it."""
                key = (self.headers.get("Authorization") or "").removeprefix("Bearer ")
                fake.keys_seen.append(key)
                if fake.valid_keys is not None and key not in fake.valid_keys and self.path != "/v1/trial":
                    self._send(401, {"error": "invalid api key"})
                    return True
                return False

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                with fake._lock:
                    fake.requests.append((parsed.path, "{}"))
                    if self._refused_key():
                        return
                    if parsed.path == "/v1/databases":
                        self._send(200, [{"name": n} for n in fake.databases])
                    else:
                        self._send(404, {"error": "no route"})

            def do_POST(self):
                parsed = urllib.parse.urlparse(self.path)
                params = dict(urllib.parse.parse_qsl(parsed.query))
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length)) if length else {}
                parts = parsed.path.strip("/").split("/")
                with fake._lock:
                    fake.requests.append((parsed.path, json.dumps(params)))
                    if self._refused_key():
                        return
                    status, out, headers = fake._route(parts, params, body)
                self._send(status, out, headers)

        return Handler

    def _route(self, parts, params, body):
        if parts == ["v1", "trial"]:
            if self.trial_status is not None:
                return self.trial_status, {"error": "no free account"}, None
            if self.trial_without_key:
                return 200, {"database": body["database"]}, None
            self.trials += 1
            self.databases.setdefault(body["database"], {})
            return 200, {"api_key": f"trial-key-{self.trials}", "database": body["database"],
                         "credit_cents": 500}, None
        if parts == ["v1", "databases"]:
            # The platform checks the plan's limit before the name.
            if self.database_limit is not None and len(self.databases) >= self.database_limit:
                return 403, {"error": "free plan database limit reached"}, None
            if body["name"] in self.databases:
                return 409, {"error": "exists"}, None
            self.databases[body["name"]] = {}
            return 200, {}, None
        op, db = parts[1], urllib.parse.unquote(parts[2])
        tables = self.databases.setdefault(db, {})
        if op == "list_tables":
            return 200, list(tables), None
        if op == "schema":
            return 200, self.schemas.get((db, body["table_name"]), []), None
        if op == "create_table":
            if body["table_name"] in tables:
                return 409, {"error": "exists"}, None
            if self.no_embedder and any(f["type"] == "embedding" for f in body["schema"]):
                return 403, {"error": "embedding columns are not enabled for this account"}, None
            types = {f["name"]: f["type"] for f in body["schema"]}
            for column in (body.get("indexes") or {}).get("fts", []):
                if types.get(column) != "large_utf8":  # the platform's rule for FTS columns
                    return 400, {"error": f'FTS column "{column}" must be LargeUtf8'}, None
            for index in (body.get("indexes") or {}).get("vector", []):
                if types.get(index["column"]) == "embedding":  # indexed by itself on the platform
                    return 400, {"error": f'indexes.vector names "{index["column"]}", an embedding column'}, None
            tables[body["table_name"]] = []
            self.schemas[(db, body["table_name"])] = body["schema"]
            return 200, {}, None
        if op == "append":
            if self.busy_appends > 0:
                self.busy_appends -= 1
                return 409, {"error": "busy"}, {"Retry-After": "0"}
            if self.failed_appends > 0:
                self.failed_appends -= 1
                return 500, {"error": "append error"}, None
            rows = tables[params["table"]]
            for r in body["data"]:
                rows.append({"_id": str(len(rows)), **r})
            return 200, {}, None
        if op == "delete":
            if self.failed_deletes > 0:
                self.failed_deletes -= 1
                return 500, {"error": "delete error"}, None
            predicate = params["predicate"]
            m = re.fullmatch(r"content = '((?:[^']|'')*)'", predicate)
            if m:
                target = m.group(1).replace("''", "'")
                gone = lambda r: r.get("content") == target  # noqa: E731
            else:
                m = re.fullmatch(r"row_id IN \((.*)\)", predicate)
                ids = {v.replace("''", "'") for v in re.findall(r"'((?:[^']|'')*)'", m.group(1))}
                gone = lambda r: r.get("row_id") in ids  # noqa: E731
            rows = tables[params["table"]]
            kept = [r for r in rows if not gone(r)]
            matched = len(rows) - len(kept)
            tables[params["table"]] = kept
            return 200, {"matched": matched, "n_tombstoned": matched, "n_not_found": 0}, None
        if op in ("hybrid_search", "bm25_search"):
            if op == "hybrid_search" and self.hybrid_refused:
                return 400, {"error": "no embedding server"}, None
            if op == "hybrid_search" and self.hybrid_statuses:
                return self.hybrid_statuses.pop(0), {"error": "scripted"}, None
            schema = self.schemas.get((db, body["table_name"]), [])
            if op == "hybrid_search" and not any(f["name"] == body["vector_field"] for f in schema):
                return 400, {"error": f"no column {body['vector_field']}"}, None
            text = body.get("text_query") or body.get("query") or ""
            field = body.get("text_field") or body.get("field_name")
            words = {w.lower() for w in re.findall(r"\w+", text)}
            hits = []
            for r in tables.get(body["table_name"], []):
                overlap = len(words & {w.lower() for w in re.findall(r"\w+", str(r.get(field, "")))})
                if overlap:
                    hits.append({**r, "score": float(overlap)})
            hits.sort(key=lambda h: -h["score"])
            return 200, {"data": hits[: body["k"]]}, None
        if op == "query_sql":
            # Only the statement the plugin runs: the skills table's rows.
            m = re.match(r"SELECT .* FROM (\w+)", body["query"])
            return 200, {"data": list(tables.get(m.group(1), []))}, None
        return 404, {"error": f"no route {op}"}, None
