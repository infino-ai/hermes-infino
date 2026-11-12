"""An in-process stand-in for the Infino data API, for unit tests.

It keeps rows in memory and answers the routes the plugin calls, with the
same status codes the platform documents (409 with Retry-After on a busy
table, 400 for a hybrid search the deployment cannot embed, 403 for an
embedding column an account cannot have, 400 for one a deployment with no
embedding server cannot, 404 for a delete in a database or table that does
not exist, 413 for an append a busy worker's memory budget cannot take).
Search is a naive keyword match:
these tests check the plugin's logic, not ranking. The real platform is
exercised by the end-to-end test on a Hermes install.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# What the platform answers (400) for a create_table declaring an embedding
# column on a deployment with no embedding server: the embedding library's
# words for a JSON schema field, which the worker passes through.
NO_EMBEDDING_SERVER = ('schema field "embedding" of type "embedding" needs an embedding server, and this '
                       'deployment has none configured')
# What a worker answers (413, plain text) for an append its connection's memory
# budget cannot take while other queries fill it: the engine's OverBudget error
# ("over budget: ...") under the worker's "<operation> error: " (the platform's
# worker/src/error.rs, classify_infino and ConnectionError::Engine).
OVER_BUDGET = "append error: over budget: the connection's memory budget is spent"
# Rows a reranked search reads before grading: the deployment's triage pool.
RERANK_POOL = 1_000


class FakePlatform:
    def __init__(self) -> None:
        self.databases: dict[str, dict[str, list[dict]]] = {}
        self.schemas: dict[tuple[str, str], list[dict]] = {}
        self.requests: list[tuple[str, str]] = []
        self.busy_appends = 0          # next N appends answer 409
        self.failed_appends = 0        # next N appends answer 500 (not retried)
        self.trials = 0                # free accounts handed out
        self.hybrid_refused = False    # hybrid search answers 400 (no embedder)
        # A search naming `rerank`: refused as a deployment without the triage
        # pass refuses it (501) unless `rerank_supported`; then graded by
        # `grade` (question, row) -> score, highest first, the best `k` kept.
        self.rerank_supported = False
        self.rerank_unknown = False    # a worker older than `rerank`: 400, an unknown field
        self.grade = None
        self.rerank_statuses: list[int] = []  # the next reranked searches answer these statuses, in order
        self.rerank_delay_s = 0.0      # a reranked search answers after this long
        self.reranked: list[dict] = []  # each reranked search's body
        self.hybrid_statuses: list[int] = []  # the next hybrid searches answer these statuses, in order
        self.failed_deletes = 0        # next N deletes answer 500
        self.refused_marker = None     # an append carrying a row with this text answers refused_status
        self.refused_status = 400
        self.over_budget = False       # every append answers 413 OVER_BUDGET (a busy worker)
        self.no_embedder = False       # create_table refuses embedding columns with 403
        self.no_embedding_server = False  # create_table refuses embedding columns with 400 (none deployed)
        self.schema_patch_refused = False  # a schema patch (adding a column) answers 403
        self.database_limit = None     # a key that may make no more: creates past this count answer 403
        self.refused_databases: set[str] = set()  # creating one of these answers 403 (a key for other databases)
        self.refused_tables: set[str] = set()  # creating a table of one of these names answers 403
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
                # A worker's error is plain text; everything else here is JSON.
                text = isinstance(body, str)
                raw = body.encode() if text else json.dumps(body if body is not None else {}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/plain" if text else "application/json")
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
                if isinstance(body, dict) and body.get("rerank") is not None and fake.rerank_delay_s:
                    time.sleep(fake.rerank_delay_s)  # outside the lock: other requests go on meanwhile
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
            # A key that may make no more databases is refused before the name is looked at.
            if self.database_limit is not None and len(self.databases) >= self.database_limit:
                return 403, {"error": "this key may not create another database"}, None
            if body["name"] in self.refused_databases:
                return 403, {"error": f"this key may not create database {body['name']}"}, None
            if body["name"] in self.databases:
                return 409, {"error": "exists"}, None
            self.databases[body["name"]] = {}
            return 200, {}, None
        op, db = parts[1], urllib.parse.unquote(parts[2])
        if op == "delete" and params["table"] not in self.databases.get(db, {}):
            return 404, {"error": f"no table {params['table']} in database {db}"}, None
        tables = self.databases.setdefault(db, {})
        if op == "list_tables":
            return 200, list(tables), None
        if op == "schema":
            if body.get("patch") is not None:
                if self.schema_patch_refused:
                    return 403, {"error": "this key may not manage the database"}, None
                schema = self.schemas.setdefault((db, body["table_name"]), [])
                have = {f["name"] for f in schema}
                schema += [f for f in body["patch"].get("fields", []) if f["name"] not in have]
            return 200, self.schemas.get((db, body["table_name"]), []), None
        if op == "create_table":
            if body["table_name"] in tables:
                return 409, {"error": "exists"}, None
            if body["table_name"] in self.refused_tables:
                return 403, {"error": "this key may not create tables in the database"}, None
            if self.no_embedder and any(f["type"] == "embedding" for f in body["schema"]):
                return 403, {"error": "embedding columns are not enabled for this account"}, None
            if self.no_embedding_server and any(f["type"] == "embedding" for f in body["schema"]):
                return 400, {"error": NO_EMBEDDING_SERVER}, None
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
            if self.over_budget:
                return 413, OVER_BUDGET, None
            if self.refused_marker and any(self.refused_marker in json.dumps(r) for r in body["data"]):
                return self.refused_status, {"error": "row refused"}, None
            if params["table"] not in tables:
                return 404, {"error": f"no table {params['table']}"}, None
            rows = tables[params["table"]]
            for r in body["data"]:
                rows.append({"_id": str(len(rows)), **r})
            return 200, {}, None
        if op == "delete":
            if self.failed_deletes > 0:
                self.failed_deletes -= 1
                return 500, {"error": "delete error"}, None
            rows = tables[params["table"]]
            gone = {id(r) for r in self._where(db, params["table"], rows, params["predicate"])}
            kept = [r for r in rows if id(r) not in gone]
            matched = len(rows) - len(kept)
            tables[params["table"]] = kept
            return 200, {"matched": matched, "n_tombstoned": matched, "n_not_found": 0}, None
        if op in ("hybrid_search", "bm25_search") and body.get("rerank") is not None:
            return self._reranked(db, op, body)
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
            # The statements the plugin runs: a table's rows (the skills load,
            # whose ORDER BY and LIMIT are not applied), and rows or a count
            # where a predicate holds (the mirror's checks, a forget's lookup).
            m = re.match(r"SELECT (.*?) FROM (\w+)(?: WHERE (.*?))?(?: ORDER BY .*)?$", body["query"], re.S)
            if m.group(2) not in tables:
                return 400, {"error": f"table '{m.group(2)}' not found"}, None
            rows = list(tables[m.group(2)])
            if m.group(3) is not None:
                rows = self._where(db, m.group(2), rows, m.group(3))
            if m.group(1).startswith("count(*)"):
                return 200, {"data": [{"n": len(rows)}]}, None
            return 200, {"data": rows}, None
        return 404, {"error": f"no route {op}"}, None

    def _reranked(self, db: str, op: str, body: dict):
        """A search with `rerank`, as the platform answers it: refused where
        there is no triage pass; otherwise the search's rows (the plain
        search's, unscripted) graded for the question, best first, under
        `rows` beside the pass's coverage."""
        self.reranked.append(body)
        if self.rerank_unknown:
            # A worker older than the field: its request type refuses it, listing the fields it knows.
            fields = ("`table_name`, `text_field`, `text_query`, `mode`, `vector_field`, `vector_query`, `k`, "
                      "`projection`") if op == "hybrid_search" else "`table_name`, `field_name`, `query`, `k`"
            return 400, {"error": f"unknown field `rerank`, expected one of {fields}"}, None
        if not self.rerank_supported:
            return 501, {"error": "rerank is not available on this deployment: it runs no triage pass "
                                  "(ask.triage); send the query without `rerank`"}, None
        if self.rerank_statuses:
            return self.rerank_statuses.pop(0), {"error": "scripted"}, None
        plain = {k: v for k, v in body.items() if k != "rerank"}
        plain["k"] = max(int(body.get("k") or 0), RERANK_POOL)
        saved = (self.hybrid_refused, self.hybrid_statuses)
        self.hybrid_refused, self.hybrid_statuses = False, []
        try:
            status, out, headers = self._route(["v1", op, db], {}, plain)
        finally:
            self.hybrid_refused, self.hybrid_statuses = saved
        if status != 200:
            return status, out, headers
        question = body["rerank"]["question"]
        grade = self.grade or (lambda q, r: float(r.get("score") or 0.0))
        rows = sorted(out["data"], key=lambda r: -grade(question, r))
        k = body["rerank"].get("k") or body.get("k") or len(rows)
        return 200, {"rows": rows[:k], "rerank": {"graded": len(rows)}}, None

    def _where(self, db: str, table: str, rows: list[dict], predicate: str) -> list[dict]:
        """The rows a SQL predicate holds for, evaluated by SQLite over the
        table's columns (its schema's and its rows'), with the engine's
        `ends_with` function."""
        if not rows:
            return []
        columns = sorted({f["name"] for f in self.schemas.get((db, table), [])} | {k for r in rows for k in r})
        con = sqlite3.connect(":memory:")
        con.create_function("ends_with", 2,
                            lambda s, t: None if s is None or t is None else int(str(s).endswith(str(t))))
        con.create_function("strpos", 2,
                            lambda s, t: None if s is None or t is None else str(s).find(str(t)) + 1)
        quoted = ", ".join('"' + c + '"' for c in columns)
        con.execute(f"CREATE TABLE t (_i INTEGER, {quoted})")
        con.executemany(f"INSERT INTO t VALUES ({', '.join('?' * (len(columns) + 1))})",
                        [(i, *[r.get(c) if not isinstance(r.get(c), (dict, list)) else json.dumps(r.get(c))
                               for c in columns]) for i, r in enumerate(rows)])
        keep = {i for (i,) in con.execute(f"SELECT _i FROM t WHERE {predicate}")}
        con.close()
        return [r for i, r in enumerate(rows) if i in keep]
