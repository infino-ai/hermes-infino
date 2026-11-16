"""A small client for the Infino platform's data API.

Only the routes the plugin uses: create a database and its tables, append
rows, recall memory, delete, and request a free account. The agent's own
retrieval is the Infino client's MCP server, not this. Standard library
only: the cloud needs nothing beyond Python itself.

Every append is a commit on the platform, and a table takes one write at a
time: a concurrent write gets 409 with `Retry-After`, and a database whose
workers are still starting gets 503 with `Retry-After`. Both are retried here.
Many rows at once go by a platform load job instead (`load`), as the Infino
client's index sends a large write.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://api.platform.infino.ws"

# Seconds to wait for one HTTP request.
REQUEST_TIMEOUT_S = 20.0
# Statuses that mean "try the identical request again after Retry-After".
RETRYABLE_STATUSES = (409, 503)
# How many times a retryable request is reissued before giving up.
MAX_RETRIES = 5
# Wait used when a retryable response carries no Retry-After header.
DEFAULT_RETRY_AFTER_S = 1.0
# Upper bound on any single wait, so a large Retry-After cannot stall a turn.
MAX_RETRY_WAIT_S = 10.0
# Seconds to wait for a load's rows file to upload.
UPLOAD_TIMEOUT_S = 300.0
# How often a load is asked how it stands, and how long it may take before the
# wait gives up on it (it goes on on the platform; the caller treats its rows
# as unsure).
LOAD_POLL_S = 2.0
LOAD_WAIT_S = 30 * 60.0
# A load's states while it may still write the table, and those of one whose
# rows landed.
LIVE_LOAD_STATES = ("admitting", "pending", "running", "cancelling")
LANDED_LOAD_STATES = ("succeeded", "synced")
# What the uploads route answers on a platform that runs no load jobs.
NO_LOADS_STATUSES = (404, 405, 501)


class InfinoError(Exception):
    """A request the platform refused, with its status and message."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class LoadsUnsupported(InfinoError):
    """The platform runs no load jobs: its uploads route is not there."""


class InfinoClient:
    """One API key against one platform URL."""

    def __init__(self, api_key: str, url: str = DEFAULT_URL, timeout: float = REQUEST_TIMEOUT_S,
                 retries: int = MAX_RETRIES):
        self._api_key = api_key
        self._url = url.rstrip("/")
        self._timeout = timeout
        # A call whose caller waits on it (a recall before a turn) takes none:
        # a busy or starting database recalls nothing that once.
        self._retries = retries

    # -- databases and tables ------------------------------------------------

    def ensure_database(self, name: str) -> None:
        """Create the database unless it already exists. The list is read
        first, so one that exists is used without asking to create it."""
        if name in self.list_databases():
            return
        try:
            self.create_database(name)
        except InfinoError as e:
            if e.status != 409:
                raise

    def create_database(self, name: str) -> None:
        """Create a new database; a name the account has already answers 409."""
        self._request("POST", "/v1/databases", body={"name": name})

    def list_databases(self) -> list[str]:
        out = self._request("GET", "/v1/databases")
        if isinstance(out, dict):
            out = out.get("databases", out.get("data", []))
        return [d if isinstance(d, str) else d.get("name", "") for d in out or []]

    def ensure_table(self, database: str, table: str, schema: list[dict], indexes: dict) -> None:
        """Create the table unless it already exists.

        A 409 here is either "name exists" (terminal, and what we want) or
        "another catalog change was in flight" (transient). Checking the
        table list tells them apart.
        """
        if table in self.list_tables(database):
            return
        body = {"table_name": table, "schema": schema, "indexes": indexes}
        try:
            self._request("POST", f"/v1/create_table/{_q(database)}", body=body)
        except InfinoError as e:
            if e.status == 409 and table in self.list_tables(database):
                return
            raise

    def column_names(self, database: str, table: str) -> list[str]:
        """`POST /v1/schema/{db}`: the table's column names."""
        out = self._request("POST", f"/v1/schema/{_q(database)}", body={"table_name": table})
        fields = out.get("fields", out.get("schema", [])) if isinstance(out, dict) else out
        return [f.get("name", "") for f in fields or [] if isinstance(f, dict)]

    def add_columns(self, database: str, table: str, fields: list[dict]) -> None:
        """`POST /v1/schema/{db}` with a patch: add columns to an existing
        table (a key that may manage the database)."""
        self._request("POST", f"/v1/schema/{_q(database)}", body={"table_name": table, "patch": {"fields": fields}})

    def drop_table(self, database: str, table: str) -> None:
        self._request("POST", f"/v1/drop_table/{_q(database)}", body={"table_name": table})

    def list_tables(self, database: str) -> list[str]:
        out = self._request("POST", f"/v1/list_tables/{_q(database)}")
        if isinstance(out, dict):
            out = out.get("tables", out.get("data", []))
        return [t if isinstance(t, str) else t.get("name", "") for t in out or []]

    # -- rows ----------------------------------------------------------------

    def append(self, database: str, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._request(
            "POST",
            f"/v1/append/{_q(database)}",
            params={"table": table},
            body={"data": rows},
        )

    def load(self, database: str, table: str, rows: list[dict[str, Any]], *,
             poll_s: Optional[float] = None, wait_s: Optional[float] = None) -> None:
        """Append rows by a platform load job, as the Infino client's index
        sends a large write: the rows as one NDJSON file, uploaded into the
        database's storage (`POST /v1/hydrate/{db}/uploads`, then a PUT to the
        session it opens, which carries no key), loaded into `table` with
        `if_exists: append` (`POST /v1/hydrate/{db}`), and waited for
        (`GET /v1/hydrate/{db}?table=`). The table keeps its columns and
        indexes, and its rows land as sent. Raises LoadsUnsupported when the
        platform runs no load jobs, InfinoError when a step is refused or the
        load fails, and TimeoutError when it outlasts `wait_s` (LOAD_WAIT_S),
        asking every `poll_s` (LOAD_POLL_S)."""
        if not rows:
            return
        poll_s = LOAD_POLL_S if poll_s is None else poll_s
        wait_s = LOAD_WAIT_S if wait_s is None else wait_s
        data = "".join(json.dumps(r, default=str) + "\n" for r in rows).encode()
        name = f"{table}-{uuid.uuid4().hex}.ndjson"
        origin = "{0.scheme}://{0.netloc}".format(urllib.parse.urlsplit(self._url))
        try:
            opened = self._request("POST", f"/v1/hydrate/{_q(database)}/uploads",
                                   body={"origin": origin, "files": [{"name": name, "size": len(data)}]})
        except InfinoError as e:
            if e.status in NO_LOADS_STATUSES:
                raise LoadsUnsupported(e.status, e.message) from None
            raise
        files = opened.get("files") if isinstance(opened, dict) else None
        prefix = opened.get("prefix") if isinstance(opened, dict) else None
        if not isinstance(files, list) or len(files) != 1 or not isinstance(prefix, str):
            raise InfinoError(0, f"the platform opened {len(files) if isinstance(files, list) else 0} "
                                 "upload sessions for 1 file")
        self._upload(files[0]["url"], data)
        self._request("POST", f"/v1/hydrate/{_q(database)}", body={
            "table": table, "source": {"kind": "prefix", "prefix": prefix}, "format": "ndjson",
            "if_exists": "append"})
        self._wait_for_load(database, table, prefix, poll_s, wait_s)

    def _upload(self, url: str, data: bytes) -> None:
        """PUT `data` to an upload session the platform opened; the session's
        URL is its credential, so no key goes with it."""
        req = urllib.request.Request(url, data=data, method="PUT", headers={"Content-Length": str(len(data))})
        try:
            with urllib.request.urlopen(req, timeout=max(self._timeout, UPLOAD_TIMEOUT_S)) as resp:
                resp.read()
        except urllib.error.HTTPError as e:
            raise InfinoError(e.code, e.read().decode("utf-8", "replace")) from None

    def _wait_for_load(self, database: str, table: str, prefix: str, poll_s: float, wait_s: float) -> None:
        """Wait until the load of `table` from the upload at `prefix` has ended.
        A table keeps one load record, whatever load made it, so a record of
        another upload (`_loads_from`) is an earlier load's, read again until
        this one's replaces it; so is none yet."""
        deadline = time.monotonic() + wait_s
        while True:
            try:
                record = self._request("GET", f"/v1/hydrate/{_q(database)}", params={"table": table})
            except InfinoError as e:
                if e.status != 404:
                    raise
                record = None
            if isinstance(record, dict) and _loads_from(record, prefix):
                state = str(record.get("state") or "")
                if state in LANDED_LOAD_STATES:
                    return
                if state not in LIVE_LOAD_STATES:
                    why = f": {record['error']}" if record.get("error") else ""
                    raise InfinoError(0, f"the load of {table} ended {state or 'with no state'}{why}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"the load of {table} from {prefix} had not ended after {wait_s:.0f} s")
            time.sleep(poll_s)

    def delete(self, database: str, table: str, predicate: str) -> dict:
        return self._request(
            "POST",
            f"/v1/delete/{_q(database)}",
            params={"table": table, "predicate": predicate},
        )

    def hybrid_search(
        self,
        database: str,
        table: str,
        text: str,
        *,
        text_field: str,
        vector_field: str,
        k: int,
        projection: Optional[list[str]] = None,
        rerank: Optional[str] = None,
    ) -> list[dict]:
        """BM25 and vector search fused; the server embeds `vector_text` for
        the vector leg. With `rerank` (the question), the platform's triage
        pass grades a wider pool of the search's rows for it and the best `k`
        come back in the grader's order (see `_rerank`)."""
        body = {
            "table_name": table,
            "text_field": text_field,
            "text_query": text,
            "mode": "or",
            "vector_field": vector_field,
            "vector_text": text,
            "k": k,
            "projection": projection,
            **_rerank(rerank, k),
        }
        return _rows(self._request("POST", f"/v1/hybrid_search/{_q(database)}", body=body, want_json=True))

    def bm25_search(
        self,
        database: str,
        table: str,
        text: str,
        *,
        field: str,
        k: int,
        projection: Optional[list[str]] = None,
        rerank: Optional[str] = None,
    ) -> list[dict]:
        """BM25 search; `rerank` as `hybrid_search` takes it."""
        body = {
            "table_name": table,
            "field_name": field,
            "query": text,
            "mode": "or",
            "k": k,
            "projection": projection,
            **_rerank(rerank, k),
        }
        return _rows(self._request("POST", f"/v1/bm25_search/{_q(database)}", body=body, want_json=True))

    def query_sql(self, database: str, sql: str) -> list[dict]:
        return _rows(
            self._request("POST", f"/v1/query_sql/{_q(database)}", body={"query": sql}, want_json=True)
        )

    # -- transport -----------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        want_json: bool = True,
    ) -> Any:
        url = self._url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if want_json:
            headers["Accept"] = "application/json"

        for attempt in range(self._retries + 1):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                message = e.read().decode("utf-8", "replace")
                if e.code in RETRYABLE_STATUSES and attempt < self._retries:
                    time.sleep(_retry_after(e.headers.get("Retry-After")))
                    continue
                raise InfinoError(e.code, message) from None
        raise AssertionError("unreachable")


def request_trial(url: str, database: str, timeout: float = REQUEST_TIMEOUT_S) -> dict:
    """Ask the platform for a free account: `POST /v1/trial`, the one route that
    answers without a key. Returns `api_key`, `database` (registered),
    `credit_cents` and `console_url`. Two refusals are ordinary: 501 (this
    platform offers no free accounts) and 409 (this network already had one)."""
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/trial", data=json.dumps({"database": database}).encode(), method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        raise InfinoError(e.code, e.read().decode("utf-8", "replace")) from None
    if not isinstance(out, dict) or not out.get("api_key"):
        raise InfinoError(0, f"no api_key in the trial response: {str(out)[:200]}")
    return out


def _loads_from(record: dict, prefix: str) -> bool:
    """Whether a load record is of the load from the upload at `prefix`: its
    request's source prefix ends with it, trailing slashes aside (the platform
    may store it resolved under the database's root), or it names none."""
    request = record.get("request")
    source = request.get("source") if isinstance(request, dict) else None
    named = source.get("prefix") if isinstance(source, dict) else None
    return not isinstance(named, str) or named.rstrip("/").endswith(prefix.rstrip("/"))


def _q(segment: str) -> str:
    return urllib.parse.quote(segment, safe="")


def _retry_after(header: Optional[str]) -> float:
    try:
        wait = float(header) if header else DEFAULT_RETRY_AFTER_S
    except ValueError:
        wait = DEFAULT_RETRY_AFTER_S
    return max(0.0, min(wait, MAX_RETRY_WAIT_S))


def _rerank(question: Optional[str], k: int) -> dict:
    """The `rerank` a search names to have its rows graded for `question` by
    the platform's triage pass, the pass the ask loop ranks with: the search
    fetches the deployment's triage pool, the grader scores each row for the
    question, and the best `k` come back, best first. A deployment that runs
    no triage pass refuses it (`RERANK_UNAVAILABLE_STATUSES`). Nothing to add
    without a question."""
    if not question:
        return {}
    return {"rerank": {"question": question, "k": k}}


# The statuses a search with `rerank` answers on a deployment that cannot
# grade: 501 from a gateway without the triage pass; 400 from a worker built
# without it (its message says "rerank is not available"), or from one older
# than the field, whose request type refuses it as an unknown field.
RERANK_UNAVAILABLE_STATUSES = (400, 501)
RERANK_UNAVAILABLE_WORDS = ("rerank is not available", "unknown field `rerank`")


def rerank_unavailable(error: "InfinoError") -> bool:
    """Whether a search's refusal says the deployment cannot rerank at all,
    rather than that this one request failed."""
    message = error.message.lower()
    return error.status == 501 or (error.status in RERANK_UNAVAILABLE_STATUSES
                                   and any(words in message for words in RERANK_UNAVAILABLE_WORDS))


def _rows(out: Any) -> list[dict]:
    """Search responses come back as `{"data": [...]}` when asked for JSON,
    and as `{"rows": [...], "rerank": {...}}` with a `rerank`."""
    if isinstance(out, dict):
        out = out.get("rows", out.get("data", []))
    return list(out or [])
