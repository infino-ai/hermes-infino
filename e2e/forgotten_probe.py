"""Measure, against a real Infino platform, what the `forgotten` table and
the lost-table check rely on, in a database of your choosing:

1. Whether the platform tells a table dropped and made again from one whose
   rows were all deleted: `list_tables` and the schema describe (default and
   `document` encoding) of one table, after its rows are deleted, and after
   it is dropped and made again.
2. That `forgotten` is made with the plugin's schema and no index, takes an
   append, and answers the plugin's lookup of ids in it.
3. What a query of a table that does not exist answers.

Usage (INFINO_API_KEY and INFINO_URL in the environment):

    python e2e/forgotten_probe.py <database>

It makes two tables, `forgotten_probe_rows` and `forgotten_probe_log`, in
that database, and drops both before it ends.
"""

import json
import os
import sys
import uuid

from hermes_infino import tables
from hermes_infino.client import InfinoClient, InfinoError

ROWS = "forgotten_probe_rows"
LOG = "forgotten_probe_log"


def describe(c: InfinoClient, db: str, table: str) -> dict:
    plain = c._request("POST", f"/v1/schema/{db}", body={"table_name": table})
    document = c._request("POST", f"/v1/schema/{db}", body={"table_name": table, "encoding": "document"})
    return {"default": plain, "document": document}


def main() -> None:
    db = sys.argv[1]
    c = InfinoClient(os.environ["INFINO_API_KEY"], os.environ["INFINO_URL"], timeout=30)
    out: dict = {}
    for t in (ROWS, LOG):
        if t in c.list_tables(db):
            c.drop_table(db, t)
    try:
        schema = [f for f in tables.MEMORY_SCHEMA if f["type"] != "embedding"]
        c.ensure_table(db, ROWS, schema, tables.MEMORY_INDEXES)
        ids = [str(uuid.uuid4()) for _ in range(3)]
        c.append(db, ROWS, [{tables.ROW_ID: i, tables.STORED: 1, "content": f"probe {n}"} for n, i in enumerate(ids)])
        out["1_list_tables_raw"] = c._request("POST", f"/v1/list_tables/{db}")
        out["1_describe_with_rows"] = describe(c, db, ROWS)
        c.delete(db, ROWS, f"{tables.ROW_ID} IN ({', '.join(map(tables.sql_string, ids))})")
        out["1_count_after_delete"] = c.query_sql(db, f"SELECT count(*) AS n FROM {ROWS}")
        out["1_describe_after_delete"] = describe(c, db, ROWS)
        c.drop_table(db, ROWS)
        c.ensure_table(db, ROWS, schema, tables.MEMORY_INDEXES)
        out["1_describe_after_drop_and_make"] = describe(c, db, ROWS)

        c.ensure_table(db, LOG, tables.FORGOTTEN_SCHEMA, tables.FORGOTTEN_INDEXES)
        out["2_log_columns"] = c.column_names(db, LOG)
        c.append(db, LOG, [{tables.FORGOTTEN_FROM: tables.MEMORY, tables.ROW_ID: i, tables.FORGOTTEN_MS: 1}
                           for i in ids[:2]])
        found = c.query_sql(db, f"SELECT {tables.ROW_ID} FROM {LOG} WHERE {tables.FORGOTTEN_FROM} = "
                                f"{tables.sql_string(tables.MEMORY)} AND {tables.ROW_ID} IN "
                                f"({', '.join(map(tables.sql_string, ids))})")
        out["2_lookup_found_expected"] = sorted(r[tables.ROW_ID] for r in found) == sorted(ids[:2])

        try:
            c.query_sql(db, f"SELECT {tables.ROW_ID} FROM forgotten_probe_absent WHERE {tables.ROW_ID} = 'x'")
            out["3_missing_table_query"] = "answered 200"
        except InfinoError as e:
            out["3_missing_table_query"] = {"status": e.status, "message": e.message[:300]}
    finally:
        for t in (ROWS, LOG):
            if t in c.list_tables(db):
                c.drop_table(db, t)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
