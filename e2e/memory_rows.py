"""Print the plugin's local memory rows (team and personal) from a Hermes home,
using the installed plugin's own code. usage: <hermes python> memory_rows.py <HERMES_HOME> [filter words]"""
import importlib.util, json, sys
from pathlib import Path
hh = Path(sys.argv[1])
pkg = hh / "plugins" / "infino"
spec = importlib.util.spec_from_file_location("hi", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
m = importlib.util.module_from_spec(spec); sys.modules["hi"] = m; spec.loader.exec_module(m)
from hi import config, tables
from hi.local import LocalStore, PERSONAL_DB_DIR
cols = "content, origin, scope, row_id"
history_cols = f"content, origin, scope, {tables.SUPERSEDED}, {tables.SUPERSEDED_BY}, {tables.PRIOR_ROW_ID}, row_id"
out = {}
for name, folder in (("team", "db"), ("personal", PERSONAL_DB_DIR)):
    store = LocalStore.under(config.state_dir(hh), folder)
    try:
        out[name] = store.query_sql(f"SELECT {cols} FROM {tables.MEMORY} ORDER BY stored_ms")
        out[name + "_history"] = store.query_sql(f"SELECT {history_cols} FROM {tables.HISTORY} ORDER BY stored_ms")
    except Exception as e:
        out[name] = f"unreadable: {e}"
print(json.dumps(out, indent=1, default=str))
