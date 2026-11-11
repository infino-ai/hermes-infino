"""Print the recall block the installed plugin builds for a question, from a
Hermes home: what the provider adds to the model's context before a turn.
usage: <hermes python> recall_block.py <HERMES_HOME> <question>"""
import importlib.util, os, sys
from pathlib import Path
hh = Path(sys.argv[1])
os.environ["HERMES_HOME"] = str(hh)
pkg = hh / "plugins" / "infino"
spec = importlib.util.spec_from_file_location("hi", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
m = importlib.util.module_from_spec(spec); sys.modules["hi"] = m; spec.loader.exec_module(m)
from hi import config
from hi.backend import Backend
block, n = Backend(config.load(hh)).recaller().recall(sys.argv[2])
print(block or "(nothing recalled)")
