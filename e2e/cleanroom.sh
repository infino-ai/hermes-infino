#!/usr/bin/env bash
# A first install of hermes-infino from the README, at a given commit, in a
# throwaway HOME, then real Hermes sessions that exercise kept (superseded)
# memories and recall. Each check prints PASS or FAIL; the run's files stay
# under e2e/runs/.
#
#   OPENROUTER_API_KEY=... [INFINO_API_KEY=... INFINO_URL=... INFINO_TEAM_DATABASE=...] \
#     e2e/cleanroom.sh <plugin commit, pushed to GitHub>
#
# The model is reached through OpenRouter. With INFINO_API_KEY, the last step
# signs in to that deployment and team database (it must exist, or the key
# must be able to create it) and checks the cloud's rows and recall there.
# GitHub access is the caller's own: `gh auth login` and `gh auth setup-git`
# (GH_CONFIG_DIR and the git config are kept from the real HOME). Keys are
# passed in the environment and never printed.
set -u
exec 2>&1
SHA=$1
S=$(cd "$(dirname "$0")" && pwd)
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY: the sessions reach their model through OpenRouter}"
RUN=$(date -u +%Y%m%d%H%M%S)
CR=$S/runs/$RUN
mkdir -p "$CR/home" "$CR/work"
# GitHub access through the caller's own gh sign-in, never a token on a command line.
export GH_CONFIG_DIR=${GH_CONFIG_DIR:-$HOME/.config/gh} GIT_CONFIG_GLOBAL=${GIT_CONFIG_GLOBAL:-$HOME/.gitconfig}
export GIT_TERMINAL_PROMPT=0
export HOME=$CR/home
step() { echo; echo "=== $*"; }
ok() { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; }
model() { "$@"; }  # OPENROUTER_API_KEY is in the environment
URL=${INFINO_URL:-}
# A team database may be reused across runs: this run's memories carry a codename of
# their own so older rows never match.
DB=${INFINO_TEAM_DATABASE:-hermes-e2e}
CODE=Kestrel${RUN: -4}
echo "run $CR, plugin commit $SHA"

step "1. Hermes, by its installer"
curl -fsSL https://hermes-agent.nousresearch.com/install.sh -o "$CR/install.sh"
bash "$CR/install.sh" --non-interactive --skip-browser > "$CR/install.log" 2>&1; echo "installer exit=$?"
HERMES=$(ls "$HOME"/.local/bin/hermes 2>/dev/null || command -v hermes)
[ -x "$HERMES" ] && ok "hermes: $("$HERMES" --version 2>&1 | head -n 1)" || { bad "no hermes"; tail -n 30 "$CR/install.log"; exit 1; }
export PATH=$HOME/.local/bin:$PATH
HH=$HOME/.hermes
NODE_BIN=$(ls -d "$HH"/tools/node-*/bin 2>/dev/null | head -n 1)
echo "node $NODE_BIN"

step "2. the Infino client (README step 1)"
cd "$CR/work"
git clone -q -b feat/local-database-mode https://github.com/infino-ai/ask-bench.git && ok "cloned" || { bad "clone"; exit 1; }
( cd ask-bench/client && PATH=$NODE_BIN:$PATH npm ci > "$CR/npm-ci.log" 2>&1 && PATH=$NODE_BIN:$PATH npm run build > "$CR/npm-build.log" 2>&1 )
CLIENT=$CR/work/ask-bench/client/dist/cli.js
[ -f "$CLIENT" ] && ok "built the client" || { bad "no client"; tail -n 20 "$CR/npm-build.log"; exit 1; }

step "3. the plugin at $SHA (README step 2)"
"$HERMES" plugins install https://github.com/infino-ai/hermes-infino/tree/main/hermes_infino --enable --yes-deps --ref "$SHA" > "$CR/plugin-install.log" 2>&1
echo "install exit=$?"; tail -n 8 "$CR/plugin-install.log"
# The Python Hermes runs plugins with: the one under its home that has the plugin's dependencies.
HPY=""
for p in $(find "$HH" -path '*/bin/python3*' \( -type f -o -type l \) 2>/dev/null); do
  "$p" -c 'import infino, pyarrow' > /dev/null 2>&1 && { HPY=$p; break; }
done
[ -n "$HPY" ] && ok "Hermes's Python with the plugin's dependencies: $HPY" || { bad "no Python with the plugin's dependencies"; exit 1; }
grep -q superseded_ms "$HH/plugins/infino/tables.py" && ok "the installed plugin is the branch (keeps superseded memories)" || { bad "installed plugin lacks the change"; exit 1; }
echo "memory.provider = $("$HERMES" config get memory.provider 2>&1 | tail -n 1)"

step "4. config (README step 3) and a model"
"$HERMES" config set model.default anthropic/claude-sonnet-5.5 > /dev/null 2>&1
"$HERMES" config set model.provider openrouter > /dev/null 2>&1
"$HERMES" config set mcp_servers.infino.command hermes > /dev/null 2>&1
"$HERMES" config set mcp_servers.infino.args '["infino", "mcp"]' > /dev/null 2>&1
"$HERMES" config set mcp_servers.infino.env.INFINO_CLIENT "$CLIENT" > /dev/null 2>&1
"$HERMES" config set mcp_servers.infino.connect_timeout 120 > /dev/null 2>&1
"$HERMES" config get mcp_servers.infino 2>&1 | tail -n 6
mem() { "$HPY" "$S/memory_rows.py" "$HH"; }
memfile() { cat "$HH/memories/MEMORY.md" 2>/dev/null; }
check() { "$HPY" - "$@" <<'PY'
import json, subprocess, sys
want = sys.argv[1]
rows = json.loads(subprocess.run([sys.argv[2], sys.argv[3], sys.argv[4]], capture_output=True, text=True).stdout)
team = rows["team"] if isinstance(rows["team"], list) else []
cur = [r for r in team if r["superseded_ms"] is None]
old = [r for r in team if r["superseded_ms"] is not None]
print("current: ", [r["content"] for r in cur])
print("superseded:", [(r["content"], r["superseded_by"]) for r in old])
ok = eval(want, {"cur": cur, "old": old, "any": any, "all": all, "len": len})
print(("PASS: " if ok else "FAIL: ") + want)
PY
}

step "5. local mode: the agent saves a memory"
cd "$CR/work"
model "$HERMES" chat -Q --max-turns 8 -q "Save this to your memory with your memory tool: our production deploys go out on Tuesdays at 10:00 UTC, staging first." | tail -n 3
sleep 3; echo "--- MEMORY.md:"; memfile
check "any('Tuesday' in r['content'] for r in cur)" "$HPY" "$S/memory_rows.py" "$HH"

step "6. the agent replaces it: the old entry is kept, superseded"
model "$HERMES" chat -Q --max-turns 8 -q "Correction: production deploys moved to Thursdays at 14:00 UTC. Use your memory tool to replace the old deploy entry (don't add a second one)." | tail -n 3
sleep 3; echo "--- MEMORY.md:"; memfile
check "any('Thursday' in r['content'] for r in cur) and not any('Tuesday' in r['content'] for r in cur) and any('Tuesday' in r['content'] and 'Thursday' in (r['superseded_by'] or '') for r in old)" "$HPY" "$S/memory_rows.py" "$HH"

step "7a. the recall block the plugin builds for the question"
BLOCK=$("$HPY" "$S/recall_block.py" "$HH" "what day do production deploys go out?")
echo "$BLOCK"
echo "$BLOCK" | grep -q "Tuesdays.*\[superseded .*replaced by: .*Thursdays" && ok "recall shows the replaced entry, marked, naming its replacement" || bad "recall lacks the superseded entry"

step "7. recall shows both, the old one marked, before the model answers"
OUT=$(model "$HERMES" chat -Q --max-turns 4 -q "Without using any tools: what day did production deploys use to go out, and what day do they go out now?" | tail -n 6)
echo "$OUT"
echo "$OUT" | grep -qi tuesday && echo "$OUT" | grep -qi thursday && ok "the answer names both the old and the current day" || bad "the answer does not name both days"
"$HPY" - "$HH/state.db" <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
found = []
for (t,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    for c in [r[1] for r in db.execute(f"PRAGMA table_info({t})")]:
        try:
            for (v,) in db.execute(f"SELECT {c} FROM {t} WHERE CAST({c} AS TEXT) LIKE '%[superseded %'"):
                found.append(str(v))
        except Exception:
            pass
if found:
    text = found[-1]
    i = text.find("## Relevant memory (Infino)")
    print(text[i:i + 900] if i >= 0 else text[:900])
    print("PASS: a recall block with a superseded memory reached the model")
else:
    print("INFO: no stored message carries the recall block (Hermes may not persist injected context)")
PY

step "8. a hand edit drops the entry: kept, superseded, nothing named as its replacement"
"$HPY" - "$HH/memories/MEMORY.md" <<'PY'
import sys
p = sys.argv[1]
SEP = "\n§\n"
entries = [e for e in open(p).read().split(SEP) if e.strip()]
kept = [e for e in entries if "Thursday" not in e]
open(p, "w").write(SEP.join(kept))
print(f"removed {len(entries) - len(kept)} entry by hand")
PY
model "$HERMES" chat -Q --max-turns 2 -q "Say hi." | tail -n 1
sleep 3
check "not any('Thursday' in r['content'] for r in cur) and any('Thursday' in r['content'] and (r['superseded_by'] or '') == '' for r in old)" "$HPY" "$S/memory_rows.py" "$HH"

step "9. infino_forget deletes every copy"
EXACT=$("$HPY" "$S/memory_rows.py" "$HH" | "$HPY" -c "import json,sys; t=json.load(sys.stdin)['team']; print(next(r['content'] for r in t if 'Tuesday' in r['content']))")
echo "forgetting: $EXACT"
model "$HERMES" chat -Q --max-turns 6 -q "Call the infino_forget tool once with content exactly: $EXACT" | tail -n 3
sleep 3
check "not any('Tuesday' in r['content'] for r in cur + old)" "$HPY" "$S/memory_rows.py" "$HH"

if [ -z "${INFINO_API_KEY:-}" ] || [ -z "$URL" ]; then
  echo; echo "=== 10. skipped: no INFINO_API_KEY and INFINO_URL to sign in with"
else
step "10. sign in: the superseded copies go to the cloud"
model "$HERMES" chat -Q --max-turns 8 -q "Save this to your memory with your memory tool: team $CODE's on-call rotation changes on Mondays." | tail -n 1
model "$HERMES" chat -Q --max-turns 8 -q "Correction: team $CODE's on-call rotation changes on Wednesdays now. Replace the old entry with your memory tool." | tail -n 1
( umask 077; printf '%s' "$INFINO_API_KEY" > "$CR/infino.key" )
"$HERMES" infino login --url "$URL" --api-key-file "$CR/infino.key" --team-database "$DB" --yes 2>&1 | tail -n 4
sleep 15
"$HERMES" infino status 2>&1 | tail -n 8
env HH="$HH" DB="$DB" URL="$URL" CODE="$CODE" "$HPY" - <<'PY'
import importlib.util, os, sys, time
from pathlib import Path
pkg = Path(os.environ["HH"]) / "plugins" / "infino"
spec = importlib.util.spec_from_file_location("hi", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
m = importlib.util.module_from_spec(spec); sys.modules["hi"] = m; spec.loader.exec_module(m)
from hi import recall, tables
from hi.client import InfinoClient, InfinoError
c = InfinoClient(os.environ["INFINO_API_KEY"], os.environ["URL"], timeout=30)
db = os.environ["DB"]
code = os.environ["CODE"]
rows = c.query_sql(db, f"SELECT content, {tables.SUPERSEDED}, {tables.SUPERSEDED_BY} FROM memory "
                       f"WHERE content LIKE '%{code}%'")
cur = [r for r in rows if r.get(tables.SUPERSEDED) is None]
old = [r for r in rows if r.get(tables.SUPERSEDED) is not None]
print("cloud current:", [r["content"] for r in cur])
print("cloud superseded:", [(r["content"], r.get(tables.SUPERSEDED_BY)) for r in old])
print(("PASS" if any("Monday" in r["content"] for r in old) and any("Wednesday" in r["content"] for r in cur) else "FAIL")
      + ": the cloud holds the current entry and the superseded one")
q = f"when does team {code}'s on-call rotation change?"
for label, rr in (("plain", None), ("ranked", q)):
    t0 = time.monotonic()
    try:
        got = c.hybrid_search(db, tables.MEMORY, q, text_field="content", vector_field="embedding", k=8,
                              projection=recall.PROJECTION, rerank=rr)
        print(f"{label} search: {(time.monotonic() - t0) * 1000:.0f} ms ->", [r["content"][:60] for r in got])
    except InfinoError as e:
        print(f"{label} search: HTTP {e.status}: {e.message[:200]}")
PY
echo "--- the recall block, signed in:"
"$HPY" "$S/recall_block.py" "$HH" "when does team $CODE's on-call rotation change?"
OUT=$(model "$HERMES" chat -Q --max-turns 4 -q "Without using any tools: when did team $CODE's on-call rotation use to change, and when does it change now?" | tail -n 4)
echo "$OUT"
echo "$OUT" | grep -qi monday && echo "$OUT" | grep -qi wednesday && ok "signed in, recall gave both the old and current day" || bad "signed in, the answer lacks a day"
grep -E "infino: .*(rank|recall)" "$HH/logs/agent.log" | tail -n 5
rm -f "$CR/infino.key"
fi
"$HERMES" gateway stop > /dev/null 2>&1; pkill -f "$CR/home" 2>/dev/null
echo; echo "=== done ($CR)"
