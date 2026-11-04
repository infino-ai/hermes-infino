# hermes-infino

Infino as the backend for [Hermes Agent](https://github.com/NousResearch/hermes-agent): everything an agent remembers, says, does and shares, kept in an Infino database, recalled before each turn and queryable with the Infino tools.

It works with no account: everything stays in an Infino database on this machine. Signing in (`hermes infino login`) pushes all of it to the team's Infino Cloud database, and from then on the cloud is caught up with every write. In the cloud the team shares one database, and the platform reads it into table cards and facts.

## What it keeps

The plugin, `infino`, is a Hermes memory provider. It keeps four tables:

| Table | What lands in it |
| --- | --- |
| `memory` | Hermes's built-in `MEMORY.md` / `USER.md` writes, and the `infino_remember` tool (team or personal). `infino_forget` deletes. |
| `conversations` | Every message of every session, with who asked. An assistant's tool calls and each tool's result are kept in their own columns. |
| `traces` | Every model call (model, provider, tokens, duration) and tool call (name, args, result, status, duration). |
| `skills` | Skills shared with `infino_publish_skill`. Signed in, every profile on the team loads them as read-only `infino:<name>` skills. |

`conversations` and `traces` share `session_id` and `turn_id`. Every row carries a `row_id`.

Before each turn the plugin recalls the most relevant memories into the context. The block says it is reference material written by people's agents, not instructions to follow. Memories name people, never "the user": teammates read them as written.

## Where it keeps them

**Without an account**, everything goes to the local database at `$HERMES_HOME/infino/db`, through the Infino engine's Python binding. Each memory is embedded on this machine with all-MiniLM-L6-v2 (384 dimensions, full precision), and recall fuses keyword and vector search. Nothing leaves the machine.

**Signed in**, every row is still written locally first, then pushed to the cloud by a sync that runs a moment after each write and at the end of each session:
- The sync pushes rows in the order they were stored and resumes where it stopped, so an outage or a refused push loses nothing and duplicates nothing. `hermes infino status` shows how far behind the cloud is and why a sync stopped.
- Forgotten memories are deleted in the cloud by `row_id`.
- The cloud embeds memory with its own model, so the local copy stops embedding.
- Recall reads the cloud first and adds this machine's rows the cloud does not have yet. While the cloud cannot be reached, recall reads the local copy: this machine's own rows, by keyword.

## The Infino tools

Hermes runs the Infino client as an MCP server through the plugin's launcher, `mcp_launch.py`. At each start the launcher reads the profile's sign-in and starts the client:
- **without an account**, over the local database (`find`, `search`, `sql`, `read`);
- **signed in**, over the team's cloud database (the same tools, plus what the platform offers).

The API key reaches the client in its environment, never on its command line. Hermes's MCP client does not pass a server's instructions to the model, so the plugin adds them to the system prompt.

## Install

1. Build the Infino client (Node.js 22.19 or newer): `client/` of `infino-ai/ask-bench`, branch `feat/local-database-mode`.
2. Copy `hermes_infino/` to `$HERMES_HOME/plugins/infino/`. Hermes installs the plugin's Python dependencies from `plugin.yaml`; the engine is pinned to `infino==0.11.15`, the version the client reads the local database with.
3. In `config.yaml`:

```yaml
memory:
  provider: infino
mcp_servers:
  infino:
    command: /path/to/hermes/venv/bin/python   # the Python Hermes runs, where the dependencies are
    args: [/path/to/.hermes/plugins/infino/mcp_launch.py]
    env:
      HERMES_HOME: /path/to/.hermes
      INFINO_CLIENT: /path/to/client/dist/cli.js
      INFINO_NODE: /path/to/node          # optional; `node` on PATH otherwise
    connect_timeout: 120
tool_budget:
  mcp_result_size_chars: 60000   # the client returns up to 60,000 characters; Hermes's default is 50,000
```

The first memory written or searched downloads the embedding model from Hugging Face (about 90 MB, once).

## Signing in

```bash
hermes infino login                          # create a free Infino account
hermes infino login --api-key-file KEY \
  --team-database TEAM [--personal-database ME]   # join a team with a key you were given
hermes infino status
hermes infino sync                           # push what the cloud does not have yet, now
```

`login` says exactly what will be uploaded and asks at the terminal before anything is created or sent. Only the person at the terminal can agree to creating an account: without a terminal it refuses, and no flag answers for them. With a key someone gave you, `--yes` agrees to the upload. The key is stored in `$HERMES_HOME/infino/credentials.json`, readable by its owner only.

Recall reads the cloud from the next turn. The Infino tools read it once Hermes restarts; until then they read the local copy, which has everything too.

## Settings

A deployment can sign a profile in through its environment instead of `login`. A variable that is set wins over the sign-in file. Hermes passes an MCP server only some of its environment, so a deployment signed in this way also names `INFINO_API_KEY` and `INFINO_TEAM_DATABASE` in the `mcp_servers.infino.env` block.

| Variable | Meaning |
| --- | --- |
| `INFINO_API_KEY` | Infino API key for this person or bot |
| `INFINO_TEAM_DATABASE` | The team's database |
| `INFINO_PERSONAL_DATABASE` | This person's database; leave unset for bots and shared profiles |
| `INFINO_AUTHOR` | Who this profile's turns and memories are from on this machine, when Hermes does not say. On a chat channel the sender is always the author, and a turn that names nobody is kept unattributed. |
| `INFINO_MCP_SERVER` | The client's name under `mcp_servers` (default `infino`) |
| `INFINO_URL` | Platform URL (default `https://api.platform.infino.ws`) |

## Development

From a Hermes checkout's activated environment (`source ./activate`), with pytest and the plugin's dependencies installed in it:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python -m pytest tests
```

The tests run against Hermes's own `MemoryManager`, the real Infino engine for the local database (with a stand-in for the embedding model), and a fake platform for the cloud.
