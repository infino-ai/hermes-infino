# Durable recall for Hermes fleets

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that stores the agent's memory and history in Infino.

It saves four things: memory, conversations, traces and skills. Before each turn it gives the agent the memories that matter for that turn. The agent can also search everything with the Infino tools.

You don't need an account. Without one, everything stays in a database on your machine. When you sign in, everything is copied to your team's database in Infino Cloud, and new data keeps going there.

```mermaid
flowchart LR
  H[Hermes agent] -- "memory, messages,<br/>model and tool calls" --> P[hermes-infino]
  P --> T[(Team database)]
  P --> M[(Personal database)]
  T -- "relevant memories,<br/>before each turn" --> H
  M -- "relevant memories,<br/>before each turn" --> H
  T -- "find, search, sql, read" --> C[Infino tools]
  C --> H
```

## What it saves

| Table | What goes in it |
| --- | --- |
| `memory` | Everything written to Hermes's `MEMORY.md` and `USER.md`, and everything saved with the `infino_remember` tool. `infino_forget` deletes a memory. |
| `conversations` | Every message in every session, and who sent it. Tool calls and tool results are in their own columns. It is a log: when context compression shortens a session, or `/undo` takes a turn back, what was already said stays. |
| `traces` | Every model call (model, provider, tokens, time taken) and every tool call (name, arguments, result, status, time taken). Secrets in arguments and results are masked the way Hermes masks what it hands any memory plugin. |
| `skills` | Skills shared with `infino_publish_skill`. Once signed in, every profile on the team can use them, as read-only `infino:<name>` skills whose description says who published them. `INFINO_TEAM_SKILLS=off` keeps a profile to its own skills. |

A conversation turn and its traces share a `session_id` and `turn_id`, so you can join them. Every row has a `row_id`.

Memories name the person they are about ("Dana prefers…"), never "the user", because teammates read them too. When recalled memories are added to a turn, they are marked as reference material, not as instructions for the agent.

## Team memory and personal memory

Memory belongs to the team unless it is about one person. Nobody has to say "team": the agent decides from what the memory is about.

| Goes to team memory | Goes to personal memory |
| --- | --- |
| Decisions, conventions, fixes, who owns what, how things are set up | A person's preferences, credentials, schedule, private matters |
| Hermes's `MEMORY.md` | Hermes's `USER.md` |
| `infino_remember` (the default) | `infino_remember` when the agent marks it personal |

```mermaid
flowchart TD
  W[New memory] --> Q{Is it about<br/>one person?}
  Q -- no --> T[(Team database<br/>every agent on the team can read it)]
  Q -- yes --> P[(Personal database<br/>only your own recall reads it)]
```

Personal memory lives in a separate database. The Infino tools never open it, so no other agent and no chat can read it. On your machine it is `$HERMES_HOME/infino/personal`. When you sign in it goes to your personal database (`--personal-database`). If you don't have one, it stays on your machine. It never goes to the team database: when the agent saves a personal memory, the tool call is still logged in `conversations` and `traces`, but with the memory's text left out.

What is said in a conversation is the conversation: `conversations` keeps every message as it was written, including anything personal someone says there.

On a gateway, where one profile answers several people, each person's personal memory is saved under their name and recalled only in their own chats with the agent.

## Group chats

When Hermes is in a group chat through the gateway, memory belongs to that chat:

- What members ask the agent to remember is saved for that chat and credited to the person who said it. It never goes into anyone's personal memory.
- Recall in the group shows that chat's notes and team memory. It does not show anyone's personal memory or other chats' notes.
- Recall anywhere else (a direct message, another group) does not show that group's notes.

```mermaid
flowchart LR
  subgraph TD1 [Team database]
    TM[Team memory]
    OPS[Notes from the ops chat]
    SALES[Notes from the sales chat]
  end
  PM[(Dana's personal memory)]
  R1[Recall in the ops chat] --> TM
  R1 --> OPS
  R2[Recall in Dana's DM] --> TM
  R2 --> PM
```

Which chat a message comes from, and whether that chat is a group, comes from Hermes. With Hermes's channel memory, its setting for the chat decides, including whether someone may write to the owner's personal memory from a group. Without channel memory, every chat except a direct message counts as a group.

**The tools are not separated by chat.** Recall keeps each chat's notes to that chat, but the Infino tools search the whole team database, so an agent can find another chat's notes and conversations with them. Separating chats in the tools needs per-chat permissions, which this version does not have.

## On your machine, and in the cloud

```mermaid
flowchart LR
  H[Hermes] -- every write --> L[(Local database<br/>on your machine)]
  L -- "sync: a few seconds after each write,<br/>and at the end of each session<br/>(only after you sign in)" --> C[(Team database<br/>in Infino Cloud)]
```

**Without an account**, everything is stored under `$HERMES_HOME/infino/` using the Infino engine. Each memory is embedded on your machine with all-MiniLM-L6-v2 (384 dimensions). Recall uses both keyword search and vector search. Nothing leaves your machine.

**After you sign in**:
- Every row is still written to your machine first. A sync then sends it to the cloud.
- The sync sends rows in the order they were saved and picks up where it stopped. If the cloud is down, nothing is lost, and a batch whose upload was cut off is checked against the cloud before it is sent again. A row the cloud refuses outright stays on your machine and is listed by `hermes infino status`, which also shows how far behind the cloud is, and why the last sync stopped if it did.
- A forgotten memory is deleted from the cloud too.
- The cloud embeds memory itself, so your machine stops embedding.
- Recall reads the cloud first and adds any rows on your machine that the cloud doesn't have yet. If the cloud can't be reached or is slow to answer, recall searches your machine's own rows by keyword.

## The Infino tools

Hermes runs the Infino client as an MCP server, started by `hermes infino mcp` (from `mcp_servers.infino` in `config.yaml`). Each time it starts, it checks whether you are signed in:
- **Not signed in:** the tools (`find`, `search`, `sql`, `read`) search the database on your machine.
- **Signed in:** the same tools search the team's database in the cloud, plus anything else the cloud offers.

The API key is passed to the client in its environment, never on its command line. Hermes does not show an MCP server's instructions to the model, so the plugin adds them to the system prompt.

## Install

These steps were run as written on a fresh machine with Hermes's own installer.

**Before you start:**
- Hermes, installed with its installer: `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash`.
- Access to the two private repositories, `infino-ai/hermes-infino` and `infino-ai/ask-bench`. Hermes clones the plugin with your GitHub sign-in: run `gh auth login` first, or put `GITHUB_TOKEN=<a token that can read the repos>` in `~/.hermes/.env`.

**1. Build the Infino client.** It needs Node.js 22.19 or newer; Hermes's installer already put one in `~/.hermes/tools/`.

```bash
git clone -b feat/local-database-mode https://github.com/infino-ai/ask-bench.git
cd ask-bench/client
export PATH="$(ls -d ~/.hermes/tools/node-*/bin | head -n 1):$PATH"
npm ci && npm run build
```

This makes `ask-bench/client/dist/cli.js`.

**2. Install the plugin.** Hermes installs its Python dependencies and makes it the memory provider (`memory.provider: infino`):

```bash
hermes plugins install https://github.com/infino-ai/hermes-infino/tree/main/hermes_infino --enable --yes-deps
```

**3. Add the Infino tools,** with the full path to the client you built in step 1:

```bash
hermes config set mcp_servers.infino.command hermes
hermes config set mcp_servers.infino.args '["infino", "mcp"]'
hermes config set mcp_servers.infino.env.INFINO_CLIENT /full/path/to/ask-bench/client/dist/cli.js
hermes config set mcp_servers.infino.connect_timeout 120
```

Hermes starts the tools with `hermes infino mcp`, inside its own environment, where the plugin's dependencies are, and on the Node its installer keeps (`INFINO_NODE` in the same `env` block picks another).

**4. Use it.** Start `hermes` as usual. Everything is kept on your machine until you sign in. The first start downloads the embedding model from Hugging Face in the background (about 90 MB, once); memories saved before it is there are found by keyword, and embedded at the next start.

## Signing in

```bash
hermes infino login                          # create a free Infino account
hermes infino login --api-key-file KEY \
  --team-database TEAM [--personal-database ME]   # join a team with a key someone gave you
hermes infino status                         # how far behind the cloud is
hermes infino sync                           # send what the cloud doesn't have yet, now
```

To use a different Infino platform, such as staging, add `--url`:

```bash
hermes infino login --url https://api.platform.staging.infino.ai --api-key-file KEY --team-database TEAM --yes
```

Before creating anything or sending anything, `login` tells you exactly what will be uploaded and asks you to confirm. Only a person at a terminal can agree to create an account: with no terminal, it refuses, and no flag can answer for you. When you sign in with a key someone gave you, `--yes` confirms the upload. The key is saved in `$HERMES_HOME/infino/credentials.json`, which only you can read.

A Hermes that is already running picks up the sign-in within half a minute: recall and writes go to the cloud from then on. The Infino tools switch to the cloud when Hermes restarts. Until then they search the copy on your machine, which has everything too.

## Fleets: settings from config or Remote Config

A fleet doesn't run `login` on each machine. It puts the team's settings in the `env` block of the Infino tools' entry in `config.yaml`, and the plugin and its tools both read them from there:

```yaml
memory:
  provider: infino
mcp_servers:
  infino:
    command: hermes
    args: [infino, mcp]
    env:
      INFINO_CLIENT: /path/to/client/dist/cli.js
      INFINO_TEAM_DATABASE: support-team
      INFINO_API_KEY: ${INFINO_API_KEY}    # a reference; the key itself is in the profile's .env
    connect_timeout: 120
```

With Hermes's [Remote Config](https://github.com/NousResearch/hermes-agent/pull/122509), the same keys come from the config service instead of a file, so an administrator sets the team database once, at the team (group) level, and every agent on the team starts with it. Remote Config refuses a key written out in full, so `INFINO_API_KEY` stays a `${INFINO_API_KEY}` reference, and the key goes in each profile's `.env`. `hermes infino status` says where the settings came from ("from the config").

```mermaid
flowchart LR
  RC[Remote Config<br/>team level] -- "memory.provider: infino<br/>mcp_servers.infino.env" --> A[Each agent's Hermes]
  ENV[".env: INFINO_API_KEY"] -- "resolves ${INFINO_API_KEY}" --> A
  A --> P[hermes-infino]
  A --> C[Infino tools]
  P --> T[(Team database)]
  C --> T
```

## Settings

Settings are read from three places; the first that has a value wins:
1. the `env` block of `mcp_servers.infino` in the profile's config (a file or Remote Config);
2. the profile's environment (`.env`);
3. the sign-in `hermes infino login` saved.

The config block comes first because Hermes starts the Infino tools with that block as their environment, so the plugin and its tools always use the same team. A `${VAR}` reference whose variable isn't set counts as not set.

| Variable | What it is |
| --- | --- |
| `INFINO_API_KEY` | The Infino API key for this person or bot |
| `INFINO_TEAM_DATABASE` | The team's database |
| `INFINO_PERSONAL_DATABASE` | This person's database. Leave it unset for bots and shared profiles. |
| `INFINO_AUTHOR` | Who this profile's messages and memories are from, when Hermes doesn't say. In a chat, the sender is always the author, and a message with no sender is saved without one. |
| `INFINO_MCP_SERVER` | The client's name under `mcp_servers` (default `infino`). Environment only, since it names the config block. |
| `INFINO_URL` | The Infino Cloud URL (default `https://api.platform.infino.ws`) |
| `INFINO_TEAM_SKILLS` | `off` stops this profile loading the skills teammates publish (default on) |

## Development

From a Hermes checkout, with its environment activated (`source ./activate`) and pytest and the plugin's dependencies installed in it:

```bash
PYTHONPATH=$PWD:$PYTHONPATH python -m pytest tests
```

The tests use Hermes's own `MemoryManager`, the real Infino engine for the local database (with a stand-in for the embedding model), and a fake Infino Cloud.
