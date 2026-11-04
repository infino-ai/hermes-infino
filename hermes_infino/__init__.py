"""hermes-infino: Infino as the backend for Hermes Agent.

A directory memory-provider plugin (installed as `$HERMES_HOME/plugins/infino/`,
selected with `memory.provider: infino`). Besides the provider, it registers the
trace hooks, which run while the provider is active, and the team's shared
skills.
"""


def register(ctx) -> None:
    from .provider import InfinoMemoryProvider
    from .skills import register_team_skills
    from .traces import register_hooks

    ctx.register_memory_provider(InfinoMemoryProvider())  # a new instance per load
    register_hooks(ctx)
    register_team_skills(ctx)
