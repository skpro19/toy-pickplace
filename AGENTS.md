## General Instructions
- use `uv` instead of `pip`
- use `uv run` for Python scripts, tests, and CLIs instead of calling `python` directly
- use Context7 when you are not sure or need official implementation details; if Context7 is unavailable, use the closest official upstream documentation source
- refer `.opencode/rules/git.md` for git related instructions

## Python Best Practices 
- do not use `dataclass` decorator
- use keyword-only arguments for functions/methods with multiple parameters: put a bare `*` after `self` (or after positional-only args), then name every remaining parameter so callers must pass them by keyword (e.g. `def append_step(self, *, obs: ..., action: ...) -> None`)