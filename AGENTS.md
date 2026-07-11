## General Instructions
- use Context7 when you are not sure or need official implementation details; if Context7 is unavailable, use the closest official upstream documentation source
- refer `.opencode/rules/git.md` for git related instructions

## uv commands
- use `uv` instead of `pip`
- use `uv add` to add dependencies
- use `uv run` for Python scripts, tests, and CLIs instead of calling `python` directly

## Python Best Practices 
- do not use `dataclass` decorator
- do not use Ruff for linting or formatting
- use keyword-only arguments for functions/methods with multiple parameters: put a bare `*` after `self` (or after positional-only args), then name every remaining parameter so callers must pass them by keyword (e.g. `def append_step(self, *, obs: ..., action: ...) -> None`)

## Git Commits
- use conventional commits format: `<type>: <short description>`
- common types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`
