## Important Instructions
- **code readability** is paramount, do not compromise it for the sake of brevity or complexity

## General Instructions
- use Context7 when you are not sure or need official implementation details; if Context7 is unavailable, use the closest official upstream documentation source
- refer `.opencode/rules/git.md` for git related instructions

## AWS Administration
- when an application profile lacks permission to update its own IAM policy, use `aws login --profile <admin-profile> --region <region>` only after the user explicitly authorizes browser-based authentication
- verify the authenticated principal with `aws sts get-caller-identity --profile <admin-profile>` before making changes
- apply the narrowest required IAM policy from a reviewed JSON file; never place credentials, account-specific tokens, or browser-login URLs in repository files
- verify the resulting S3 operations using the workload profile, not the administrative profile
- run `aws logout --profile <admin-profile>` after verification so administrative browser credentials do not remain active


## uv commands
- use `uv` instead of `pip`
- use `uv add` to add dependencies
- use `uv run` for Python scripts, tests, and CLIs instead of calling `python` directly

## Python Best Practices 
- do not use `dataclass` decorator
- do not use Ruff for linting or formatting
- use keyword-only arguments for functions/methods with multiple parameters: put a bare `*` after `self` (or after positional-only args), then name every remaining parameter so callers must pass them by keyword (e.g. `def append_step(self, *, obs: ..., action: ...) -> None`)
- keep a function's return annotation on the `def` line when it fits (e.g. `def make_dagger_round_seeds(*, seed: int, rounds: int) -> list[int]:`); do not place it on a separate line

## Git Commits
- follow `.opencode/rules/git.md` for the full git policy; the rules below are the ones most often missed
- use conventional commits: `<type>: <short summary>` — one subject line only
- common types: `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `ci`, `build`, `perf`, `revert`
- do **not** add a commit body unless the user explicitly asks for one
- summary after `type:` must start with a lowercase letter, have no trailing period, and stay concise (about 6–12 words)
- use plain types only — no scopes (`feat(scope): ...` is wrong)
- prefer intent/outcome over implementation detail; avoid vague subjects like `update`, `changes`, `misc`
- before committing, check recent style with `git log --oneline -10` and match it
- if the message is wrong, fix it **before** pushing; never push then amend/force-push unless the user explicitly asks to rewrite published history

Examples:
- `feat: add viewer camera capture and recalibrate top-camera`
- `fix: prevent calibration overwrite unless toggle was pressed`
- `docs: clarify host-client branch sync workflow`
