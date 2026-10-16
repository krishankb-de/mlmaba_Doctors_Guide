# CLAUDE.md

Guidance for Claude Code in this repository.

## What this is

A research chat interface for chest X-ray report generation:
- `app/` holds the FastAPI server and the ES-module page;
- `hybrid_xmamba/` is the model package;
- `scripts/` holds the training, evaluation and test code and the cluster wrappers;
- `configs/` holds the Hydra configs;
- `tests/` holds the tests.

`app/README.md` is the runbook.

## Environment

Use the virtual environment `venv/` for every run and install:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt -r requirements_eval.txt -r app/requirements.txt -r requirements-e2e.txt
venv/bin/pip install -e .
```

## Checks (run before calling any change complete)

```bash
bash scripts/validate.sh                                    # must end "All gates passed."
venv/bin/python -m pytest tests/e2e -m e2e -q
venv/bin/python scripts/chat_ui_layout_check.py
venv/bin/python scripts/chat_ui_browser_check.py --no-evidence
venv/bin/python scripts/chat_ui_features_check.py --no-evidence
```

GitHub runs the same checks on every push, plus lint and the push guards (only `main`; each pushed commit passes the hook,
adds no local-only file and carries the noreply identity): `.github/workflows/checks.yml`.

When you add a module or a config key, add an assertion to `tests/test_willi_parity.py`.

## Rules

- **Data (MIMIC-CXR data use agreement):**
  - No MIMIC-derived byte is ever committed: no weights, gallery files, session DB, exports, logs, report text, study ids or test-split images.
  - The pre-commit hook (`bash scripts/install_hooks.sh`) refuses such files. Install it in every clone.
  - Read cluster logs only through `bash scripts/chat_remote.sh summary|state|queue`. Never copy raw logs to the laptop.
- **No plans, notes or secrets in the repository:**
  - Never commit a token, a credential or a filled-in env file.
  - Cluster settings (login host, user, paths) live in the git-ignored `scripts/chat_cluster.env`; `scripts/chat_cluster.env.example` shows the keys.
- **No public deployment.** The code may be on GitHub, but the app is not offered to anyone else:
  - never publish the GitHub Pages copy (`scripts/pages_build.py`, `scripts/pages_publish.sh`) or enable GitHub Pages;
  - never open a public link (`app/tunnel/public_demo.sh`);
  - never start a server in public mode, with a `BIND` other than `127.0.0.1`, or with `CORS_ORIGINS`;
  - never add deploy or hosting configuration. The one workflow, `.github/workflows/checks.yml`, only runs the checks on every
    push (no deploy, no secrets, no uploads).
  The server runs in private mode on its compute node's loopback, reached only through your own SSH tunnel; the laptop dev server
  listens on 127.0.0.1.
- **Ports:** 8000 is the laptop dev server and 8010 the tunnel to the cluster server. Tests start their own servers on free ports.
- **Defaults reproduce the published decoding protocol** (`app/schemas.py` `Options`). Keep them, and keep `scan_impl` and `tfla_impl` pinned explicitly in every model yaml.
- **Cluster:**
  - Changes are additive only: never delete or overwrite a cluster file or directory you did not create.
  - Write down a prediction before every job.
  - Never run two servers on one `CHAT_HOME`.

<!-- antislop:start -->
## antislop
For UI, copy, people, mobile layout, or code comments work, read these installed skill files directly (use these paths even if a same-named global skill exists):
- Core filter, always on: `antislop`: `.claude/skills/antislop/SKILL.md`
Before starting, follow the core's "Two Usage Modes" section in strict order: explicit session instruction first, then global preference, then ask. A session instruction always wins. For a resolved mode, say `antislop active: <mode> (session override).` or `antislop active: <mode> (global preference).` once before presenting findings or making edits, using the actual mode and source. Acknowledging the user's request without naming the source does not replace this notice.
Only an explicit choice of antislop during or after selects a session mode. A request to review, audit, or avoid file edits does not select a mode; read the global preference in that case. Another skill's mode does not select antislop's mode.
If the mode is unresolved, ask during/after and end the response; wait for the answer before any UI review, planning, or concept. For read-only tasks, put the active-mode notice only at the start of the final answer, never in progress messages. For editing tasks, announce before the first edit and omit it from the final answer.
To update antislop later: `npx antislop-ai --update`, or run `npx antislop-ai` and pick Overwrite them.
<!-- antislop:end -->
