# Chat server runbook

The chat app turns a chest X-ray into a report. It shows the report as it is written, the most similar training X-rays, the
matching training reports and CheXbert labels. Sessions are stored, replayed and exported.

The app runs as one SLURM job on the cluster. You reach it from the laptop through an SSH tunnel. Your cluster settings (the login
host, your user and the cluster paths) live in `scripts/chat_cluster.env`, which git ignores: copy `scripts/chat_cluster.env.example`
and fill it in. The commands below that name `$CLUSTER_HOST` read it after `source scripts/chat_cluster.env`. The job wrappers in
`scripts/` carry placeholders for the cluster's partition, account, QOS and the nodes a job must avoid (`@SLURM_PARTITION@` and
the like); `bash scripts/chat_remote.sh sync` fills them in on the cluster copy from the `SLURM_*` settings, so a wrapper submitted
there, through `chat_remote.sh submit` or by hand with `sbatch`, carries them.

## 1. The data rules (MIMIC-CXR DUA)

- **Everything MIMIC-derived stays on the cluster.** That covers:
  - the dataset, under `dataset/mimic_full`;
  - the checkpoints: `outputs/` of the thesis checkout, and `models/` in `CHAT_HOME`;
  - the server's home, `CHAT_HOME` (by default `/sc/home/$USER/chat_sessions/`): the gallery, the session database, uploads and logs.

  Never copy any of it to the laptop or to a synced folder (iCloud, Dropbox).
- **What keeps it out of git.**
  - `outputs/` is in `.gitignore`.
  - `results/` and `chat_sessions/` are not. On a machine that has them, only the pre-commit hook refuses them.
  - `bash scripts/install_hooks.sh` installs that hook (`scripts/check_no_restricted_files.sh`). It refuses `outputs/`, `results/`, `uploads/`, `chat_sessions/`, `logs/`, `*.ckpt`, `*.npy`, `*.parquet` and `*.db`, among others; the script's header has the full list.
- **Private mode** (the default) shows MIMIC-derived content to anyone who reaches the server.
  - It listens only on its compute node's loopback, so only processes on that node reach it: your tunnel, and any other job or user process on the same node. Compute nodes are shared.
  - If `${CHAT_HOME}/app_token` exists, the server requires the token in private mode too.
- **Public mode** shows nothing MIMIC-derived (`app/redact.py`):
  - retrieval is rank and similarity only;
  - there is no reference text and no score stage;
  - checkpoint paths are cut to file names;
  - error text the server did not write is replaced.
- **What reaches the laptop.** On a private server, a turn's stream and an export carry MIMIC-derived text: reports, retrieved reports and study ids. The browser and curl display them. Do not redirect a stream into a file you keep. Keep an export out of synced folders and out of git, and delete it when you are done.
- **Online AI services.** PhysioNet's guidance on credentialed data treats sending it to an online AI service as sharing it. Never paste a report, a study id or an image into one.
- **Cluster logs.** Read them only through `bash scripts/chat_remote.sh summary <log>`.
  - The summary shows fixed line shapes and masks long digit runs.
  - The job log carries no report text by design. The API's own stderr goes to `${CHAT_HOME}/logs/` on the cluster.

## 2. Start the server (cluster)

**Once per laptop.** Copy `scripts/chat_cluster.env.example` to `scripts/chat_cluster.env` and fill it in.

**Once per cluster account.**
- Run `bash scripts/chat_remote.sh sync` first, which creates the cluster checkout. Then run `bash scripts/chat_remote.sh submit scripts/chat_cluster_setup_h100.sh MAIN_REPO=<thesis checkout>`. It is additive: it creates symlinks and installs the web dependencies into overlay directories.
- The gallery is built once per image tower. See the headers of `scripts/build_retrieval_gallery_h100.sh` and `scripts/label_gallery_reports_cpu_h100.sh`.

**Every start**, from the laptop, in this repository:

```bash
bash scripts/chat_remote.sh sync                                      # rsync the tree to the cluster checkout (never --delete)
bash scripts/chat_remote.sh submit scripts/serve_chat_h100.sh MODELS=m3eos,m3 \
    'DRIFT_NOTE=CPU decode; matched the published GPU decode on 20/20 probe studies'
bash scripts/chat_remote.sh queue                                     # wait for state R
bash scripts/chat_remote.sh summary logs/chat_server_<jobid>.log      # the start-up lines below
```

The job runs on CPU with 8 cores and 32 GB, for up to 24 h, and is started with `--requeue`. It runs two processes:
- the CheXbert labeller, in its own venv;
- the API, `python -m app.server`.

Only one server can use a `CHAT_HOME` at a time, because the session database is locked by the server that opened it. Cancel the running job before you start another one on the same home.

A normal start-up summary looks like this:

```
=== sync <commit> clean ===
=== chat server: node=<node> mode=private bind=127.0.0.1 device=cpu gallery=g13d_m3_v1 job=<id> ===
=== labeller up ===
[server] starting: engine=real device=cpu mode=private threads=8 models=hybrid_150m_m3_rrg_eos,hybrid_150m_m3_rrg
[server] sqlite journal_mode=wal
[server] recovered 0 running turn(s)
[server] gallery: 191462 images, 194125 report rows, labels done
[server] retention: purged 0 session(s), 0 saved file(s) (private 45 d, public 7 d)
[server] serving: retrieval=on labels=on published=on
```

**Levers.** Pass them as `NAME=value` after the wrapper. They reach the job's environment.

| Lever | Meaning |
|---|---|
| `MODELS` | The models served; the first one is the default (`m3` if unset).<br>`m3`: `hybrid_150m_m3_rrg`, the published model.<br>`13d`: `hybrid_150m_v2_rrg`.<br>`m3eos`: `hybrid_150m_m3_rrg_eos`, trained to end its reports.<br>`MODELS=m3eos,m3` makes the EOS model the default and keeps the published model selectable.<br>Compare mode (section 4) needs a Mamba-3 model and `13d`, for example `MODELS=m3eos,m3,13d`. |
| `DRIFT_NOTE` | The CPU-versus-GPU drift sentence, shown on the published models' cards. On a CPU server, the EOS model's card says instead that its CPU decode was not probed against a GPU decode. |
| `MODE` | `private` (the default) or `public`. Public mode needs the token file (section 5). |
| `BIND` | The bind address; the default is `127.0.0.1`. Any other address needs the token file. |
| `GALLERY` | A gallery build directory. The default is `${CHAT_HOME}/gallery/g13d_m3_v1`; `none` serves without retrieval. |
| `RETENTION_DAYS` | Days a session is kept: a whole number from 1 to 3650. See section 7. |
| `CORS_ORIGINS` | Origins allowed to call the API from another site, comma-separated: `https://host[:port]`, or `http://localhost[:port]`. `CORS_ORIGINS=https://krishankb-de.github.io` serves the GitHub Pages copy (section 4). Unset: no cross-origin calls, as before. The log shows only how many origins. |
| `CHAT_HOME`, `LABELER_WAIT_S`, `PUBLISHED_MODEL`, `PUBLISHED_FLOOR` | See the header of `scripts/serve_chat_h100.sh`. A non-default `CHAT_HOME` is not found by the tunnel (section 3). |

For a GPU server, use `scripts/serve_chat_gpu_h100.sh`. It is the same job on one GPU (`DEVICE=cuda` by default), and its log is `logs/chat_server_gpu_<jobid>.log`. `DEVICE=cuda` on the CPU wrapper fails, because that job has no GPU.

**Refusals.** The EOS model needs its finished run in `CHAT_HOME`: the checkpoint, the `DONE` marker, and run metadata saying it was trained to end its reports. If any of these is missing, the API refuses to start with one ERROR line.

**Retention at every start.** Every start runs the retention sweep (section 7). It deletes:
- sessions older than their mode's retention days (7 for public, 45 for private, counted from creation), whether or not they were deleted in the page;
- upload directories that no live session owns;
- saved exports in `exports/` older than the private retention days, counted from the time in their file name.

## 3. Reach it from the laptop (tunnel)

```bash
bash app/tunnel/tunnel.sh                                          # then open http://localhost:8000
LOCAL_PORT=8010 bash app/tunnel/tunnel.sh                          # when a dev server already uses 8000
nohup env LOCAL_PORT=8010 bash app/tunnel/tunnel.sh > ~/chat_tunnel.log 2>&1 &   # in the background
curl -s localhost:8010/healthz                                     # status, mode, default_model, code_version, retention_days
```

How the tunnel works:
- The job writes where it serves to `chat_sessions/endpoint` in your cluster home once the API is up. The tunnel reads that file over ssh and forwards `localhost:LOCAL_PORT` to it. A server started with another `CHAT_HOME` writes its endpoint elsewhere, so the tunnel does not find it.
- While a forward runs, the tunnel reads the file again every `RECHECK_S` seconds (default 60). If the server moved (a requeue onto another node or port) or stopped, the forward is stopped at that re-read. A new forward starts once a server is back. No hand-work is needed.
- `VIA=jump` (the default) forwards to the node's loopback through the login node. A server with no `app_token` needs no token.
- `VIA=login` needs a server bound to another address, and so a token.
- Other settings:
  - `LOGIN`, `CLUSTER_USER`;
  - `RETRY_SLEEP` (default 15);
  - `MAX_ATTEMPTS`. With 0 (the default) the tunnel never stops. It counts attempts: one read of the endpoint and its forward. Re-reads are not counted, and a live forward is never ended by it.
- Exit codes: 1 if the local port is in use, 2 for a bad setting.
- On macOS, a listener bound to a wildcard address can make a used port look free.

## 4. Use it

Open `http://localhost:<LOCAL_PORT>`. If the server has a token, enter it in Settings → Access token; on a 401 the page tells you to.

**From the GitHub Pages copy.** Not offered now: this repository publishes no Pages copy, and the page is used only through your own tunnel. What follows records how the copy works. The same page can be served from `https://krishankb-de.github.io/hybrid_model_mamba_xlstm/`. That needs `scripts/pages_publish.sh` to have run, and Pages switched on once in the repository settings: Pages → Deploy from a branch → `gh-pages`, `/ (root)`. The copy holds the page's code only. Your browser talks straight to your tunnel, so no report, image or study ever goes to GitHub.
- **The origin.** Start the server with `CORS_ORIGINS=https://krishankb-de.github.io` (section 2). Without it, the browser blocks every request.
- **The API server.** Open the Pages URL, then set Settings → API server to `http://localhost:<LOCAL_PORT>` (your tunnel). Until it is set, the page sends nothing and says so.
- **The token.** Changing the API server clears the access token, so enter the token for the new server. A private server without `app_token` needs none.
- **Browsers.** Chrome may ask once to let the site reach devices on your local network: allow it. Safari may refuse a secure page's requests to `http://localhost`, so use Chrome or Firefox.
- **A shared origin.** Every Pages site of the GitHub account shares the origin `https://krishankb-de.github.io`, and with it the allow-list entry and the stored token. This is the account's only Pages site; keep third-party scripts off any other.
- **Republishing.** After a change to `app/static`, commit it, then run `bash scripts/pages_publish.sh` from a checkout of the branch with a clean `app/static`. It adds one commit to `gh-pages` and never force-pushes. GitHub caches pages for about 10 minutes.

- **The welcome page.** The page opens on Dr. Green's welcome page when its address names no chat. **Start reading** opens a new chat; **Continue your last case** goes back to the newest chat. A link to a chat (`#/s/<id>`, `#/new`) opens the chat directly. Dr. Green in the banner opens the welcome page again, and Escape closes it.
- **Start a turn.** Attach an X-ray by click, drag and drop, or paste. On a private server you can instead pick a test-split study. The stages run in order: preprocess → encode → retrieve → generate → label → score.
- **How a turn is shown** (Settings → Showing results; a setting of this browser, never sent to the server):
  - *When complete, top to bottom* (the default). From Send, the card is the reading card: the X-ray and the stages as they run. It stays until the turn has ended and the pause has passed (5 s from Send unless changed; 0 to 5 s). Then the card is shown in order: the report line by line, then the findings, the similar X-rays, the matching reports, the images and how it was made. **Show all now** skips the rest. An error or a Stop is shown at once. A chat opened from the sidebar is shown whole.
  - *Live, as the report is written.* The card is drawn from the first event, as before.
- **What a card shows**, top to bottom: the report, the findings (CheXbert labels), the similar X-rays, the matching reports, the images of the turn, and how it was made (stages and model). Click an image to open the full-screen viewer, with side by side, zoom, pan, brightness and contrast.
- **The drawer** holds the settings in groups: the model, decoding (beam, token budget), similar cases and findings, report text, showing results, and the connection (API server, access token). **Restore defaults** puts back the page's defaults; it leaves the model, the API server and the token as they are.
- **The page's defaults are not the published protocol.**
  - "Stop when the report starts repeating" and the display repair are both on.
  - Turn the stop switch off in Settings to decode as published. Only then does a test study's card compare the live report with the published one.
  - The repair changes only what is shown: repeated sentences and a cut-off ending are hidden. The stored report is the model's own output.
- **Text commands** change the settings of the turn they are sent with, on top of the drawer: `beam N`, `greedy`, `tokens N`, `retrieve N`, `reference: <text>`, `repair on|off`. A command sent without an image re-runs the chat's last X-ray with that change.
- **The two models.**
  - The EOS model ends its report where it was trained to.
  - The published model never ends by itself: it stops at the token budget, or at the first repeated sentence while the stop switch is on.
- **Compare mode.** On a server that serves a Mamba-3 model and 13D, Settings shows "Compare with <name>", off by default.
  - With it on, each X-ray turn is decoded by both models, one after the other. The card shows the two reports side by side, or stacked on a narrow window. Each has its own summary, labels, Copy and Show raw. The similar X-rays, the matching reports and the published comparison stay the first model's.
  - 13D has no decode cache, so a compare turn decodes both reports uncached. It takes longer than a normal turn on the CPU server.
  - A note or question sent with compare on is answered as usual, with no second column.
  - Without both models the server refuses compare: "Compare needs a Mamba-3 model and the 13D model on this server."
  - The JSON export holds both reports. The Markdown export adds a section for the second model.
  - The cluster server serves `m3eos,m3`, so compare mode is off there until 13D is added at a server start.
- **The laptop dev server** runs the tiny engine:
  ```bash
  venv/bin/python -c "import uvicorn; from app.server import create_app; uvicorn.run(create_app(home='/tmp/cxrchat_dev', tiny_step_delay_s=0.05, tiny_gallery=True), host='127.0.0.1', port=8000, log_level='warning')"
  ```
  The tiny engine has random weights and a synthetic gallery, so its text is never a report. Its cards say so, and the banner shows a "test model" badge. Real reports come only from the cluster server.
  To try compare mode on the laptop, add `tiny_models=['hybrid_150m_m3_rrg', 'hybrid_150m_v2_rrg']` to that `create_app` call. The tiny engine then serves both names, each with its own random weights, and their cards still carry the test-model marking.

## 5. Public demo (only when you decide to)

Not offered now: this repository runs no public demo and no server in public mode. The steps below record how one works.

1. Create the token file once, on the cluster:
   ```bash
   (umask 077; head -c 32 /dev/urandom | base64 > ~/chat_sessions/app_token)
   ```
   The file is mode 0600, owned by you, and never printed by the server. Once it exists, the server requires the token in private mode too.
2. Cancel the private job, which holds the session database (section 2). Then submit with `MODE=public` and start the tunnel.
3. Run `LOCAL_PORT=<port> bash app/tunnel/public_demo.sh`.
   - It refuses unless the server's `/healthz` says `"mode": "public"`.
   - It opens a cloudflared quick tunnel and checks the mode again every `DEMO_RECHECK_S` seconds (default 10).
   - It closes the link and exits 1 if the server stops saying public, or does not answer within 5 s.
4. Give visitors the token out of band. Read it with `cat ~/chat_sessions/app_token` in your own terminal, and never paste it into a log or a chat. Visitors enter it in Settings → Access token.
5. When the demo ends, stop the script (Ctrl-C). Anyone with the link reaches the server until you do. Then:
   - run `pgrep cloudflared`, because a `public_demo.sh` killed with SIGKILL leaves cloudflared running;
   - cancel the public job and start the private one again.

How public mode works:
- **Tokens.** Every `/v1` route needs `Authorization: Bearer <token>`. The session and message routes also need an `X-Client-Id` header. The page sends both. `/healthz`, the page and `/docs` need neither.
- **Rate limit.** 6 turns per client per 10 minutes. The 429 says when to try again (`Retry-After`). A text-only note that runs no model spends nothing when it is at most 1,000 characters long; a longer one counts as a turn. In either mode, message text over 20,100 characters is refused with a 422 before anything is stored. The limit is keyed on the client's own `X-Client-Id`, so it is a courtesy, not a defence. The hard bound is the queue: 4 turns running or waiting at once.
- **Retention.** 7 days for public sessions.

## 6. Stop, restart, requeue

- **Stop the server:** `ssh "$CLUSTER_HOST" scancel <jobid>`.
  - The API removes the endpoint file first.
  - It then ends a running turn as an error with `server_restart`, and exits 0.
  - A running tunnel stops its forward at its next re-read, within `RECHECK_S`.
- **Requeue:** `ssh "$CLUSTER_HOST" scontrol requeue <jobid>`. What is expected, which `scripts/chat_requeue_drill.py --login "$CLUSTER_HOST"` checks end to end:
  - the job comes back, possibly on another node, and appends to the same log;
  - the cut-off turn stays an error;
  - the tunnel re-attaches without help.
- **Stop the tunnel:** Ctrl-C, or kill its process. With no server, it waits and retries.
- **Known limit.** An upload aborted by a TCP reset within about 5 ms of its last byte can still start a turn, because the kernel delivers the reset only after the server has written. Browsers and the ssh tunnel close with FIN instead, so this does not arise through the tunnel.

## 7. Retention

- A session is deleted `RETENTION_DAYS` days after it was **created**: 45 by default on a private server and 7 on a public one.
- The sweep runs at every start-up and then every 24 hours.
- Each sweep uses the server's own days for its own mode and the default for the other mode. So a public demo's sessions are gone after 7 days, whatever the server runs now.
- A session with a running turn waits for the next sweep.
- **Deleting a session in the page** hides it and removes its uploaded images at once. Its stored rows stay until the session reaches its retention age. On a private server, those rows include the event log with the report text.
- **A saved export** (`exports/`, written by `POST /v1/sessions/{id}/export/save`, private mode only) is deleted once the time in its file name is older than the private retention days: 45 by default, or the server's own `RETENTION_DAYS` on a private server. Its session's age does not matter. A leftover temporary file from an interrupted save is deleted after 1 day. The sweep removes only file names the save route writes: it never follows a symbolic link and never enters a subdirectory.

## 8. What is stored (all on the cluster, under `CHAT_HOME`)

| Path | What |
|---|---|
| `chat.db` | SQLite (WAL, exclusive locking, which is safe on NFS). It holds sessions, messages and every streamed event, in the order they were sent. The event log is the source of truth for replay, resume and export. |
| `uploads/<session>/<sha256>/` | An uploaded image and its model input, stored per session by content: turns of a session that send the same bytes share them. Removed when the session is deleted. |
| `exports/` | Session exports saved on the server (`POST /v1/sessions/{id}/export/save`, private mode only): `session-<id>-<UTC time>.json` or `.md`, mode 0600, in a 0700 directory. Deleted after the private retention days (section 7). |
| `logs/` | `server_<jobid>.log` (the API's stderr: the uvicorn log and tracebacks) and `labeler_<jobid>.log`. |
| `endpoint` | One JSON line, `{host, port, mode, pid, started_at}`, present while a server is serving. |
| `app_token` | The token: needed for public mode or a non-loopback bind, and required in every mode once it exists. Mode 0600, never printed. |
| `models/`, `gallery/` | Inputs. The server reads them and never writes them. |

The API also:
- logs one JSON line per request on stderr, with no query string;
- sets `X-Request-Id` on every response;
- refuses a request over 21 MB with 413.

## 9. API walkthrough (curl)

This walkthrough uses a private server with no `app_token`, through the tunnel; set `B` to the tunnel's `LOCAL_PORT`.
- A server with an `app_token` needs `Authorization: Bearer <token>` on every `/v1` request. That covers public mode, and private mode once the token file exists. Keep the header in a file of mode 0600 and pass it with `-H @<file>`, so the token never appears on a command line.
- Public mode also needs an `X-Client-Id` header on the session and message routes.
- The options below are the published protocol.
- The stream prints MIMIC-derived text to your terminal (section 1).

```bash
B=http://localhost:8010
S=$(curl -s -X POST "$B/v1/sessions" -H 'Content-Type: application/json' -d '{}' | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
curl -N -X POST "$B/v1/sessions/$S/messages" -F 'image=@xray.png' -F 'options={"decode": "beam", "beam_size": 3, "max_new_tokens": 100, "cached_decode": true, "compile": false, "k_images": 4, "k_reports": 3, "label": true, "reference": null, "display_repair": false, "stop_on_repeat": false}'
curl -s "$B/v1/sessions/$S/export?format=md" -o session.md    # MIMIC-derived on a private server: see section 1
curl -s -X DELETE "$B/v1/sessions/$S" -o /dev/null -w '%{http_code}\n'   # 204
```

The turn streams Server-Sent Events, in this order:
1. `message_start`.
2. For each stage, `stage_start` and then `stage_end`.
   - A stage skipped before it starts sends only its `stage_end`, with a `skipped` reason. A label stage whose labeller fails after it started sends `stage_start`, then a skipped `stage_end`.
   - In public mode the score stage sends nothing.
   - Inside the generate stage come `content_block_start`, one `content_block_delta` per decode step (a `beam_snapshot`), and `content_block_stop`.
3. `message_stop`, which carries the report.

While no event comes, a `: ping` comment arrives every 15 s. Every route is documented at `$B/docs`.
