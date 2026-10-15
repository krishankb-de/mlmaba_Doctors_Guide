"""The serving command line of the chat app (CHAT_UI_PLAN.md P7-B): `python -m app.server`.

    python -m app.server --engine real --mode private --home ~/chat_sessions --gallery <build dir> \\
        --labeler http://127.0.0.1:8001 --endpoint-file ~/chat_sessions/endpoint [--token-file ~/chat_sessions/app_token]

scripts/serve_chat_h100.sh (CPU) and scripts/serve_chat_gpu_h100.sh run it, with the CheXbert labeller beside it in the same SLURM job.
This is app.server's create_app behind a command line, plus what a serving process needs and a test fixture does not:

* R6. A bind off loopback and public mode need a token. It is read from a file, which must be mode 0600 and the user's own, and never from
  the command line or the environment, so it stays out of `scontrol show job`, `ps` and shell history.
* `--port 0` picks a free port, and the endpoint file {host, port, mode, pid, started_at} says which, once uvicorn is serving. It is
  written atomically (mode 0600) and removed first when the shutdown begins, so that a tunnel stops finding a server that is going away.
* A serving process refuses what the factory tolerates: a gallery that will not open (unless --allow-no-gallery), a real engine asked to
  run on CUDA where there is none.
* P8-B. RETENTION_DAYS, in the environment, is how many days a session is kept (45 private, 7 public when it is not set; app/retention.py).
  A value that is not a whole number from 1 to 3650 is refused here, as one ERROR line and exit status 2, before anything is loaded.
* P9-B. --cors-origin ORIGIN, repeatable, lets the pages of that origin call the API (the GitHub Pages copy of the page): an origin and nothing
  after it, http only for this computer, refused otherwise in one ERROR line that names no value. Without it there is no CORS at all, as before.
* Once uvicorn serves, SIGTERM and SIGINT end it cleanly: the endpoint file goes, no new connections, a running or queued turn ends as an
  error with server_restart (the worker's own shutdown, the ending a crash's recovery gives too), and the exit status is 0. uvicorn re-raises
  a signal it handled once it has shut down, which would end the process by that signal, so this module takes the two signals itself. While
  the model loads there is no handler yet. SIGTERM takes the default action: the process ends on the spot, with the signal as its status (a
  handler could not do better, since nothing in Python runs while torch reads a checkpoint), and the wrapper, which sent the signal, counts
  that end as a stop. SIGINT is Python's own KeyboardInterrupt, raised in the thread that waits for the engines, which are built on the
  worker's thread: create_app's refused start stops the worker and closes the database, and the stop waits for the start-up on the worker to
  end, so the process is still there until the load is over; then it says `ERROR interrupted while starting` and exits with status 130.
* Its stdout is the job log's for the life of the process, and a job log carries only === , [server] , RESULT and ERROR lines (R7). What
  the thesis loader prints while a model loads (it names the checkpoint), and anything else that is printed while the server runs, goes to
  stderr instead, with uvicorn's own log and the request log (one JSON line per request, app/hardening.py: no query string, no body), which
  the wrapper keeps in a file under CHAT_HOME. Nothing printed here names the endpoint, the
  token, a path or any report text, and a refusal never quotes a value it was given (it could be a secret put in the wrong place): an unknown
  argument is named only when it is a long option's name.
"""
import argparse
import contextlib
import io
import json
import os
import re
import signal
import socket
import stat
import sys
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import urlsplit

import torch
import uvicorn

from app.labels import RuleLabeler
from app.retention import resolve_days
from app.server import MODEL_NAMES, ModelUnavailable, _is_loopback, create_app

MODEL_ALIASES = {"m3": "hybrid_150m_m3_rrg", "13d": "hybrid_150m_v2_rrg",
                 "m3eos": "hybrid_150m_m3_rrg_eos"}   # --models m3,13d; m3eos is the model trained to end its reports (P9-G5a)
GRACEFUL_S = 10                    # uvicorn's timeout_graceful_shutdown: SLURM sends SIGKILL KillWait (30 s by default) after SIGTERM
MAX_TOKEN_FILE_BYTES = 4096
TOKEN = re.compile(r"[\x21-\x7e]+")                             # visible ASCII, no space: it travels in an Authorization header
JOB_LOG_LINE = re.compile(r"^(=== |\[server\] |RESULT |ERROR)")  # the four shapes a job log carries (`chat_remote.sh summary` shows these)

# P9-B: an origin as --cors-origin takes it, and as the page's API server setting does (app/static/app.js normalizeApiServer; both are held to
# tests/fixtures/origins.json): a scheme, a host and an optional port, then nothing but an optional /. http only for this computer.
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_ORIGIN = re.compile(r"(https?)://(\[::1\]|(?:" + _OCTET + r"\.){3}" + _OCTET + r"|(?:" + _LABEL + r"\.)*[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?)"
                     r"(?::([1-9]\d{0,4}))?/?")
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "[::1]")
DEFAULT_PORTS = {"http": "80", "https": "443"}
CORS_ORIGIN_MSG = ("--cors-origin takes an origin and nothing after it: https://host[:port], or http://localhost[:port] (also 127.0.0.1 "
                   "or [::1]) for this computer")


def emit(line: str) -> None:
    """One line on stdout, written whole in a single call. print() writes the text and the newline separately, which on unbuffered
    output are two writes, and SLURM's SIGTERM reaches the wrapper and this process at the same moment: a line of the wrapper's could
    land between the two halves of ours."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def say(message: str) -> None:
    emit("[server] " + message)


# ---- the arguments -------------------------------------------------------------------------------------------------------------

# What argparse's own messages quote: the value it refused. That can be a secret in the wrong place (`--token SECRET`, `--allow-compile=SECRET`),
# and a refusal is a line of a job log, so a message is cut down to what was wrong. "invalid choice: 'X' (...)" and "invalid int value: 'X'" have
# a colon before the value, "ignored explicit argument 'X'" (a flag that was handed one with =) a space.
_QUOTES_A_VALUE = re.compile(r"(invalid (?:choice|\w+ value)|ignored explicit argument)[: ].*$", re.DOTALL)
# An unknown argument is named only when it has the shape of a long option's name. One that only starts with a dash (`-SECRET`, which argparse
# takes for an option) could be a value, and is counted as unrecognized without its text.
_OPTION_NAME = re.compile(r"--[a-z][a-z0-9-]*")


class _Parser(argparse.ArgumentParser):
    def parse_args(self, args: Optional[Sequence[str]] = None, namespace: Any = None) -> argparse.Namespace:
        parsed, extras = self.parse_known_args(args, namespace)
        if extras:      # argparse would print all of them, values included
            names = sorted({head for head in (arg.split("=", 1)[0] for arg in extras) if _OPTION_NAME.fullmatch(head)})
            self.error("unrecognized arguments: " + (", ".join(names) if names else "(no option name to show)"))
        return parsed

    def error(self, message: str) -> None:
        """A refusal is one ERROR line on stdout, which is what a job log keeps, and exit status 2; the usage goes to stderr. Neither shows a
        value that was given."""
        emit("ERROR " + _QUOTES_A_VALUE.sub(r"\1", message))
        self.print_usage(sys.stderr)
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    """No abbreviations: `--token abc` must not become `--token-file abc` and leave a secret on the command line."""
    p = _Parser(prog="python -m app.server", allow_abbrev=False,
                description="The chat server: app.server's create_app behind a command line (CHAT_UI_PLAN.md P7-B).")
    p.add_argument("--engine", required=True, choices=("tiny", "real"), help="tiny: a synthetic engine; real: the report models")
    p.add_argument("--mode", required=True, choices=("private", "public"), help="public needs a token file (R6)")
    p.add_argument("--home", required=True, metavar="DIR", help="CHAT_HOME: the database, uploads and gallery; outside any checkout")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="where the real engine runs (default cpu)")
    p.add_argument("--gallery", metavar="DIR", help="a gallery build directory; it must open, or the server stops")
    p.add_argument("--allow-no-gallery", action="store_true", help="serve without retrieval when --gallery does not open")
    p.add_argument("--labeler", default="none", metavar="URL|rule|none",
                   help="the CheXbert labeller's URL, the keyword RuleLabeler (rule) or none (default)")
    p.add_argument("--host", default="127.0.0.1", help="bind address; anything off loopback needs a token file (R6)")
    p.add_argument("--port", type=int, default=0, help="0 picks a free port; the endpoint file says which")
    p.add_argument("--endpoint-file", metavar="PATH", help="written when the server is serving: {host, port, mode, pid, started_at}")
    p.add_argument("--token-file", metavar="PATH", help="the token: a file of mode 0600 that you own, never an argument or a variable")
    p.add_argument("--models", default="m3", metavar="m3[,13d,m3eos]", help="the models served, the first is the default (m3, 13d, m3eos)")
    p.add_argument("--threads", type=int, default=8, help="CPU threads of the real engine")
    p.add_argument("--drift-note", default="", metavar="TEXT",
                   help="the CPU-versus-GPU drift sentence the cards of the published models carry (measured for them only; m3eos has its own, on a CPU server)")
    p.add_argument("--allow-compile", action="store_true", help="accept the compile option")
    p.add_argument("--published-model", metavar="DIR", help="the published dump of the default model (hyps.txt), with --published-floor")
    p.add_argument("--published-floor", metavar="DIR", help="the published dump of the retrieval floor (hyps.txt)")
    p.add_argument("--cors-origin", action="append", metavar="ORIGIN",
                   help="an origin whose pages may call this API, such as the GitHub Pages copy's (P9-B); repeatable. https://host[:port], "
                        "or http://localhost[:port] (also 127.0.0.1 or [::1]). None, the default: no CORS at all")
    return p


class TokenFileError(ValueError):
    """The token file cannot be used. The message names neither the file nor anything in it."""


def read_token(path: str) -> str:
    """The token in `path`, without its surrounding whitespace. The file must be a regular file, mode 0600 exactly, and the user's own.
    It is opened without blocking (a pipe would otherwise hold the open until someone wrote to it) and judged by what was opened."""
    try:
        fd = os.open(os.path.expanduser(path), os.O_RDONLY | os.O_NONBLOCK)
    except OSError:
        raise TokenFileError("the token file cannot be read") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise TokenFileError("the token file is not a regular file")
        if info.st_uid != os.getuid():
            raise TokenFileError("the token file is not owned by you")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise TokenFileError("the token file must be mode 0600 (chmod 600)")
        data = os.read(fd, MAX_TOKEN_FILE_BYTES + 1)
    except OSError:
        raise TokenFileError("the token file cannot be read") from None
    finally:
        os.close(fd)
    if len(data) > MAX_TOKEN_FILE_BYTES:
        raise TokenFileError("the token file is too large")
    try:
        text = data.decode("ascii").strip()
    except UnicodeDecodeError:
        raise TokenFileError("the token must be visible ASCII without spaces") from None
    if not text:
        raise TokenFileError("the token file is empty")
    if TOKEN.fullmatch(text) is None:
        raise TokenFileError("the token must be visible ASCII without spaces")
    return text


def _models(parser: argparse.ArgumentParser, text: str) -> tuple:
    names = []
    for part in text.split(","):
        name = part.strip()
        if not name:
            parser.error("--models has an empty name")
        full = MODEL_ALIASES.get(name, name)
        if full not in MODEL_NAMES:
            parser.error("--models: unknown model (the models are {})".format(", ".join(sorted(MODEL_ALIASES))))
        if full in names:
            parser.error("--models names a model twice")
        names.append(full)
    return tuple(names)


def normalize_origin(text: Any) -> Optional[str]:
    """An origin as a browser sends it in an Origin header (lower case, no default port, no trailing slash), or None for anything that is not
    one of the shapes P9-B takes: https://host[:port] for any host, http://localhost, 127.0.0.1 or [::1] with an optional port for this
    computer only. A path, a query, a fragment, a user name, a wildcard, a host that is not plain ASCII or an empty value is refused. The page's
    API server setting is the same rule (app/static/app.js normalizeApiServer, held to tests/fixtures/origins.json as this is)."""
    if not isinstance(text, str):
        return None
    raw = text.strip(" \t\n\r\f\v")
    if len(raw) > 300 or re.fullmatch(r"[\x21-\x7e]+", raw) is None:
        return None
    m = _ORIGIN.fullmatch(raw.lower())
    if m is None:
        return None
    scheme, host, port = m.group(1), m.group(2), m.group(3)
    if len(host) > 253 or (port is not None and int(port) > 65535) or (scheme == "http" and host not in LOOPBACK_HOSTS):
        return None
    if port == DEFAULT_PORTS[scheme]:
        port = None
    return "{}://{}{}".format(scheme, host, ":" + port if port else "")


def _labeler_ok(value: str) -> bool:
    if value in ("rule", "none"):
        return True
    try:
        parts = urlsplit(value)
        return parts.scheme in ("http", "https") and bool(parts.hostname)
    except ValueError:
        return False


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """The arguments, checked: every refusal is one ERROR line and exit status 2, before anything is loaded or opened. The result has
    `models` as a tuple of config names and `token` (the token file's text, or None)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be 0 to 65535")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    args.models = _models(parser, args.models)
    if not _labeler_ok(args.labeler):
        parser.error("--labeler must be a URL (http or https), rule or none")
    if bool(args.published_model) != bool(args.published_floor):
        parser.error("--published-model and --published-floor go together")
    args.cors_origins = ()   # P9-B: normalised, in the order given, each once; none (the default) gives no CORS middleware
    for given in args.cors_origin or ():
        origin = normalize_origin(given)
        if origin is None:
            parser.error(CORS_ORIGIN_MSG)
        if origin not in args.cors_origins:
            args.cors_origins += (origin,)
    args.token = None
    if args.token_file:
        try:
            args.token = read_token(args.token_file)
        except TokenFileError as exc:
            parser.error(str(exc))
    if args.mode == "public" and args.token is None:
        parser.error("public mode needs a token file (R6)")
    if not _is_loopback(args.host) and args.token is None:
        parser.error("a bind off loopback needs a token file (R6)")
    try:   # RETENTION_DAYS (P8-B) is read by create_app; a bad one stops the start here, as one ERROR line, before anything is loaded
        resolve_days(args.mode)
    except ValueError as exc:
        parser.error(str(exc))
    return args


# ---- the endpoint file ---------------------------------------------------------------------------------------------------------

def write_endpoint(path: str, port: int, mode: str) -> None:
    """{host, port, mode, pid, started_at} as one JSON line, mode 0600, replacing any older file whole (written beside it, then renamed)."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"host": socket.gethostname(), "port": int(port), "mode": mode, "pid": os.getpid(),
               "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    fd, temporary = tempfile.mkstemp(prefix=".endpoint.", suffix=".tmp", dir=str(target.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(target))
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def remove_endpoint(path: str) -> None:
    """Remove the endpoint file this process wrote, and no other: a server that started later (a requeue lands on another node, and a
    slow old one may still be stopping) has its own."""
    target = Path(path).expanduser()
    try:
        data = json.loads(target.read_text())
    except (OSError, ValueError):
        return
    if isinstance(data, dict) and data.get("pid") == os.getpid() and data.get("host") == socket.gethostname():
        with contextlib.suppress(OSError):
            target.unlink()


# ---- stdout for the life of the process ------------------------------------------------------------------------------------------

class JobLogFilter(io.TextIOBase):
    """Stdout for the life of the process, while the app loads, while it serves and while it stops: a whole line of a shape a job log may
    carry goes to `out`, every other line to `rest` (stderr, which the wrapper keeps in a file). The thesis loader prints `Loaded checkpoint:
    <path>` and the like, and a library can print while the server is up; they are the server log's. The worker thread and the event loop can
    both write, so a line is judged under a lock."""

    def __init__(self, out: Any, rest: Any):
        super().__init__()
        self._out, self._rest, self._partial = out, rest, ""
        self._lock = threading.Lock()

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        with self._lock:
            self._partial += text
            *lines, self._partial = self._partial.split("\n")
            for line in lines:
                (self._out if JOB_LOG_LINE.match(line) else self._rest).write(line + "\n")
        return len(text)

    def flush(self) -> None:
        for stream in (self._out, self._rest):
            try:
                stream.flush()
            except ValueError:      # closed under us: a collected filter is closed (and so flushed) when the streams it wrapped may be gone
                pass

    def finish(self) -> None:
        """What follows the last newline has been judged by no newline: it goes aside, since it cannot be a whole line of any shape."""
        with self._lock:
            if self._partial:
                self._rest.write(self._partial + "\n")
                self._partial = ""
        self.flush()


# ---- the server ----------------------------------------------------------------------------------------------------------------

class ServingServer(uvicorn.Server):
    """uvicorn.Server with the three things a serving job needs: the endpoint file once it is serving and not before, the turns stopped as
    the shutdown begins (not after uvicorn's graceful period has run out on a stream that waits for them), and the termination signals taken
    by this class, so that a stop ends in a normal return."""

    def __init__(self, config: uvicorn.Config, app: Any, endpoint_file: Optional[str] = None, mode: str = "private"):
        super().__init__(config)
        self.chat_app, self.endpoint_file, self.mode = app, endpoint_file, mode

    async def startup(self, sockets: Any = None) -> None:
        await super().startup(sockets=sockets)
        if not self.started or self.should_exit:      # a signal came during start-up: the shutdown follows at once
            return
        if self.endpoint_file:
            write_endpoint(self.endpoint_file, self.servers[0].sockets[0].getsockname()[1], self.mode)
        state = self.chat_app.state
        say("serving: retrieval={} labels={} published={}".format(*("on" if x is not None else "off"
                                                                     for x in (state.gallery, state.labeler, state.published))))

    async def shutdown(self, sockets: Any = None) -> None:
        say("stopping")
        if self.endpoint_file:      # first of all: a tunnel that looks for this server stops finding it now, not once the drain is over
            remove_endpoint(self.endpoint_file)
        worker = self.chat_app.state.worker
        worker.cap = 0      # no turn is accepted from here on: a request that is late gets the busy 429, not a place behind the stop
        # The worker stops every turn at its next step: each ends as an error with server_restart, its stream closes, and uvicorn's wait for
        # open connections ends. The lifespan joins the worker again afterwards; a second shutdown() is harmless.
        threading.Thread(target=worker.shutdown, name="stop-turns", daemon=True).start()
        await super().shutdown(sockets=sockets)
        say("stopped")

    @contextlib.contextmanager
    def capture_signals(self):
        """SIGINT and SIGTERM go to handle_exit as in uvicorn, but nothing is re-raised afterwards (see the module's docstring)."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous = {sig: signal.signal(sig, self.handle_exit) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


# ---- main ----------------------------------------------------------------------------------------------------------------------

def _release(app: Any) -> None:
    """Let go of an app that was built and will not be served: its worker thread and the database's exclusive lock."""
    app.state.worker.shutdown()
    app.state.store.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.engine == "real" and args.device == "cuda" and not torch.cuda.is_available():
        emit("ERROR CUDA is not available on this node")
        return 1
    # From here to the end of the process stdout is the job log's, and only lines of its four shapes reach it: while the app loads (the
    # thesis loader names a checkpoint), while it serves (a library that prints) and while it stops. Everything else goes to stderr.
    held = JobLogFilter(sys.stdout, sys.stderr)
    try:
        with contextlib.redirect_stdout(held):
            return _serve(args)
    finally:
        held.finish()


def _serve(args: argparse.Namespace) -> int:
    say("starting: engine={} device={} mode={} threads={}{}".format(
        args.engine, args.device, args.mode, args.threads, " models=" + ",".join(args.models) if args.engine == "real" else ""))
    labeler, labeler_url = None, None
    if args.labeler == "rule":
        labeler = RuleLabeler()
    elif args.labeler != "none":
        labeler_url = args.labeler
    published = {"model": args.published_model, "floor": args.published_floor} if args.published_model else None
    cors = {"cors_origins": args.cors_origins} if args.cors_origins else {}   # P9-B: no flag, no argument: create_app's own default, no CORS
    try:
        app = create_app(engine=args.engine, mode=args.mode, home=args.home, host=args.host, token=args.token,
                         gallery_dir=args.gallery, labeler_url=labeler_url, labeler=labeler, models=args.models,
                         allow_compile=args.allow_compile, drift_note=args.drift_note, device=args.device, threads=args.threads,
                         published_dirs=published, **cors)
    except KeyboardInterrupt:
        emit("ERROR interrupted while starting")
        return 130
    except ModelUnavailable as exc:     # P9-G5a: a model asked for has no finished checkpoint, or an EOS entry is not marked EOS-trained by its run;
                                        # the message names the model and never a path
        emit("ERROR {}".format(exc))
        return 1
    except Exception as exc:    # its text can hold a path: the class here, the traceback in the server log (stderr)
        emit("ERROR cannot start the server ({})".format(type(exc).__name__))
        traceback.print_exc()
        return 1
    if args.gallery and app.state.gallery is None and not args.allow_no_gallery:
        _release(app)       # create_app is lenient about a gallery for tests; a serving process is not
        emit("ERROR the retrieval gallery could not be opened (see the [server] gallery line)")
        return 1
    config = uvicorn.Config(app, host=args.host, port=args.port, access_log=False, timeout_graceful_shutdown=GRACEFUL_S)
    server = ServingServer(config, app, endpoint_file=args.endpoint_file, mode=args.mode)
    try:
        server.run()
    except SystemExit as exc:   # uvicorn's own: the port is taken, or the app's start-up failed
        code = exc.code if isinstance(exc.code, int) else 1
        emit("ERROR the server did not start (exit {})".format(code))
        return code
    return 0
