"""scripts/vm_setup.sh: the chat server's environment and systemd unit on a Linux VM.

The script is run for real in a temp copy of the repo, with stub uv, curl, sha256sum and tar and stub venv pythons that run the
script's own import checks against fake modules. Nothing is downloaded or installed, and HOME points into the temp tree.
"""
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "vm_setup.sh"
TEMPLATE = REPO_ROOT / "scripts" / "vm" / "cxr-chat.service.in"
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"
UV_SHA = "f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645"

CACHE_VARS = ("XDG_CACHE_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE")   # the script runs its paths without these

FAKE_MODULES = {   # what the script's import checks need; the stub pythons put this directory on PYTHONPATH
    "torch.py": '__version__ = "%s"\n\n\nclass version:\n    cuda = None\n',
    "torchvision.py": '__version__ = "%s"\n',
    "transformers.py": (
        '__version__ = "4.57.6"\n\n\nclass AutoTokenizer:\n    @staticmethod\n    def from_pretrained(name):\n'
        '        import os\n        with open(os.environ["STUB_DIR"] + "/prefetch.calls", "a") as f:\n'
        '            f.write("%s HF_HOME=%s OFFLINE=%s ENV=%s\\n" % (name, os.environ.get("HF_HOME"), os.environ.get("HF_HUB_OFFLINE"),\n'
        '                sorted(k for k in __CV__ if k in os.environ)))\n'
    ).replace("__CV__", repr(CACHE_VARS)),
    "open_clip.py": (
        '__version__ = "3.3.0"\n\n\ndef create_model_from_pretrained(name):\n    import os\n'
        '    with open(os.environ["STUB_DIR"] + "/prefetch.calls", "a") as f:\n        f.write(name + "\\n")\n'
    ),
    "f1chexbert.py": "",
    "sklearn.py": '__version__ = "1.7.2"\n',
    "fastapi.py": '__version__ = "0.143.2"\n',
    "uvicorn.py": "",
    "httpx.py": '__version__ = "0.28.0"\n',
    "python_multipart.py": "",
}

FAKE_CHEXBERT_MODULES = {   # only the CheXbert venv's python sees these
    "appdirs.py": (   # like the real one: XDG_CACHE_HOME first
        'import os\n\n\ndef user_cache_dir(name):\n'
        '    return os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.join(os.environ["HOME"], ".cache"), name)\n'),
    "huggingface_hub.py": (
        'import os\nimport shutil\n\n\ndef hf_hub_download(repo_id, filename, **kw):\n'
        '    with open(os.environ["STUB_DIR"] + "/chexbert.calls", "a") as f:\n'
        '        f.write("%s %s HF_HOME=%s OFFLINE=%s kw=%s ENV=%s\\n" % (repo_id, filename, os.environ.get("HF_HOME"),'
        ' os.environ.get("HF_HUB_OFFLINE"), sorted(kw), sorted(k for k in __CV__ if k in os.environ)))\n'
        '    d = os.path.join(os.environ["HF_HOME"], "hub", "snapshots", "abc")\n    os.makedirs(d)\n'
        '    path = os.path.join(d, filename)\n    shutil.copy(os.environ["STUB_DIR"] + "/cx_payload", path)\n    return path\n'
    ).replace("__CV__", repr(CACHE_VARS)),
}
CX_SHA = "6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1"
CX_GOOD = b"good chexbert"   # the stub sha256sum reports the pinned hash for this content

PY_STUB = """#!/bin/bash
# stands in for a venv's python: the script's own `-c` check, run by the test interpreter against fake modules (-S hides the
# test interpreter's site-packages, so a module that is missing is missing). The CheXbert venv has its own torch, as the real one does.
case "$0" in
  *.venv_chexbert*) export PYTHONPATH="${PYTHONPATH:-}:__FAKE_CX__:__FAKE__" ;;
  *) export PYTHONPATH="${PYTHONPATH:-}:__FAKE__" ;;
esac
exec "__PYTHON__" -S "$@"
"""

UV_STUB = """#!/bin/bash
# stands in for uv: records argv; "venv" makes the stub python; "pip install --target" fills the directory; a call whose
# arguments contain the text in $STUB_DIR/uv.fail_on fails
{ echo "@@"; for a in "$@"; do printf '%s\\n' "$a"; done; } >> "$STUB_DIR/uv.calls"
if [ -s "$STUB_DIR/uv.fail_on" ] && [[ "$*" == *"$(cat "$STUB_DIR/uv.fail_on")"* ]]; then echo "error: simulated" >&2; exit 1; fi
case "$1" in
  venv)
    for last in "$@"; do :; done
    mkdir -p "$last/bin"; cp "$STUB_DIR/py_stub" "$last/bin/python"; chmod 755 "$last/bin/python" ;;
  pip)
    target=""
    while [ $# -gt 0 ]; do if [ "$1" = "--target" ]; then target="$2"; fi; shift; done
    if [ -n "$target" ]; then mkdir -p "$target"; : > "$target/installed"; fi ;;
esac
exit 0
"""

CURL_STUB = """#!/bin/bash
# records the URL, writes fake archive bytes to the --output file
out=""
while [ $# -gt 0 ]; do
  case "$1" in --output) out="$2"; shift ;; http*) echo "$1" >> "$STUB_DIR/curl.calls" ;; esac
  shift
done
echo "fake archive" > "$out"
"""

SHA_STUB = """#!/bin/bash
# the uv archive gets the sha in $STUB_DIR/sha; a file holding "good chexbert" gets the pinned CheXbert hash (a real file is 1.3 GB);
# every other file gets its real hash
case "$1" in
  *uv.tar.gz) echo "$(cat "$STUB_DIR/sha")  $1" ;;
  *) if [ "$(head -c 13 "$1" 2>/dev/null)" = "good chexbert" ]; then echo "__CX_SHA__  $1"; exit 0; fi
     if [ -x /usr/bin/sha256sum ]; then exec /usr/bin/sha256sum "$@"; else exec /usr/bin/shasum -a 256 "$@"; fi ;;
esac
"""

ID_STUB = """#!/bin/bash
# a root user for `id -u`; every other question goes to the real id
if [ "${1:-}" = "-u" ]; then echo 0; else exec /usr/bin/id "$@"; fi
"""

TAR_STUB = """#!/bin/bash
# extracts the "uv" of the archive: a copy of the stub uv into the -C directory
dest=""
while [ $# -gt 0 ]; do if [ "$1" = "-C" ]; then dest="$2"; fi; shift; done
cp "$STUB_DIR/uv_stub" "$dest/uv"; chmod 755 "$dest/uv"
"""


def _write(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def run_in_group(argv: List[str], timeout: float, **kw) -> Tuple[subprocess.CompletedProcess, bool]:
    """Run argv in its own process group. On a timeout the whole group is killed (a plain timeout kills only the bash, and a
    child such as the image-count `python -c` outlives it), then TimeoutExpired is raised. Returns the result and whether a
    process of the group was still alive after the parent ended (it is killed)."""
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, start_new_session=True, **kw)
    try:
        out, err = proc.communicate(timeout=timeout)
    except BaseException:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    stray = _group_alive(proc.pid)
    if stray:
        os.killpg(proc.pid, signal.SIGKILL)
    return subprocess.CompletedProcess(argv, proc.returncode, out, err), stray


def _snapshot(root: Path) -> Dict[str, int]:
    """Every file under root with its size and mtime: a re-run may add, never change or remove."""
    found = {}  # type: Dict[str, int]
    for base, dirs, files in os.walk(str(root)):
        for name in files:
            full = os.path.join(base, name)
            found[full] = os.lstat(full).st_mtime_ns
    return found


class Box:
    """A temp copy of the repo (the script, its template and pin files), a stub tool directory and a HOME inside the tree."""

    def __init__(self, root: Path):
        self.root = root
        self.strays = 0   # runs that left a process behind; the fixture fails the test
        self.repo, self.stubs, self.home, self.fake = root / "repo", root / "stubs", root / "home", root / "fake_modules"
        self.fake_cx = root / "fake_chexbert"
        (self.repo / "scripts" / "vm").mkdir(parents=True)
        shutil.copy(str(SCRIPT), str(self.repo / "scripts" / "vm_setup.sh"))
        for f in (REPO_ROOT / "scripts" / "vm").iterdir():
            shutil.copy(str(f), str(self.repo / "scripts" / "vm" / f.name))
        self.home.mkdir()
        self.fake.mkdir()
        self.fake_cx.mkdir()
        self.set_torch("2.11.0+cpu")
        self.set_torchvision("0.26.0+cpu")
        self.set_chexbert_torch("2.13.0+cpu")
        python = (PY_STUB.replace("__FAKE_CX__", str(self.fake_cx)).replace("__FAKE__", str(self.fake))
                  .replace("__PYTHON__", sys.executable))
        _write(self.stubs / "py_stub", python)
        _write(self.stubs / "uv_stub", UV_STUB, executable=True)
        _write(self.stubs / "curl", CURL_STUB, executable=True)
        _write(self.stubs / "sha256sum", SHA_STUB.replace("__CX_SHA__", CX_SHA), executable=True)
        _write(self.stubs / "tar", TAR_STUB, executable=True)
        _write(self.stubs / "sha", UV_SHA + "\n")
        _write(self.stubs / "uv.fail_on", "")
        for name, body in FAKE_CHEXBERT_MODULES.items():
            _write(self.fake_cx / name, body)
        (self.stubs / "cx_payload").write_bytes(CX_GOOD)

    @property
    def cx_target(self) -> Path:
        return self.home / ".cache" / "chexbert" / "chexbert.pth"

    def put_cx(self, content: bytes) -> None:
        self.cx_target.parent.mkdir(parents=True, exist_ok=True)
        self.cx_target.write_bytes(content)

    def set_torch(self, version: str) -> None:
        _write(self.fake / "torch.py", FAKE_MODULES["torch.py"] % version)
        for name, body in FAKE_MODULES.items():
            if name not in ("torch.py", "torchvision.py"):
                _write(self.fake / name, body)

    def set_torchvision(self, version: str) -> None:
        _write(self.fake / "torchvision.py", FAKE_MODULES["torchvision.py"] % version)

    def set_chexbert_torch(self, version: str) -> None:
        _write(self.fake_cx / "torch.py", FAKE_MODULES["torch.py"] % version)

    def meminfo(self, kb: int = 7987200) -> Path:
        path = self.root / "meminfo"
        _write(path, "MemFree:         1234 kB\nMemTotal:       %d kB\nSwapTotal:      0 kB\n" % kb)
        return path

    def env(self, **extra: str) -> Dict[str, str]:
        env = {"PATH": "%s:/usr/bin:/bin" % self.stubs, "HOME": str(self.home), "STUB_DIR": str(self.stubs)}
        env.update(extra)
        return env

    def run(self, *args: str, **extra: str) -> subprocess.CompletedProcess:
        done, stray = run_in_group([BASH, str(self.repo / "scripts" / "vm_setup.sh")] + list(args), 120, cwd=str(self.root),
                                   env=self.env(**extra))
        self.strays += int(stray)
        return done

    def uv_calls(self) -> List[List[str]]:
        log = self.stubs / "uv.calls"
        if not log.exists():
            return []
        return [rec.splitlines() for rec in log.read_text().split("@@\n") if rec.strip()]


@pytest.fixture
def box(tmp_path):
    made = Box(tmp_path)
    yield made
    assert made.strays == 0, "a run left a child process behind (it was killed)"


def test_a_timeout_kills_the_whole_process_group_not_just_the_parent(tmp_path):
    pidfile = tmp_path / "child.pid"
    with pytest.raises(subprocess.TimeoutExpired):
        run_in_group([BASH, "-c", 'sleep 300 & echo $! > "%s"; wait' % pidfile], 1.5)
    pid = int(pidfile.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the child of the timed-out run is still running")


def test_a_run_that_leaves_a_child_is_reported_and_the_child_is_killed():
    done, stray = run_in_group([BASH, "-c", "sleep 300 >/dev/null 2>&1 & exit 0"], 30)
    assert done.returncode == 0 and stray
    clean, none = run_in_group([BASH, "-c", "exit 0"], 30)
    assert clean.returncode == 0 and not none


def _lines(proc: subprocess.CompletedProcess) -> List[str]:
    return proc.stdout.splitlines()


SENTINELS = [".venv/.setup_ok", ".venv_chexbert/.setup_ok", ".chat_deps/.setup_ok", ".chat_deps_chexbert/.setup_ok"]


# ── static ────────────────────────────────────────────────────────────────────

def test_script_passes_bash_syntax_check():
    assert subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True).returncode == 0


def test_script_has_no_sudo_and_deletes_only_its_own_temporary_directory():
    code = [l for l in SCRIPT.read_text().splitlines() if not l.lstrip().startswith("#")]
    assert not any(re.search(r"\bsudo\b", l) for l in code if "ERROR run as the service user" not in l), "sudo only in the root guard's message"
    assert "| sudo" not in SCRIPT.read_text() + TEMPLATE.read_text(), "no pipe into sudo: the unit goes to a file first"
    removals = [l for l in code if re.search(r"\brm\b", l)]
    assert removals and all(re.search(r'rm -rf "\$\{dl(:-)?\}"|rm -f "\$\{tmp(:-)?\}"', l) for l in removals)
    assert not any(re.search(r"\b(rmdir|unlink|truncate)\b", l) for l in code)
    assert ".local/bin" not in "\n".join(code), "uv and the pythons never go to ~/.local/bin"


def test_every_requirements_file_the_script_names_exists():
    named = set(re.findall(r"scripts/vm/(requirements[\w.-]*\.txt)", SCRIPT.read_text()))
    assert len(named) == 6
    for name in named:
        assert (REPO_ROOT / "scripts" / "vm" / name).is_file(), name


def test_script_and_template_hold_no_host_user_or_address():
    text = SCRIPT.read_text() + TEMPLATE.read_text()
    assert not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text.replace("127.0.0.1", ""))
    assert "/Users/" not in text and "/home/" not in text


def test_template_sets_slurm_submit_dir_to_the_repo_and_pins_private_loopback():
    text = TEMPLATE.read_text()
    assert 'Environment="SLURM_SUBMIT_DIR=@REPO@"' in text
    assert "WorkingDirectory=@REPO@" in text
    assert 'Environment="MODE=private"' in text and 'Environment="BIND=127.0.0.1"' in text
    assert "KillMode=mixed" in text
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert [l for l in code.splitlines() if "CORS" in l] == ['Environment="CORS_ORIGINS="'], "pinned empty, nothing else"
    assert "token" not in code.lower()


def test_template_restarts_always_with_a_start_limit_a_daily_restart_and_memory_limits():
    lines = TEMPLATE.read_text().splitlines()
    for line in ("StartLimitIntervalSec=3600", "StartLimitBurst=5", "Restart=always", "RestartSec=30", "RuntimeMaxSec=1d",
                 "TimeoutStopSec=90", "MemoryHigh=@MEMORY_HIGH@", "MemoryMax=@MEMORY_MAX@",
                 'Environment="SLURM_CPUS_PER_TASK=@THREADS@"'):
        assert line in lines, line
    assert "Restart=on-failure" not in lines
    assert lines.index("[Unit]") < lines.index("StartLimitBurst=5") < lines.index("[Service]") < lines.index("Restart=always")


# ── install ───────────────────────────────────────────────────────────────────

def test_install_runs_the_steps_in_order_and_writes_every_sentinel(box):
    done = box.run("install", SCRATCH_ROOT=str(box.root / "scratch"), CHAT_HOME=str(box.root / "chat"))
    assert done.returncode == 0, done.stdout + done.stderr
    lines = _lines(done)
    assert lines[-1] == "=== END vm setup ==="
    assert all(l.startswith(("[setup] ", "=== ")) for l in lines)
    assert (box.stubs / "curl.calls").read_text().split() == [
        "https://github.com/astral-sh/uv/releases/download/0.13.6/uv-x86_64-unknown-linux-gnu.tar.gz"]
    assert (box.home / ".local" / "share" / "cxr-chat" / "uv-0.13.6" / "uv").is_file()
    assert not (box.home / ".local" / "bin").exists()
    kinds = []
    for call in box.uv_calls():
        if call[0] == "python":
            kinds.append("python")
        elif call[0] == "venv":
            kinds.append("venv " + call[-1])
        elif "--target" in call:
            kinds.append("overlay " + call[call.index("--target") + 1])
        else:
            kinds.append("pins " + call[call.index("--python") + 1].split("/")[0] + " " + call[-1].split("/")[-1])
    assert kinds == [
        "python", "venv .venv",
        "pins .venv requirements-serve-torch.txt", "pins .venv requirements-serve.txt",
        "venv .venv_chexbert",
        "pins .venv_chexbert requirements-chexbert-torch.txt", "pins .venv_chexbert requirements-chexbert.txt",
        "overlay .chat_deps", "overlay .chat_deps_chexbert"]
    calls = box.uv_calls()
    assert calls[0] == ["python", "install", "--no-bin", "3.11.15"]
    assert calls[1][:3] == ["venv", "--quiet", "--python"] and calls[1][3] == "3.11.15"
    torch_call = calls[2]
    assert "--no-deps" in torch_call and torch_call[torch_call.index("--index-url") + 1] == "https://download.pytorch.org/whl/cpu"
    assert "--index-url" not in calls[3]
    for name in SENTINELS:
        assert (box.repo / name).is_file(), name
    assert any(l.startswith("[setup] main venv: python ") and "torch 2.11.0+cpu" in l for l in lines)
    assert any(l.startswith("[setup] chexbert venv: python ") for l in lines)
    prefetch = (box.stubs / "prefetch.calls").read_text().splitlines()
    assert prefetch[0].startswith("gpt2 HF_HOME=%s OFFLINE=0" % (box.root / "scratch" / ".hf"))
    assert prefetch[1] == "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    assert (box.root / "scratch" / ".hf" / ".prefetch_ok").is_file()
    assert (box.root / "chat").is_dir() and oct((box.root / "chat").stat().st_mode & 0o777) == "0o700"
    assert (box.root / "chat" / "logs").is_dir() and (box.repo / "logs").is_dir()
    log = box.repo / "logs" / "chat_server_vm.log"
    assert log.is_file() and log.stat().st_size == 0 and oct(log.stat().st_mode & 0o777) == "0o600"


def _pin_hashes(box: Box, *names: str) -> List[str]:
    out = []
    for name in names:
        digest = hashlib.sha256((box.repo / "scripts" / "vm" / name).read_bytes()).hexdigest()
        out.append("%s  %s" % (digest, name))
    return out


def test_every_sentinel_holds_a_date_and_the_hashes_of_the_pin_files_its_step_used(box):
    assert box.run("install").returncode == 0
    expected = {
        ".venv/.setup_ok": ["requirements-serve-torch.txt", "requirements-serve.txt"],
        ".venv_chexbert/.setup_ok": ["requirements-chexbert-torch.txt", "requirements-chexbert.txt"],
        ".chat_deps/.setup_ok": ["requirements-overlay.txt"],
        ".chat_deps_chexbert/.setup_ok": ["requirements-overlay-chexbert.txt"],
    }
    for sentinel, names in expected.items():
        lines = (box.repo / sentinel).read_text().splitlines()
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", lines[0]), sentinel
        assert lines[1:] == _pin_hashes(box, *names), sentinel


def test_a_sha_mismatch_refuses_the_archive_and_installs_nothing(box):
    _write(box.stubs / "sha", "0" * 64 + "\n")
    done = box.run("install")
    assert done.returncode == 1
    lines = _lines(done)
    assert "[setup] ERROR uv archive sha256 does not match the pinned value: refusing it" in lines
    assert lines[-1] == "=== END vm setup: 1 ERROR line(s) above ==="
    assert not (box.home / ".local").exists()
    assert box.uv_calls() == []
    assert not (box.repo / ".venv").exists()


def test_a_rerun_skips_every_finished_step_and_changes_nothing(box):
    first = box.run("install")
    assert first.returncode == 0, first.stdout + first.stderr
    for log in ("uv.calls", "curl.calls", "prefetch.calls"):
        (box.stubs / log).unlink()
    before = _snapshot(box.root)
    again = box.run("install")
    assert again.returncode == 0, again.stdout + again.stderr
    assert _lines(again)[-1] == "=== END vm setup ==="
    assert box.uv_calls() == [] and not (box.stubs / "curl.calls").exists() and not (box.stubs / "prefetch.calls").exists()
    text = again.stdout
    assert "uv 0.13.6: present" in text and "main venv: done, skipped" in text and "hf prefetch: done, skipped" in text
    assert _snapshot(box.root) == before, "nothing was added, changed or removed"


@pytest.mark.parametrize("sentinel,pin,step", [
    (".venv", "requirements-serve.txt", "main venv"),
    (".venv", "requirements-serve-torch.txt", "main venv"),
    (".venv_chexbert", "requirements-chexbert.txt", "chexbert venv"),
    (".chat_deps", "requirements-overlay.txt", "overlay main"),
    (".chat_deps_chexbert", "requirements-overlay-chexbert.txt", "overlay chexbert"),
])
def test_a_rerun_with_changed_pin_files_stops_and_touches_nothing(box, sentinel, pin, step):
    assert box.run("install").returncode == 0
    with open(str(box.repo / "scripts" / "vm" / pin), "a") as f:
        f.write("# a new pin\n")
    (box.stubs / "uv.calls").unlink()
    before = _snapshot(box.root)
    again = box.run("install")
    assert again.returncode == 1
    lines = _lines(again)
    assert "[setup] ERROR %s pins changed since it was installed: move %s aside and re-run" % (step, sentinel) in lines
    assert lines[-1] == "=== END vm setup: 1 ERROR line(s) above ==="
    assert box.uv_calls() == [], "nothing was reinstalled"
    assert _snapshot(box.root) == before, "the directory was neither deleted nor changed"


def test_a_sentinel_in_the_old_format_counts_as_changed(box):
    assert box.run("install").returncode == 0
    (box.repo / ".chat_deps" / ".setup_ok").write_text("2026-01-01T00:00:00Z\n")
    again = box.run("install")
    assert again.returncode == 1
    assert "[setup] ERROR overlay main pins changed since it was installed: move .chat_deps aside and re-run" in _lines(again)


def test_a_failed_install_leaves_no_sentinel_and_the_next_run_installs_again(box):
    _write(box.stubs / "uv.fail_on", "requirements-serve.txt")
    first = box.run("install")
    assert first.returncode == 1
    assert "[setup] ERROR install failed: main" in _lines(first)
    assert not (box.repo / ".venv" / ".setup_ok").exists()
    _write(box.stubs / "uv.fail_on", "")
    second = box.run("install")
    assert second.returncode == 0, second.stdout + second.stderr
    assert (box.repo / ".venv" / ".setup_ok").is_file()


def test_a_cuda_torch_fails_the_main_check_and_writes_no_sentinel(box):
    box.set_torch("2.11.0+cu128")
    done = box.run("install")
    assert done.returncode == 1
    assert "[setup] ERROR venv check failed: main (torch must be 2.11.0, torchvision a CPU build)" in _lines(done)
    assert not (box.repo / ".venv" / ".setup_ok").exists()


def test_a_cuda_torchvision_fails_the_main_check_and_writes_no_sentinel(box):
    box.set_torchvision("0.26.0+cu128")
    done = box.run("install")
    assert done.returncode == 1
    assert "[setup] ERROR venv check failed: main (torch must be 2.11.0, torchvision a CPU build)" in _lines(done)
    assert not (box.repo / ".venv" / ".setup_ok").exists()


@pytest.mark.parametrize("version", ["2.13.0+cu128", "2.13.0", "2.11.0+cpu"])
def test_a_chexbert_torch_that_is_not_2_13_0_cpu_fails_its_check_and_writes_no_sentinel(box, version):
    box.set_chexbert_torch(version)
    done = box.run("install")
    assert done.returncode == 1
    assert "[setup] ERROR venv check failed: chexbert" in _lines(done)
    assert (box.repo / ".venv" / ".setup_ok").is_file() and not (box.repo / ".venv_chexbert" / ".setup_ok").exists()


def test_a_prefetch_failure_points_at_vm_remote_summary_not_the_raw_log(box):
    _write(box.fake / "open_clip.py", '__version__ = "3.3.0"\n\n\ndef create_model_from_pretrained(name):\n    raise RuntimeError("simulated")\n')
    done = box.run("install", SCRATCH_ROOT=str(box.root / "scratch"), CHAT_HOME=str(box.root / "chat"))
    assert done.returncode == 1
    assert "[setup] ERROR hf prefetch failed: bash scripts/vm_remote.sh summary logs/vm_setup_prefetch.log" in _lines(done)


def test_a_wrong_torch_version_fails_the_main_check(box):
    box.set_torch("2.10.0+cpu")
    done = box.run("install")
    assert done.returncode == 1
    assert not (box.repo / ".venv" / ".setup_ok").exists()


def test_unknown_command_prints_usage_and_exits_2(box):
    done = box.run("deploy")
    assert done.returncode == 2 and "usage:" in done.stderr


# ── check ─────────────────────────────────────────────────────────────────────

CHECK_FILES = [
    "outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt",
    "outputs/h100_report_gen_m3_tower13d_s42/run_metadata.json",
    "results/report_gen_m3_test_split_s42/hyps.txt",
    "results/report_gen_m3_test_split_s42/refs.txt",
    "results/report_gen_m3_test_split_s42/chexbert_labels.json",
    "results/retrieval_floor_test_split/hyps.txt",
]


def _check_tree(box: Path, images_present: int, images_total: int) -> Dict[str, str]:
    """A synthetic tree that passes `check`, with a real parquet pair whose image column points at fake files."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    repo, chat, scratch = box.repo, box.root / "chat", box.root / "scratch"
    for rel in CHECK_FILES + SENTINELS:
        _write(repo / rel, "x")
    eos = chat / "models" / "report_gen_m3_eos_s42"
    for rel in ("checkpoints/last.ckpt", "DONE", "run_metadata.json"):
        _write(eos / rel, "x")
    gallery = chat / "gallery" / "g13d_m3_v1"
    _write(gallery / "manifest.json", "{}")
    _write(scratch / ".hf" / ".prefetch_ok", "x")
    secret = box.root / "secret_study_dir"
    paths = []
    for i in range(images_total):
        p = secret / ("img_%d.jpg" % i)
        if i < images_present:
            _write(p, "x")
        paths.append(str(p))
    pd.DataFrame({"image": paths}).to_parquet(str(gallery / "img_meta.parquet"))
    pd.DataFrame({"image": paths[:2]}).to_parquet(str(gallery / "test_meta.parquet"))
    # the venv python of this tree is the test interpreter itself, which has pandas
    _write(repo / ".venv" / "bin" / "python", '#!/bin/bash\nexec "%s" "$@"\n' % sys.executable, executable=True)
    _write(repo / ".venv_chexbert" / "bin" / "python", (box.stubs / "py_stub").read_text(), executable=True)
    box.put_cx(CX_GOOD)
    return {"CHAT_HOME": str(chat), "SCRATCH_ROOT": str(scratch)}


def test_check_passes_on_a_complete_tree_and_prints_counts_never_a_path(box):
    env = _check_tree(box, 5, 5)
    done = box.run("check", **env)
    assert done.returncode == 0, done.stdout + done.stderr
    lines = _lines(done)
    assert "[setup] ok checkpoints/last.ckpt" in lines
    assert "[setup] ok report_gen_m3_test_split_s42/chexbert_labels.json" in lines
    assert "[setup] ok report_gen_m3_eos_s42/DONE" in lines
    assert "[setup] ok g13d_m3_v1/manifest.json" in lines
    assert "[setup] ok .venv/.setup_ok" in lines and "[setup] ok .hf/.prefetch_ok" in lines
    assert "[setup] ok chexbert/chexbert.pth" in lines
    assert "[setup] images present 5/5 (train)" in lines
    assert "[setup] images present 2/2 (test)" in lines
    assert lines[-1] == "=== END vm setup ==="
    assert all(l.startswith(("[setup] ", "=== ")) for l in lines)
    assert str(box.root) not in done.stdout + done.stderr and "secret_study_dir" not in done.stdout + done.stderr


def test_check_reports_a_missing_input_and_a_missing_image_and_exits_1(box):
    env = _check_tree(box, 3, 5)
    (box.repo / "results" / "retrieval_floor_test_split" / "hyps.txt").unlink()
    (box.root / "chat" / "models" / "report_gen_m3_eos_s42" / "DONE").unlink()
    done = box.run("check", **env)
    assert done.returncode == 1
    lines = _lines(done)
    assert "[setup] ERROR missing retrieval_floor_test_split/hyps.txt" in lines
    assert "[setup] ERROR missing report_gen_m3_eos_s42/DONE" in lines
    assert "[setup] ok checkpoints/last.ckpt" in lines, "the later checks still ran"
    assert "[setup] images present 3/5 (train)" in lines
    assert "[setup] images present 2/2 (test)" in lines
    assert lines[-1] == "=== END vm setup: 3 ERROR line(s) above ==="
    assert "secret_study_dir" not in done.stdout + done.stderr


def test_check_requires_the_engines_run_metadata_of_the_main_model(box):
    env = _check_tree(box, 5, 5)
    (box.repo / "outputs" / "h100_report_gen_m3_tower13d_s42" / "run_metadata.json").unlink()
    done = box.run("check", **env)
    assert done.returncode == 1
    assert "[setup] ERROR missing h100_report_gen_m3_tower13d_s42/run_metadata.json" in _lines(done)
    assert _lines(done)[-1] == "=== END vm setup: 1 ERROR line(s) above ==="


def test_check_on_an_empty_tree_lists_every_input_as_missing(box):
    done = box.run("check", CHAT_HOME=str(box.root / "chat"), SCRATCH_ROOT=str(box.root / "scratch"))
    assert done.returncode == 1
    missing = [l for l in _lines(done) if l.startswith("[setup] ERROR missing ")]
    assert len(missing) == 17
    assert "[setup] ERROR missing h100_report_gen_m3_tower13d_s42/run_metadata.json" in missing
    assert "[setup] ERROR missing .hf/.prefetch_ok" in missing
    assert "[setup] ERROR missing chexbert/chexbert.pth" in missing
    assert "[setup] ERROR missing .venv/bin/python: no image check" in _lines(done)


def test_check_reports_the_chexbert_weights_as_missing_or_with_a_wrong_hash_without_path_or_hash(box):
    env = _check_tree(box, 5, 5)
    box.cx_target.write_bytes(b"something else")
    wrong = box.run("check", **env)
    assert wrong.returncode == 1
    assert "[setup] ERROR unexpected sha256 chexbert/chexbert.pth" in _lines(wrong)
    assert _lines(wrong)[-1] == "=== END vm setup: 1 ERROR line(s) above ==="
    box.cx_target.unlink()
    gone = box.run("check", **env)
    assert gone.returncode == 1 and "[setup] ERROR missing chexbert/chexbert.pth" in _lines(gone)
    for out in (wrong, gone):
        assert str(box.root) not in out.stdout + out.stderr and CX_SHA not in out.stdout + out.stderr


# ── the CheXbert weights ──────────────────────────────────────────────────────

def test_install_downloads_the_chexbert_weights_into_a_temporary_hf_cache_and_places_them_atomically(box):
    done = box.run("install", SCRATCH_ROOT=str(box.root / "scratch"), CHAT_HOME=str(box.root / "chat"))
    assert done.returncode == 0, done.stdout + done.stderr
    lines = _lines(done)
    assert "[setup] chexbert weights: downloaded, sha256 verified" in lines
    assert lines.index("[setup] chexbert weights: downloaded, sha256 verified") < lines.index(
        "[setup] hf prefetch: gpt2 tokenizer and BiomedCLIP cached")
    assert box.cx_target.read_bytes() == CX_GOOD
    assert [p.name for p in box.cx_target.parent.iterdir()] == ["chexbert.pth"], "no temporary file is left"
    call = (box.stubs / "chexbert.calls").read_text().splitlines()
    assert len(call) == 1 and call[0].startswith("StanfordAIMI/RRG_scorers chexbert.pth HF_HOME=") and "OFFLINE=0 kw=[]" in call[0]
    hf_home = call[0].split("HF_HOME=")[1].split(" ")[0]
    assert hf_home != str(box.root / "scratch" / ".hf") and not os.path.exists(hf_home), "the script's own temp dir is gone"
    assert str(box.home) not in done.stdout and CX_SHA not in done.stdout


def test_install_skips_the_download_when_the_chexbert_weights_have_the_pinned_hash(box):
    box.put_cx(CX_GOOD)
    before = box.cx_target.stat().st_mtime_ns
    done = box.run("install")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "[setup] chexbert weights: present, sha256 verified" in _lines(done)
    assert not (box.stubs / "chexbert.calls").exists()
    assert box.cx_target.stat().st_mtime_ns == before and box.cx_target.read_bytes() == CX_GOOD


def test_install_stops_on_chexbert_weights_with_another_hash_and_leaves_the_file_alone(box):
    box.put_cx(b"someone else's file")
    done = box.run("install")
    assert done.returncode == 1
    assert "[setup] ERROR chexbert weights: chexbert.pth has an unexpected sha256: move it aside and re-run" in _lines(done)
    assert not (box.stubs / "chexbert.calls").exists()
    assert box.cx_target.read_bytes() == b"someone else's file"
    assert [p.name for p in box.cx_target.parent.iterdir()] == ["chexbert.pth"]
    assert not (box.home / "vm_scratch" / ".hf" / ".prefetch_ok").exists(), "the run ended before the prefetch"


def test_a_download_with_the_wrong_hash_is_refused_and_writes_nothing_to_the_target(box):
    (box.stubs / "cx_payload").write_bytes(b"truncated or swapped")
    done = box.run("install")
    assert done.returncode == 1
    assert "[setup] ERROR chexbert weights: the download has an unexpected sha256: nothing written" in _lines(done)
    assert not box.cx_target.exists() and list(box.cx_target.parent.iterdir()) == [], "no temporary file or directory is left"
    call = (box.stubs / "chexbert.calls").read_text().splitlines()
    assert not os.path.exists(call[0].split("HF_HOME=")[1].split(" ")[0])


def test_the_chexbert_cache_directory_comes_from_the_venv_python_not_from_a_fixed_path(box):
    other = box.root / "elsewhere"
    _write(box.fake_cx / "appdirs.py", 'def user_cache_dir(name):\n    return "%s/" + name\n'.replace("\\n", "\n") % other)
    done = box.run("install")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (other / "chexbert" / "chexbert.pth").read_bytes() == CX_GOOD and not box.cx_target.exists()


def test_the_cache_variables_of_the_operators_shell_change_neither_check_nor_the_downloads(box):
    other = {name: str(box.root / "elsewhere" / name) for name in CACHE_VARS}
    env = _check_tree(box, 5, 5)
    ok = box.run("check", **dict(env, **other))
    assert ok.returncode == 0 and "[setup] ok chexbert/chexbert.pth" in _lines(ok), ok.stdout + ok.stderr
    fresh = Box(box.root / "fresh")
    done = fresh.run("install", SCRATCH_ROOT=str(fresh.root / "scratch"), CHAT_HOME=str(fresh.root / "chat"), **other)
    assert done.returncode == 0, done.stdout + done.stderr
    assert fresh.cx_target.read_bytes() == CX_GOOD and not (box.root / "elsewhere").exists()
    assert (fresh.stubs / "chexbert.calls").read_text().splitlines()[0].endswith("ENV=[]")
    assert (fresh.stubs / "prefetch.calls").read_text().splitlines()[0].endswith("ENV=[]")
    box.strays += fresh.strays


def test_a_failed_chexbert_download_prints_the_error_and_leaves_nothing_in_the_target_directory(box):
    _write(box.fake_cx / "huggingface_hub.py", (
        'import os\n\n\ndef hf_hub_download(repo_id, filename, **kw):\n'
        '    os.makedirs(os.path.join(os.environ["HF_HOME"], "hub"))\n'
        '    open(os.path.join(os.environ["HF_HOME"], "hub", "partial"), "w").write("half")\n'
        '    raise RuntimeError("simulated network failure")\n'))
    done = box.run("install", SCRATCH_ROOT=str(box.root / "scratch"), CHAT_HOME=str(box.root / "chat"))
    assert done.returncode == 1
    lines = _lines(done)
    assert "[setup] ERROR chexbert weights download failed: bash scripts/vm_remote.sh summary logs/vm_setup_chexbert.log" in lines
    assert lines[-1] == "=== END vm setup: 1 ERROR line(s) above ==="
    assert box.cx_target.parent.is_dir() and list(box.cx_target.parent.iterdir()) == [], "no temp dir or temp file is left"
    assert not (box.root / "scratch" / ".hf" / ".prefetch_ok").exists()


# ── unit ──────────────────────────────────────────────────────────────────────

def _unit(box: Box, **extra: str) -> subprocess.CompletedProcess:
    env = {"CHAT_HOME": str(box.home / "chat_sessions"), "SCRATCH_ROOT": "/srv/scratch", "THREADS": "6"}
    if "MEMINFO" not in extra:
        env["MEMINFO"] = str(box.meminfo())
    env.update(extra)
    return box.run("unit", **env)


def test_unit_fills_every_placeholder_and_pins_private_loopback(box):
    done = _unit(box, MODELS="m3,13d", DRIFT_NOTE="CPU decode here.")
    assert done.returncode == 0, done.stderr
    text = done.stdout
    assert not re.search(r"@[A-Z_]+@", text)
    user = subprocess.check_output(["id", "-un"], text=True).strip()
    group = subprocess.check_output(["id", "-gn"], text=True).strip()
    repo = str(box.repo)
    for line in ("User=" + user, "Group=" + group, "WorkingDirectory=" + repo,
                 'Environment="SLURM_SUBMIT_DIR=%s"' % repo, 'Environment="CHAT_HOME=%s"' % (box.home / "chat_sessions"),
                 'Environment="SCRATCH_ROOT=/srv/scratch"', 'Environment="MODELS=m3,13d"',
                 'Environment="DRIFT_NOTE=CPU decode here."', 'Environment="MODE=private"', 'Environment="BIND=127.0.0.1"',
                 "ExecStart=/bin/bash scripts/serve_chat_h100.sh",
                 "StandardOutput=append:%s/logs/chat_server_vm.log" % repo,
                 "StandardError=append:%s/logs/chat_server_vm.log" % repo, "KillMode=mixed",
                 'Environment="SLURM_CPUS_PER_TASK=6"', 'Environment="CORS_ORIGINS="', "StartLimitIntervalSec=3600", "StartLimitBurst=5",
                 "Restart=always", "RuntimeMaxSec=1d", "MemoryHigh=5850M", "MemoryMax=7020M"):
        assert line in text.splitlines(), line
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert [l for l in code.splitlines() if "CORS" in l] == ['Environment="CORS_ORIGINS="'] and "token" not in code.lower()


def test_unit_defaults_the_models_and_the_drift_note(box):
    text = _unit(box).stdout
    assert 'Environment="MODELS=m3eos,m3"' in text
    assert 'Environment="DRIFT_NOTE=CPU decode on this server; not probed against the published GPU decode."' in text


def test_unit_doubles_percent_signs_and_leaves_ampersands_alone(box):
    done = _unit(box, DRIFT_NOTE="100% sure & %n", HOME="/srv/h%n", CHAT_HOME="/srv/h%n/chat_sessions")
    assert done.returncode == 0
    assert 'Environment="DRIFT_NOTE=100%% sure & %%n"' in done.stdout
    assert 'Environment="CHAT_HOME=/srv/h%%n/chat_sessions"' in done.stdout


def test_unit_never_fills_a_marker_that_a_value_brings(box):
    done = _unit(box, DRIFT_NOTE="see @USER@ and @REPO@")
    assert 'Environment="DRIFT_NOTE=see @USER@ and @REPO@"' in done.stdout


@pytest.mark.parametrize("name,value", [
    ("DRIFT_NOTE", 'has "quote'), ("DRIFT_NOTE", "has \\ backslash"), ("DRIFT_NOTE", "two\nlines"),
    ("CHAT_HOME", '/srv/a"b'), ("CHAT_HOME", "/srv/a\\b"), ("SCRATCH_ROOT", "/srv/a\\b"), ("MODELS", 'm3"'),
])
def test_unit_refuses_a_quote_a_backslash_or_a_newline(box, name, value):
    done = _unit(box, **{name: value})
    assert done.returncode == 1 and done.stdout == "" and done.stderr.startswith("[setup] ERROR " + name)
    assert value not in done.stderr


@pytest.mark.parametrize("models", ["M3", "m3 ", "m3;x", "a" * 33, "m-3"])
def test_unit_refuses_a_bad_models_value(box, models):
    done = _unit(box, MODELS=models)
    assert done.returncode == 1 and done.stdout == "" and "[setup] ERROR MODELS" in done.stderr


def test_unit_refuses_relative_paths(box):
    assert _unit(box, CHAT_HOME="chat").returncode == 1
    assert _unit(box, SCRATCH_ROOT="scratch").returncode == 1


def test_unit_refuses_a_chat_home_other_than_home_chat_sessions_and_says_why(box):
    done = _unit(box, CHAT_HOME="/srv/chat")
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.startswith("[setup] ERROR CHAT_HOME must be $HOME/chat_sessions: the tunnel and vm_remote.sh healthz read")
    assert "/srv/chat" not in done.stderr


def test_unit_takes_threads_from_the_environment_and_defaults_to_the_online_cpus(box):
    assert 'Environment="SLURM_CPUS_PER_TASK=64"' in _unit(box, THREADS="64").stdout
    assert 'Environment="SLURM_CPUS_PER_TASK=256"' in _unit(box, THREADS="256").stdout
    online = subprocess.check_output(["getconf", "_NPROCESSORS_ONLN"], text=True).strip()
    env = {"CHAT_HOME": str(box.home / "chat_sessions"), "MEMINFO": str(box.meminfo())}
    done = box.run("unit", **env)
    if int(online) <= 256:
        assert 'Environment="SLURM_CPUS_PER_TASK=%s"' % online in done.stdout, done.stdout + done.stderr


@pytest.mark.parametrize("threads", ["0", "257", "abc", "-1", "1.5", "4 ", "1000", "0x10"])
def test_unit_refuses_threads_that_are_not_a_whole_number_from_1_to_256(box, threads):
    done = _unit(box, THREADS=threads)
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr == "[setup] ERROR THREADS must be a whole number from 1 to 256\n"


def test_unit_defaults_the_memory_limits_to_75_and_90_percent_of_memtotal_in_whole_mib(box):
    text = _unit(box, MEMINFO=str(box.meminfo(8190976))).stdout
    assert "MemoryHigh=5999M" in text.splitlines() and "MemoryMax=7199M" in text.splitlines()


def test_unit_takes_memory_limits_from_the_environment_and_needs_no_meminfo_then(box):
    text = _unit(box, MEMORY_HIGH="6G", MEMORY_MAX="7000M", MEMINFO=str(box.root / "no_such_meminfo")).stdout
    assert "MemoryHigh=6G" in text.splitlines() and "MemoryMax=7000M" in text.splitlines()
    half = _unit(box, MEMORY_HIGH="4000M").stdout
    assert "MemoryHigh=4000M" in half.splitlines() and "MemoryMax=7020M" in half.splitlines()


@pytest.mark.parametrize("name", ["MEMORY_HIGH", "MEMORY_MAX"])
@pytest.mark.parametrize("value", ["6000", "6000MB", "1.5G", "-1M", "M", "6 G", "max", "50%"])
def test_unit_refuses_a_memory_limit_that_is_not_a_number_and_k_m_or_g(box, name, value):
    done = _unit(box, **{name: value})
    assert done.returncode == 1 and done.stdout == ""
    assert done.stderr.startswith("[setup] ERROR " + name + " must be a number and K, M or G")


def test_unit_without_memtotal_and_without_both_values_is_an_error(box):
    empty = box.root / "meminfo_empty"
    _write(empty, "MemFree: 1 kB\n")
    for meminfo in (str(empty), str(box.root / "no_such_meminfo")):
        done = _unit(box, MEMINFO=meminfo)
        assert done.returncode == 1 and done.stdout == "" and done.stderr.startswith("[setup] ERROR no MemTotal"), done.stderr
        assert _unit(box, MEMINFO=meminfo, MEMORY_HIGH="1G").returncode == 1
        assert _unit(box, MEMINFO=meminfo, MEMORY_HIGH="1G", MEMORY_MAX="2G").returncode == 0


@pytest.mark.parametrize("extra", [
    {"DRIFT_NOTE": 'has "quote'}, {"MODELS": "M3"}, {"CHAT_HOME": "/srv/chat"}, {"THREADS": "0"},
    {"MEMORY_HIGH": "6G", "MEMORY_MAX": "lots"},   # the last check of the render: everything before it passed
])
def test_a_refused_render_prints_nothing_on_stdout_so_a_redirect_never_holds_an_error(box, extra):
    done = _unit(box, **extra)
    assert done.returncode == 1 and done.stdout == "" and done.stderr.startswith("[setup] ERROR ")
    assert done.stderr.count("\n") == 1


def test_a_good_render_writes_nothing_to_stderr(box):
    done = _unit(box)
    assert done.returncode == 0 and done.stderr == "" and done.stdout.startswith("# The chat server as a systemd service")


# ── the root guard ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("command", ["install", "check", "unit"])
def test_every_command_refuses_to_run_as_root_and_does_nothing(box, command):
    _write(box.root / "rootbin" / "id", ID_STUB, executable=True)
    env = {"PATH": "%s:%s:/usr/bin:/bin" % (box.root / "rootbin", box.stubs), "CHAT_HOME": str(box.home / "chat_sessions"),
           "THREADS": "4", "MEMINFO": str(box.meminfo())}
    before = _snapshot(box.root)
    done = box.run(command, **env)
    assert done.returncode == 1
    msg = "[setup] ERROR run as the service user, not root (sudo goes only in front of install and systemctl, outside this script)\n"
    if command == "unit":   # a unit file must never receive the error
        assert done.stdout == "" and done.stderr == msg
    else:
        assert done.stdout == msg
    assert box.uv_calls() == [] and _snapshot(box.root) == before and not (box.repo / "logs").exists()


# ── the unit, read by systemd ─────────────────────────────────────────────────

@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd-analyze is not installed")
def test_systemd_analyze_verifies_a_rendered_unit(box, tmp_path):
    (box.repo / "scripts" / "serve_chat_h100.sh").write_text("#!/bin/bash\n")
    (box.repo / "logs").mkdir(exist_ok=True)
    done = _unit(box)
    assert done.returncode == 0, done.stderr
    unit = tmp_path / "cxr-chat.service"
    unit.write_text(done.stdout)
    checked = subprocess.run(["systemd-analyze", "verify", str(unit)], capture_output=True, text=True, timeout=60)
    assert checked.returncode == 0, checked.stdout + checked.stderr
