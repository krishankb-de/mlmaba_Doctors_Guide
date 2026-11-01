"""The Mac-side door to the always-on VM (scripts/vm_remote.sh) and the files that keep its settings local.

Nothing here touches a real host. Every test runs a COPY of vm_remote.sh inside a throwaway git tree, with `ssh` and
`rsync` replaced by stub executables that record their argv (and, for ssh, the script sent on stdin), and CHAT_VM_ENV
pointing at a fake env file. The summary and healthz tests make the ssh stub run what it is sent against synthetic
data on this machine instead. Synthetic data only (R7).
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
VM_SH = REPO_ROOT / "scripts" / "vm_remote.sh"
CLUSTER_SH = REPO_ROOT / "scripts" / "chat_remote.sh"
ENV_EXAMPLE = REPO_ROOT / "scripts" / "chat_vm.env.example"
EXCLUDE = REPO_ROOT / ".rsync-exclude-chat"
CHECKS_YML = REPO_ROOT / ".github" / "workflows" / "checks.yml"

# The Mac's /bin/bash is 3.2: the oldest shell this script has to work in.
BASH = "/bin/bash" if os.path.exists("/bin/bash") else "bash"

STATUS_COMMAND = ("systemctl show 'cxr-chat' -p ActiveState -p SubState -p MainPID -p NRestarts"
                  " -p ExecMainStartTimestamp")
STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z [0-9a-f]{40} (clean|dirty)$")


def fake_env(**overrides: str) -> str:
    """The text of a chat_vm.env; values are double-quoted so a test can put a single quote in one."""
    values = {"VM_HOST": "fakevm", "VM_REPO": "/fake/chat_ui", "VM_SERVICE": "cxr-chat"}
    values.update(overrides)
    return "".join('%s="%s"\n' % (key, value) for key, value in values.items())


# One stub serves as both ssh and rsync: it appends "@@ <name>", one argument per line, "@@END" to calls.log, saves what
# it got on stdin to $STUB_DIR/<name>.stdin, prints $STUB_DIR/<name>.stdout when that file exists, and exits with the
# code in <name>.rc (default 0).
STUB = """#!/bin/bash
{ echo "@@ __NAME__"; for a in "$@"; do printf '%s\\n' "$a"; done; echo "@@END"; } >> "$STUB_DIR/calls.log"
if [ ! -t 0 ]; then cat > "$STUB_DIR/__NAME__.stdin"; fi
if [ -f "$STUB_DIR/__NAME__.stdout" ]; then cat "$STUB_DIR/__NAME__.stdout"; fi
if [ -f "$STUB_DIR/__NAME__.rc" ]; then exit "$(cat "$STUB_DIR/__NAME__.rc")"; fi
exit 0
"""

# Stands in for ssh: $1 is the host, $2 the command line the script sends. Run that line here, with the script's stdin.
RUNNING_SSH = """#!/bin/bash
exec bash -c "$2"
"""


class Sandbox:
    """A throwaway git tree holding a copy of vm_remote.sh, with ssh/rsync stubbed out."""

    def __init__(self, root: Path, env_text: Optional[str] = None):
        self.repo = root / "repo"
        self.stubs = root / "stubs"
        self.bin = root / "bin"
        (self.repo / "scripts").mkdir(parents=True)
        self.stubs.mkdir()
        self.bin.mkdir()
        shutil.copy(str(VM_SH), str(self.repo / "scripts" / "vm_remote.sh"))
        shutil.copy(str(EXCLUDE), str(self.repo / ".rsync-exclude-chat"))
        for name in ("ssh", "rsync"):
            stub = self.bin / name
            stub.write_text(STUB.replace("__NAME__", name))
            stub.chmod(0o755)
        self.env_file = root / "vm.env"
        self.env_file.write_text(env_text if env_text is not None else fake_env())
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "init")

    @staticmethod
    def _clean_env() -> Dict[str, str]:
        # No GIT_* leaking in (a hook running pytest would otherwise point git at the real repo), and none of the
        # settings vm_remote.sh reads.
        drop = ("VM_HOST", "VM_REPO", "VM_SERVICE", "CHAT_VM_ENV")
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_") and k not in drop}
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_NOSYSTEM"] = "1"
        return env

    def git(self, *args: str) -> str:
        cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
               "-c", "core.hooksPath=%s" % os.devnull] + list(args)
        return subprocess.run(cmd, cwd=str(self.repo), env=self._clean_env(), check=True,
                              capture_output=True, text=True).stdout

    def run(self, *args: str, env_file: Optional[str] = None,
            extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        env = self._clean_env()
        env["PATH"] = "%s%s%s" % (self.bin, os.pathsep, env["PATH"])
        env["STUB_DIR"] = str(self.stubs)
        env["CHAT_VM_ENV"] = env_file if env_file is not None else str(self.env_file)
        env.update(extra_env or {})
        return subprocess.run([BASH, str(self.repo / "scripts" / "vm_remote.sh")] + list(args),
                              cwd=str(self.repo), env=env, stdin=subprocess.DEVNULL, timeout=60,
                              capture_output=True, text=True)

    def calls(self) -> List[List[str]]:
        """Every stub invocation so far, in order, as [name, arg1, arg2, ...]."""
        log = self.stubs / "calls.log"
        if not log.exists():
            return []
        calls, current = [], None
        for line in log.read_text().splitlines():
            if line.startswith("@@ "):
                current = [line[3:]]
            elif line == "@@END":
                calls.append(current)
                current = None
            else:
                current.append(line)
        return calls

    def ssh_stdin(self) -> str:
        return (self.stubs / "ssh.stdin").read_text()

    def run_ssh_locally(self) -> None:
        (self.bin / "ssh").write_text(RUNNING_SSH)


@pytest.fixture
def sandbox(tmp_path):
    return Sandbox(tmp_path)


def _code(path: Path) -> str:
    """The script minus its comment lines (the header names the flag it refuses to use)."""
    return "\n".join(l for l in path.read_text().splitlines() if not l.lstrip().startswith("#"))


# ── static checks ─────────────────────────────────────────────────────────────

def test_script_passes_bash_syntax_check():
    done = subprocess.run([BASH, "-n", str(VM_SH)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_vm_remote_never_deletes_or_removes():
    """The VM side is additive only. No rsync --delete (or --remove-*), no rm."""
    code = _code(VM_SH)
    assert "--delete" not in code and "--remove" not in code
    assert not re.search(r"(^|[\s;&|(])rm(\s|$)", code, re.M)


def test_vm_remote_has_no_command_that_prints_a_log_or_a_file():
    code = _code(VM_SH)
    # the only cat reads <proc root>/<pid>/comm, and the only tail trims the filtered summary
    assert len(re.findall(r"\bcat\b", code)) == 1 and '"$proc_root/$pid/comm"' in code
    assert len(re.findall(r"\btail -", code)) == 1 and "tail -n 200" in code
    for word in ("journalctl", "cmdline", "less ", "head ", "sed -n"):
        assert word not in code, word


def test_every_quote_call_is_the_right_hand_side_of_an_assignment():
    """quote()'s `exit 2` is lost unless errexit sees the assignment: inline in a larger word it sends an empty argument."""
    lines = [l for l in _code(VM_SH).splitlines() if "$(quote" in l]
    assert len(lines) >= 5
    for line in lines:
        assert re.search(r'\w+\+?=\(?"\$\(quote ', line), line


def test_script_never_enables_public_mode_a_wide_bind_or_cors():
    code = _code(VM_SH).lower()
    for word in ("cors", "0.0.0.0", "bind=", "public"):
        assert word not in code, word


def _definition(path: Path, start: str, closing: str) -> str:
    text = path.read_text()
    begin = text.index(start)
    return text[begin:text.index(closing, begin) + len(closing)]


def test_summary_definitions_are_byte_identical_to_chat_remote_sh():
    """R7: the VM's summary filter is exactly as strict as the cluster one. Pattern, quote() and mask() are copies."""
    for start, closing in (("SUMMARY_PATTERN='", "'\n"), ("quote() {", "\n}\n"), ("mask() {", "\n}\n")):
        here, there = _definition(VM_SH, start, closing), _definition(CLUSTER_SH, start, closing)
        assert here == there, start
        assert len(here) > 60


# ── settings: example, env file, refusals ────────────────────────────────────

def test_env_example_holds_placeholders_for_the_three_keys():
    out = subprocess.run([BASH, "-c", 'source "$1"; printf "%s\\n" "$VM_HOST" "$VM_REPO" "$VM_SERVICE"', "_", str(ENV_EXAMPLE)],
                         capture_output=True, text=True, check=True).stdout.splitlines()
    host, repo, service = out
    assert host == "<vm-alias>" and repo.startswith("/home/<vm-user>/") and service == "cxr-chat"
    text = ENV_EXAMPLE.read_text()
    for key in ("VM_HOST", "VM_REPO", "VM_SERVICE"):
        assert re.search(r"^# %s: " % key, text, re.M), key


def test_refuses_without_the_env_file_and_names_the_example(sandbox):
    done = sandbox.run("sync", env_file="/nonexistent")
    assert done.returncode == 1
    assert "missing /nonexistent" in done.stderr and "chat_vm.env.example" in done.stderr
    assert sandbox.calls() == []
    assert not (sandbox.repo / ".sync_stamp").exists()


@pytest.mark.parametrize("key", ["VM_HOST", "VM_REPO"])
def test_an_env_file_missing_a_key_is_refused_whatever_the_callers_shell_exports(tmp_path, key):
    text = "".join(l for l in fake_env().splitlines(keepends=True) if not l.startswith(key + "="))
    box = Sandbox(tmp_path, env_text=text)
    done = box.run("status", extra_env={"VM_HOST": "exported-host", "VM_REPO": "/exported/repo"})
    assert done.returncode == 1 and key in done.stderr, done.stderr
    assert box.calls() == []


@pytest.mark.parametrize("value", ["two words", "a;reboot", "x|y", "$(id)", "a/b", "-oProxyCommand=x", "-x", "it's", "a:b"])
def test_a_vm_host_that_no_ssh_host_name_could_be_is_refused_without_echoing_it(tmp_path, value):
    box = Sandbox(tmp_path, env_text=fake_env(VM_HOST=value))
    for args in (("sync",), ("status",), ("summary", "logs/x.log")):
        done = box.run(*args)
        assert done.returncode == 1 and "VM_HOST" in done.stderr, (args, done.stderr)
        assert value not in done.stderr
    assert box.calls() == [] and not (box.repo / ".sync_stamp").exists()


@pytest.mark.parametrize("value", [
    "relative/path", "chat", "/", "//", "///", "/fake/./chat", "/fake/../chat", "/fake/..", "/.", "/fake/it's",
])
def test_a_vm_repo_that_is_not_a_plain_absolute_path_is_refused_without_echoing_it(tmp_path, value):
    box = Sandbox(tmp_path, env_text=fake_env(VM_REPO=value))
    for args in (("sync",), ("status",), ("healthz",), ("summary", "logs/x.log")):
        done = box.run(*args)
        assert done.returncode == 1 and "VM_REPO" in done.stderr, (args, done.stderr)
        assert len(value) <= 3 or value not in done.stderr    # "/" is part of every message
    assert box.calls() == [] and not (box.repo / ".sync_stamp").exists()


@pytest.mark.parametrize("value", ["/fake/a b", "/fake/a\tb", "/fake/chat ", "/fake/ chat"])
def test_a_vm_repo_with_whitespace_is_refused_without_echoing_it(tmp_path, value):
    """ssh splits the remote path at whitespace, so mkdir and rsync would write to different places."""
    box = Sandbox(tmp_path, env_text=fake_env(VM_REPO=value))
    for args in (("sync",), ("status",), ("healthz",), ("summary", "logs/x.log")):
        done = box.run(*args)
        assert done.returncode == 1 and "VM_REPO" in done.stderr and "whitespace" in done.stderr, (args, done.stderr)
        assert value not in done.stderr and "/fake" not in done.stderr
    assert box.calls() == [] and not (box.repo / ".sync_stamp").exists()


@pytest.mark.parametrize("value", ["Cxr", "cxr_chat", "-cxr", "cxr chat", "cxr.service", "a" * 65, "cxr;reboot", "it's", "../x"])
def test_a_vm_service_that_is_not_a_unit_name_is_refused_without_echoing_it(tmp_path, value):
    box = Sandbox(tmp_path, env_text=fake_env(VM_SERVICE=value))
    for args in (("status",), ("restart",), ("mem",)):
        done = box.run(*args)
        assert done.returncode == 1 and "VM_SERVICE" in done.stderr, (args, done.stderr)
        assert value not in done.stderr
    assert box.calls() == []


def test_vm_service_defaults_to_cxr_chat(tmp_path):
    text = "".join(l for l in fake_env().splitlines(keepends=True) if not l.startswith("VM_SERVICE="))
    box = Sandbox(tmp_path, env_text=text)
    assert box.run("status", extra_env={"VM_SERVICE": "exported-unit"}).returncode == 0
    assert box.calls() == [["ssh", "fakevm", STATUS_COMMAND]]


@pytest.mark.parametrize("repo, host", [("/fake//chat/", "user@vm-1.example"), ("/home/a-b/x", "vm_alias"), ("/fake/..hidden/x", "h")])
def test_plain_paths_and_host_names_are_accepted(tmp_path, repo, host):
    box = Sandbox(tmp_path, env_text=fake_env(VM_REPO=repo, VM_HOST=host))
    assert box.run("status").returncode == 0


def test_a_single_quote_in_the_log_path_is_refused_before_ssh(sandbox):
    done = sandbox.run("summary", "logs/it's.log")
    assert done.returncode == 2 and "single quote" in done.stderr
    done = sandbox.run("summary")
    assert done.returncode != 0
    assert sandbox.calls() == []


@pytest.mark.parametrize("args", [(), ("bogus",), ("submit",), ("log", "x")])
def test_missing_or_unknown_command_prints_usage_on_stderr_and_exits_2(sandbox, args):
    done = sandbox.run(*args)
    assert done.returncode == 2 and done.stdout == ""
    header = []
    for line in VM_SH.read_text().splitlines()[1:]:
        if not line.startswith("#"):
            break
        header.append(re.sub(r"^# ?", "", line))
    assert done.stderr.splitlines() == header, "the whole header comment"
    for sub in ("sync", "summary", "status", "start|stop|restart", "healthz", "mem"):
        assert "vm_remote.sh %s" % sub in done.stderr, sub
    assert sandbox.calls() == []


def test_usage_needs_no_env_file(sandbox):
    assert sandbox.run("bogus", env_file="/nonexistent").returncode == 2


# ── sync ──────────────────────────────────────────────────────────────────────

def test_sync_sends_the_tree_without_the_stamp_and_then_the_stamp_alone_last(sandbox):
    done = sandbox.run("sync")
    assert done.returncode == 0, done.stderr
    repo = sandbox.repo
    assert sandbox.calls() == [
        ["ssh", "fakevm", "mkdir -p '/fake/chat_ui/logs'"],
        ["rsync", "-az", "--exclude-from=%s/.rsync-exclude-chat" % repo, "--exclude=/.sync_stamp", "%s/" % repo,
         "fakevm:/fake/chat_ui/"],
        ["rsync", "-az", "%s/.sync_stamp" % repo, "fakevm:/fake/chat_ui/"],
    ]
    assert done.stdout.startswith("[sync] %s -> fakevm:/fake/chat_ui/ (" % repo)
    assert STAMP_RE.match((repo / ".sync_stamp").read_text().strip())


def test_sync_sends_no_delete_flag_and_no_sed_fill_in(sandbox):
    assert sandbox.run("sync").returncode == 0
    for call in sandbox.calls():
        for arg in call:
            assert "--delete" not in arg and "--remove" not in arg, call
            assert "sed" not in arg and "@SLURM" not in arg, call
    assert [c[0] for c in sandbox.calls()] == ["ssh", "rsync", "rsync"]


def test_a_failed_tree_transfer_never_sends_or_rewrites_the_stamp(sandbox):
    (sandbox.repo / ".sync_stamp").write_text("OLD STAMP\n")
    (sandbox.stubs / "rsync.rc").write_text("23\n")
    done = sandbox.run("sync")
    assert done.returncode == 23
    assert [c[0] for c in sandbox.calls()] == ["ssh", "rsync"], "no second rsync after a failed first one"
    assert (sandbox.repo / ".sync_stamp").read_text() == "OLD STAMP\n"


def test_sync_stamp_is_time_head_and_cleanliness(sandbox):
    head = sandbox.git("rev-parse", "HEAD").strip()
    stamp_file = sandbox.repo / ".sync_stamp"

    assert sandbox.run("sync").returncode == 0
    stamp = stamp_file.read_text().strip()
    assert STAMP_RE.match(stamp), stamp
    assert stamp.split()[1] == head and stamp.endswith(" clean")

    # an untracked file outside the five source dirs does not make the stamp dirty
    (sandbox.repo / "notes.txt").write_text("scratch\n")
    assert sandbox.run("sync").returncode == 0
    assert stamp_file.read_text().strip().endswith(" clean")

    # one inside scripts/ does
    (sandbox.repo / "scripts" / "extra.sh").write_text("echo hi\n")
    assert sandbox.run("sync").returncode == 0
    dirty = stamp_file.read_text().strip()
    assert STAMP_RE.match(dirty) and dirty.split()[1] == head and dirty.endswith(" dirty")


# ── status / start / stop / restart ──────────────────────────────────────────

def test_status_is_one_systemctl_show(sandbox):
    (sandbox.stubs / "ssh.stdout").write_text("ActiveState=active\nSubState=running\n")
    done = sandbox.run("status")
    assert done.returncode == 0, done.stderr
    assert done.stdout == "ActiveState=active\nSubState=running\n"
    assert sandbox.calls() == [["ssh", "fakevm", STATUS_COMMAND]]


@pytest.mark.parametrize("verb", ["start", "stop", "restart"])
def test_start_stop_restart_run_sudo_systemctl_and_then_show_the_status(tmp_path, verb):
    box = Sandbox(tmp_path, env_text=fake_env(VM_SERVICE="chat-2"))
    (box.stubs / "ssh.stdout").write_text("ActiveState=active\n")
    done = box.run(verb)
    assert done.returncode == 0, done.stderr
    assert box.calls() == [
        ["ssh", "fakevm", "sudo -n systemctl %s 'chat-2'" % verb],
        ["ssh", "fakevm", STATUS_COMMAND.replace("'cxr-chat'", "'chat-2'")],
    ]
    assert done.stdout == "ActiveState=active\nActiveState=active\n"


def test_a_failed_verb_stops_before_the_status(sandbox):
    (sandbox.stubs / "ssh.rc").write_text("1\n")
    done = sandbox.run("restart")
    assert done.returncode == 1
    assert len(sandbox.calls()) == 1


# ── healthz ───────────────────────────────────────────────────────────────────

def test_healthz_sends_a_script_on_stdin_to_bash_s(sandbox):
    assert sandbox.run("healthz").returncode == 0
    assert sandbox.calls() == [["ssh", "fakevm", "bash -s"]]
    script = sandbox.ssh_stdin()
    assert '$HOME/chat_sessions/endpoint' in script
    assert "curl -sS --fail-with-body -m 10 \"http://127.0.0.1:$port/healthz\"" in script
    assert "no endpoint: is the service running? (bash scripts/vm_remote.sh status)" in script


# Records its argv, prints curl.body (default: an ok body) and exits with curl.rc (default 0).
CURL_STUB = """#!/bin/bash
printf '%s\\n' "$@" > "$STUB_DIR/curl.args"
if [ -f "$STUB_DIR/curl.body" ]; then cat "$STUB_DIR/curl.body"; else echo '{"status": "ok"}'; fi
if [ -f "$STUB_DIR/curl.rc" ]; then exit "$(cat "$STUB_DIR/curl.rc")"; fi
exit 0
"""


def _healthz(tmp_path: Path, endpoint_text: Optional[str], curl_rc: Optional[int] = None,
             curl_body: Optional[str] = None) -> "tuple":
    box = Sandbox(tmp_path / "box")
    if curl_rc is not None:
        (box.stubs / "curl.rc").write_text("%d\n" % curl_rc)
    if curl_body is not None:
        (box.stubs / "curl.body").write_text(curl_body)
    box.run_ssh_locally()
    curl = box.bin / "curl"
    curl.write_text(CURL_STUB)
    curl.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    if endpoint_text is not None:
        (home / "chat_sessions").mkdir()
        (home / "chat_sessions" / "endpoint").write_text(endpoint_text)
    done = box.run("healthz", extra_env={"HOME": str(home)})
    args = box.stubs / "curl.args"
    return done, (args.read_text().split() if args.exists() else None)


def test_healthz_without_an_endpoint_file_says_so_and_exits_1(tmp_path):
    done, curl_args = _healthz(tmp_path, None)
    assert done.returncode == 1
    assert "no endpoint: is the service running? (bash scripts/vm_remote.sh status)" in done.stdout + done.stderr
    assert curl_args is None


def test_healthz_reads_the_port_and_prints_the_body(tmp_path):
    done, curl_args = _healthz(tmp_path, json.dumps({"port": 43210, "extra": "x"}))
    assert done.returncode == 0, done.stderr
    assert curl_args == ["-sS", "--fail-with-body", "-m", "10", "http://127.0.0.1:43210/healthz"]
    assert done.stdout.strip() == '{"status": "ok"}'


@pytest.mark.parametrize("text", [
    '{"port": 0}', '{"port": 65536}', '{"port": -1}', '{"port": "8000"}', '{"port": 80.5}', '{"port": true}',
    '{"host": "x"}', "not json", "", "[8000]",
])
def test_healthz_refuses_an_endpoint_without_a_valid_port(tmp_path, text):
    done, curl_args = _healthz(tmp_path, text)
    assert done.returncode == 1, (text, done.stdout, done.stderr)
    assert curl_args is None, "curl must not run"


def test_healthz_error_answer_prints_the_body_says_so_and_exits_1(tmp_path):
    body = '{"detail": "model not loaded"}\n'
    done, curl_args = _healthz(tmp_path, json.dumps({"port": 8000}), curl_rc=22, curl_body=body)
    assert done.returncode == 1
    assert done.stdout == body
    assert "answered /healthz with an error" in done.stderr
    assert "no answer" not in done.stderr


def test_healthz_without_an_answer_says_so_and_exits_1(tmp_path):
    done, _ = _healthz(tmp_path, json.dumps({"port": 8000}), curl_rc=7, curl_body="")
    assert done.returncode == 1
    assert "no answer from /healthz" in done.stderr and "answered" not in done.stderr


def test_healthz_accepts_the_port_range_ends(tmp_path):
    for port in (1, 65535):
        done, curl_args = _healthz(tmp_path / str(port), json.dumps({"port": port}))
        assert done.returncode == 0 and curl_args[-1] == "http://127.0.0.1:%d/healthz" % port


# ── mem ───────────────────────────────────────────────────────────────────────

def test_mem_sends_a_script_on_stdin_with_the_unit_as_its_argument(sandbox):
    assert sandbox.run("mem").returncode == 0
    assert sandbox.calls() == [["ssh", "fakevm", "bash -s -- 'cxr-chat'"]]
    script = sandbox.ssh_stdin()
    assert "systemctl show \"$unit\" -p MemoryCurrent -p MemoryPeak -p MemorySwapCurrent -p MemorySwapPeak" in script
    assert "systemctl show \"$unit\" -p ControlGroup" in script
    assert 'cgroup_root="${VM_CGROUP_ROOT:-/sys/fs/cgroup}"' in script
    assert 'proc_root="${VM_PROC_ROOT:-/proc}"' in script
    assert "$cgroup_root$cgroup/cgroup.procs" in script
    assert "$proc_root/$pid/comm" in script and "$proc_root/$pid/status" in script
    assert "VmRSS" in script and "VmSwap" in script
    assert "cmdline" not in script, "arguments hold paths: comm only"


SYSTEMCTL_STUB = """#!/bin/bash
# show <unit> -p A -p B ...: print what the test queued; --value prints just the value of the one property
if [[ "$*" == *ControlGroup* ]]; then echo "/system.slice/nonexistent-unit.service"; exit 0; fi
cat "$STUB_DIR/systemctl.show"
"""


def test_mem_converts_bytes_to_mib_and_prints_n_a_for_unset_values(tmp_path):
    box = Sandbox(tmp_path)
    box.run_ssh_locally()
    systemctl = box.bin / "systemctl"
    systemctl.write_text(SYSTEMCTL_STUB)
    systemctl.chmod(0o755)
    (box.stubs / "systemctl.show").write_text(
        "MemoryCurrent=1073741824\nMemoryPeak=[not set]\nMemorySwapCurrent=infinity\nMemorySwapPeak=3145728\n"
        "MemoryMax=18446744073709551615\n")
    done = box.run("mem")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "MemoryCurrent 1024 MiB",
        "MemoryPeak n/a",
        "MemorySwapCurrent n/a",
        "MemorySwapPeak 3 MiB",
        "MemoryMax n/a",
        "pid comm rss_mib swap_mib",
    ]


SYSTEMCTL_CGROUP_STUB = """#!/bin/bash
if [[ "$*" == *ControlGroup* ]]; then echo "/fake.slice/cxr-chat.service"; exit 0; fi
echo "MemoryCurrent=1048576"
"""


def _proc(root: Path, pid: int, comm: str, status: Optional[str]) -> None:
    (root / str(pid)).mkdir(parents=True)
    (root / str(pid) / "comm").write_text(comm + "\n")
    (root / str(pid) / "cmdline").write_bytes(b"/secret/path/SYNTHETIC\0--token=SYNTHETIC\0")
    if status is not None:
        (root / str(pid) / "status").write_text(status)


def test_mem_lists_pid_comm_rss_and_swap_per_process_and_never_the_command_line(tmp_path):
    box = Sandbox(tmp_path / "box")
    box.run_ssh_locally()
    systemctl = box.bin / "systemctl"
    systemctl.write_text(SYSTEMCTL_CGROUP_STUB)
    systemctl.chmod(0o755)
    cgroup, proc = tmp_path / "cgroup", tmp_path / "proc"
    (cgroup / "fake.slice" / "cxr-chat.service").mkdir(parents=True)
    # 101 and 102 exist; 103 has already exited (no /proc entry); 104 has a comm but no swap line
    (cgroup / "fake.slice" / "cxr-chat.service" / "cgroup.procs").write_text("101\n102\n103\n104\n")
    _proc(proc, 101, "python3", "Name:\tpython3\nVmRSS:\t  2097152 kB\nVmSwap:\t  10240 kB\n")
    _proc(proc, 102, "uvicorn", "VmRSS:\t1536 kB\nVmSwap:\t0 kB\n")
    _proc(proc, 104, "sh", "VmRSS:\t3072 kB\n")
    done = box.run("mem", extra_env={"VM_CGROUP_ROOT": str(cgroup), "VM_PROC_ROOT": str(proc)})
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        "MemoryCurrent 1 MiB",
        "pid comm rss_mib swap_mib",
        "101 python3 2048 10",
        "102 uvicorn 1 0",
        "104 sh 3 0",
    ]
    assert "SYNTHETIC" not in done.stdout + done.stderr


def test_mem_reads_the_real_cgroup_and_proc_roots_unless_told_otherwise(sandbox):
    assert sandbox.run("mem").returncode == 0
    script = sandbox.ssh_stdin()
    assert script.count("/sys/fs/cgroup") == 1 and script.count("VM_PROC_ROOT:-/proc}") == 1


# ── summary (R7) ──────────────────────────────────────────────────────────────

SUMMARY_PATTERN = (
    r"^(RESULT |\[(probe|golden|gallery|labels|gates|server|setup|compile)\]|=== |ERROR|Traceback"
    r"|([A-Za-z_][A-Za-z0-9_.]*)?(Error|Exception):|[[:space:]]*(Elapsed \(wall|Maximum resident)|  Missing keys|  prefix_k =)"
)


def test_summary_sends_only_a_grep_of_the_fixed_pattern(sandbox):
    assert sandbox.run("summary", "logs/chat_server.log").returncode == 0
    assert sandbox.calls() == [[
        "ssh", "fakevm",
        "grep -aE '%s' '/fake/chat_ui/logs/chat_server.log' | tail -n 200" % SUMMARY_PATTERN,
    ]]


def _summary_of(tmp_path: Path, log_lines: List[str]) -> List[str]:
    """`summary` end to end over a synthetic log: the script's own grep pattern, then its own mask (R7: synthetic data only)."""
    vm = tmp_path / "vm"
    (vm / "logs").mkdir(parents=True)
    (vm / "logs" / "x.log").write_text("\n".join(log_lines) + "\n")
    box = Sandbox(tmp_path / "box", env_text=fake_env(VM_REPO=str(vm)))
    box.run_ssh_locally()
    done = box.run("summary", "logs/x.log")
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


def test_summary_shows_only_the_allowed_shapes(tmp_path):
    keep = [
        "=== chat server: unit=cxr-chat ===",
        "[setup] ok",
        "[probe] n=4 median_ms=12.5",
        "[golden] pass 3/3",
        "[gallery] rows=1000",
        "[labels] micro_f1=0.47",
        "[gates] ok",
        "[server] listening on 127.0.0.1:8000",
        "[compile] skipped",
        'RESULT {"rouge_l": 0.19}',
        "ERROR missing something",
        "Traceback (most recent call last):",
        "RuntimeError: boom",
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:12.30",
        "\tMaximum resident set size (kbytes): 123456",
        "  Missing keys: []",
        "  prefix_k = 32",
    ]
    drop = [
        "The heart size is normal. No pleural effusion or pneumothorax.",
        "s12345678 PA view, no acute cardiopulmonary process",
        '  File "x.py", line 3, in <module>',
        "[other] some debug line",
        "[setup",
        " [setup] indented",
        "warning: not an error line",
        "a report that says ERROR inside the text",
        "XRESULT {}",
        "Study s12345678 Error: x",
        "",
    ]
    interleaved = []
    for i in range(max(len(keep), len(drop))):
        if i < len(drop):
            interleaved.append(drop[i])
        if i < len(keep):
            interleaved.append(keep[i])
    expected = ["RuntimeError: <msg>" if l == "RuntimeError: boom" else l for l in keep]    # mask() blanks the message
    assert _summary_of(tmp_path, interleaved) == expected


def test_summary_masks_long_digit_runs_ids_and_exception_text(tmp_path):
    shown = _summary_of(tmp_path, [
        "[probe] study s12345678 subject p87654321 job 2589358",
        "[gallery] 0a1b2c3d-1a2b3c4d-5e6f7a8b-9c0d1e2f-3a4b5c6d kept",
        'RESULT {"rouge_l": 0.1899234512, "n": 12345678901234}',
        "ValueError: could not parse 'SYNTHETIC findings'",
        "sqlite3.OperationalError: database is locked SYNTHETIC",
        "[probe] KeyError: 'SYNTHETIC'",
        "=== " + "x" * 500,
    ])
    assert shown == [
        "[probe] study s<num> subject p<num> job 2589358",
        "[gallery] <id> kept",
        'RESULT {"rouge_l": 0.1899234512, "n": <num>}',
        "ValueError: <msg>",
        "sqlite3.OperationalError: <msg>",
        "[probe] KeyError: <msg>",
        "=== " + "x" * 296,
    ]
    assert "SYNTHETIC" not in "\n".join(shown)


def test_summary_is_capped_at_the_last_200_matching_lines(tmp_path):
    shown = _summary_of(tmp_path, ["[probe] %d\n[other] %d" % (i, i) for i in range(250)])
    assert shown == ["[probe] %d" % i for i in range(50, 250)]


# ── settings stay local ──────────────────────────────────────────────────────

def test_the_vm_env_file_is_gitignored_and_the_example_is_not():
    ignored = [l.strip() for l in (REPO_ROOT / ".gitignore").read_text().splitlines()]
    assert "scripts/chat_vm.env" in ignored
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")

    def check_ignore(path: str) -> int:
        return subprocess.run(["git", "check-ignore", "-q", path], cwd=str(REPO_ROOT)).returncode
    assert check_ignore("scripts/chat_vm.env") == 0
    assert check_ignore("scripts/chat_vm.env.example") == 1
    assert check_ignore("scripts/vm_remote.sh") == 1


def test_the_exclude_file_keeps_the_vm_env_file_at_home():
    lines = [l for l in EXCLUDE.read_text().splitlines() if l and not l.startswith("#")]
    assert "/scripts/chat_vm.env" in lines and "/scripts/chat_cluster.env" in lines


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync not installed")
def test_the_vm_env_file_does_not_cross_under_a_real_rsync(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    for rel in ("scripts/vm_remote.sh", "scripts/chat_vm.env", "scripts/chat_vm.env.example"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text("x\n")
    dst.mkdir()
    done = subprocess.run(["rsync", "-az", "--exclude-from=%s" % EXCLUDE, "%s/" % src, "%s/" % dst],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in (dst / "scripts").iterdir()) == ["chat_vm.env.example", "vm_remote.sh"]


def test_the_local_only_file_regex_in_checks_yml_matches_the_vm_env_file():
    patterns = [l for l in CHECKS_YML.read_text().splitlines() if r"scripts/chat_cluster\.env" in l and "grep -E" in l]
    assert patterns, "the local-only-file regex moved"
    for line in patterns:
        regex = re.search(r"grep -E '([^']+)'", line).group(1)
        assert re.search(regex, "scripts/chat_vm.env"), line
        assert re.search(regex, "scripts/chat_cluster.env"), line
        assert not re.search(regex, "scripts/chat_vm.env.example"), line
