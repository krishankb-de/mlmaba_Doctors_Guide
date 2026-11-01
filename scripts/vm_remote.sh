#!/usr/bin/env bash
# vm_remote.sh — the Mac-side door to the always-on VM that runs the chat server as a systemd service.
#
#   bash scripts/vm_remote.sh sync                     # rsync this tree -> $VM_REPO (never --delete)
#   bash scripts/vm_remote.sh summary <log path relative to $VM_REPO>   # R7-safe lines only
#   bash scripts/vm_remote.sh status                   # ActiveState, SubState, MainPID, NRestarts, start time
#   bash scripts/vm_remote.sh start|stop|restart       # sudo -n systemctl <verb> $VM_SERVICE, then status
#   bash scripts/vm_remote.sh healthz                  # GET /healthz on the port in ~/chat_sessions/endpoint
#   bash scripts/vm_remote.sh mem                      # unit memory and swap in MiB, then pid/comm/rss/swap per process
#
# The sibling of chat_remote.sh for the VM: no SLURM fill-in, no MAIN_REPO, and the server is a service, not a job. Reads
# VM_HOST, VM_REPO and VM_SERVICE (default cxr-chat) from scripts/chat_vm.env, or from the file CHAT_VM_ENV names. The server
# listens on the VM's 127.0.0.1 only; you reach it through your own ssh tunnel (app/tunnel/tunnel.sh).
# `healthz` needs curl 7.76 or newer on the VM (--fail-with-body); Ubuntu 24.04 ships 8.5.
#
# Additive only: rsync never gets --delete, and nothing here removes a file on the VM. The stamp goes last and alone, so an
# interrupted transfer can not leave a fresh "clean" stamp over a partial tree.
# R7: there is deliberately no command that prints a log or a file. `summary` shows only lines of the fixed shapes in
# SUMMARY_PATTERN, then masks them: the same pattern, quote() and mask() as chat_remote.sh (tests/test_vm_remote.py compares them
# byte for byte). `mem` prints a process's comm, never its command line, because arguments hold paths. `healthz` prints the
# API's JSON body, for an error answer too.
# Every setting is checked before the first ssh. A refusal names the key and never prints its value.
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2; exit 2; }

cmd="${1:-}"
case "$cmd" in
  sync|summary|status|start|stop|restart|healthz|mem) shift ;;
  *) usage ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CHAT_VM_ENV:-$REPO_ROOT/scripts/chat_vm.env}"
[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE: copy scripts/chat_vm.env.example" >&2; exit 1; }
# A key the caller's shell exports must not stand in for one the file lacks.
unset VM_HOST VM_REPO VM_SERVICE
# shellcheck source=/dev/null
source "$ENV_FILE"
[[ -n "${VM_HOST:-}" ]] || { echo "VM_HOST is not set in $ENV_FILE: copy scripts/chat_vm.env.example" >&2; exit 1; }
[[ -n "${VM_REPO:-}" ]] || { echo "VM_REPO is not set in $ENV_FILE: copy scripts/chat_vm.env.example" >&2; exit 1; }
VM_SERVICE="${VM_SERVICE:-cxr-chat}"

host_re='^[A-Za-z0-9._@-]+$'
service_re='^[a-z0-9][a-z0-9-]{0,63}$'
if [[ ! "$VM_HOST" =~ $host_re || "$VM_HOST" == -* ]]; then
  echo "VM_HOST in $ENV_FILE holds a character no ssh host name has, or starts with -" >&2; exit 1
fi
# The check is lexical: VM_REPO is where rsync writes, so it is absolute, is not /, and has no "." or ".." segment.
if [[ "$VM_REPO" != /* || "${VM_REPO//\//}" == "" || "$VM_REPO/" == */./* || "$VM_REPO/" == */../* || "$VM_REPO" == *"'"* ]]; then
  echo "VM_REPO in $ENV_FILE must be an absolute path below /, with no . or .. segment and no single quote" >&2; exit 1
fi
# rsync hands the remote path to ssh as one argument, ssh joins it into a remote shell string, and that string splits at a
# space: mkdir and rsync would then write to different places.
if [[ "$VM_REPO" == *[[:space:]]* ]]; then
  echo "VM_REPO in $ENV_FILE must not contain whitespace" >&2; exit 1
fi
if [[ ! "$VM_SERVICE" =~ $service_re ]]; then
  echo "VM_SERVICE in $ENV_FILE must be lower-case letters, digits and - (at most 64, not starting with -)" >&2; exit 1
fi

# Lines a summary may show (R7). Everything else in a log stays on the VM. An exception line starts with its class
# name, dotted or not (sqlite3.OperationalError:, urllib.error.URLError:, HTTPException:); mask() blanks the message.
SUMMARY_PATTERN='^(RESULT |\[(probe|golden|gallery|labels|gates|server|setup|compile)\]|=== |ERROR|Traceback|([A-Za-z_][A-Za-z0-9_.]*)?(Error|Exception):|[[:space:]]*(Elapsed \(wall|Maximum resident)|  Missing keys|  prefix_k =)'

# Always call quote as the right-hand side of an assignment (x="$(quote ...)"): its `exit 2` happens in a subshell, and
# inside a larger word, such as the ssh command string, errexit would not see it and an empty argument would be sent.
quote() {   # single-quote one argument for the remote bash; refuse embedded single quotes
  [[ "$1" != *"'"* ]] || { echo "argument contains a single quote: $1" >&2; exit 2; }
  printf "'%s'" "$1"
}

mask() {    # R7: exception text, dicom-style ids and 8+ digit runs (MIMIC ids, but not the tail of a decimal) never reach the terminal
  sed -E 's/(Error|Exception):.*/\1: <msg>/; s/[0-9a-f]{8}(-[0-9a-f]{8}){4}/<id>/g; s/(^|[^0-9.])[0-9]{8,}/\1<num>/g' | cut -c1-300
}

# The scripts below run on the VM: `bash -s -- <unit>` reads one from stdin, so no quoting of them crosses the ssh command line.
read -r -d '' HEALTHZ_SCRIPT <<'REMOTE' || true
endpoint="$HOME/chat_sessions/endpoint"
if [ ! -f "$endpoint" ]; then
  echo 'no endpoint: is the service running? (bash scripts/vm_remote.sh status)' >&2
  exit 1
fi
port="$(python3 -c 'import json, sys
port = json.load(open(sys.argv[1]))["port"]
if type(port) is not int or not 1 <= port <= 65535:
    raise SystemExit(1)
print(port)' "$endpoint" 2>/dev/null)" || { echo 'the endpoint file holds no valid port' >&2; exit 1; }
rc=0
curl -sS --fail-with-body -m 10 "http://127.0.0.1:$port/healthz" || rc=$?
if [ "$rc" -eq 22 ]; then   # 4xx/5xx: curl has printed the API's JSON error body
  echo >&2
  echo 'the server answered /healthz with an error' >&2
  exit 1
elif [ "$rc" -ne 0 ]; then
  echo 'no answer from /healthz' >&2
  exit 1
fi
echo
REMOTE

read -r -d '' MEM_SCRIPT <<'REMOTE' || true
unit="$1"
cgroup_root="${VM_CGROUP_ROOT:-/sys/fs/cgroup}"   # the tests point these two at fake trees
proc_root="${VM_PROC_ROOT:-/proc}"
mib() {   # bytes -> MiB; "[not set]", infinity and the unset sentinel -> n/a
  if [[ "$1" =~ ^[0-9]+$ && "$1" != 18446744073709551615 ]]; then echo "$(( $1 / 1048576 )) MiB"; else echo "n/a"; fi
}
while IFS='=' read -r key value; do
  [ -n "$key" ] && printf '%s %s\n' "$key" "$(mib "$value")"
done < <(systemctl show "$unit" -p MemoryCurrent -p MemoryPeak -p MemorySwapCurrent -p MemorySwapPeak)
cgroup="$(systemctl show "$unit" -p ControlGroup --value)"
procs="$cgroup_root$cgroup/cgroup.procs"
echo 'pid comm rss_mib swap_mib'
if [ -n "$cgroup" ] && [ -r "$procs" ]; then
  while read -r pid; do
    comm="$(cat "$proc_root/$pid/comm" 2>/dev/null)" || continue
    awk -v pid="$pid" -v comm="$comm" '/^VmRSS:/ { rss = $2 } /^VmSwap:/ { swap = $2 } END { printf "%s %s %d %d\n", pid, comm, rss / 1024, swap / 1024 }' "$proc_root/$pid/status"
  done < "$procs"
fi
REMOTE

show_unit() {
  local qunit
  qunit="$(quote "$VM_SERVICE")"
  ssh "$VM_HOST" "systemctl show $qunit -p ActiveState -p SubState -p MainPID -p NRestarts -p ExecMainStartTimestamp"
}

case "$cmd" in
  sync)
    qlogs="$(quote "$VM_REPO/logs")"
    stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ) $(git -C "$REPO_ROOT" rev-parse HEAD)"
    if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- app scripts hybrid_xmamba configs tests)" ]]; then
      stamp="$stamp dirty"
    else
      stamp="$stamp clean"
    fi
    ssh "$VM_HOST" "mkdir -p $qlogs"
    rsync -az --exclude-from="$REPO_ROOT/.rsync-exclude-chat" --exclude=/.sync_stamp "$REPO_ROOT/" "$VM_HOST:$VM_REPO/"
    printf '%s\n' "$stamp" > "$REPO_ROOT/.sync_stamp"
    rsync -az "$REPO_ROOT/.sync_stamp" "$VM_HOST:$VM_REPO/"
    echo "[sync] $REPO_ROOT -> $VM_HOST:$VM_REPO/ ($stamp)"
    ;;
  summary)
    log="$(quote "$VM_REPO/${1:?log path}")"
    qpattern="$(quote "$SUMMARY_PATTERN")"
    ssh "$VM_HOST" "grep -aE $qpattern $log | tail -n 200" | mask
    ;;
  status)
    show_unit
    ;;
  start|stop|restart)
    qunit="$(quote "$VM_SERVICE")"
    ssh "$VM_HOST" "sudo -n systemctl $cmd $qunit"
    show_unit
    ;;
  healthz)
    ssh "$VM_HOST" "bash -s" <<< "$HEALTHZ_SCRIPT"
    ;;
  mem)
    qunit="$(quote "$VM_SERVICE")"
    ssh "$VM_HOST" "bash -s -- $qunit" <<< "$MEM_SCRIPT"
    ;;
esac
