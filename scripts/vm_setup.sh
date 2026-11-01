#!/bin/bash
# ============================================================================
# The chat server on a Linux VM (x86_64, systemd): the same wrapper, scripts/serve_chat_h100.sh, kept running by a service.
#   bash scripts/vm_setup.sh [install]   # build the environment in this repo copy; the default
#   bash scripts/vm_setup.sh check       # one line per required input, then the image counts
#   bash scripts/vm_setup.sh unit        # print the systemd unit; install it with
#       bash scripts/vm_setup.sh unit > ~/cxr-chat.service && sudo install -m 0644 ~/cxr-chat.service /etc/systemd/system/cxr-chat.service && sudo systemctl daemon-reload
#     `unit` prints the unit to stdout only when the whole render worked; a refusal prints its [setup] ERROR line to stderr and nothing to stdout.
# Knobs (environment): CHAT_HOME ($HOME/chat_sessions; `unit` accepts no other, because the tunnel and vm_remote.sh healthz read
# ~/chat_sessions/endpoint), SCRATCH_ROOT ($HOME/vm_scratch), VM_TOOLS ($HOME/.local/share/cxr-chat), GALLERY ($CHAT_HOME/gallery/g13d_m3_v1),
# MODELS (m3eos,m3), DRIFT_NOTE (the CPU-decode sentence below; after measuring the published model's CPU decode on this server, e.g.
# `scripts/chat_realturn_probe.py http://localhost:8020 --rows 0-19` with published_equal on all rows, re-render the unit with
# DRIFT_NOTE="CPU decode; matched the published GPU decode on N/20 probe studies"). For `unit` only: THREADS (the online CPUs, 1 to 256), MEMORY_HIGH and
# MEMORY_MAX (75% and 90% of MemTotal in $MEMINFO, /proc/meminfo; or a number and K, M or G).
# Every command refuses to run as root: the service user owns the files, and sudo goes only in front of install and systemctl, outside this script.
#
# install, in this order, each step skipped once its sentinel exists (the .setup_ok files, as chat_cluster_setup_h100.sh has them; a
# sentinel is written only after the install and the import check have both passed, because a failed install leaves its directory
# behind). A sentinel holds its date and the sha256 of the pin files its step used; a run that finds other pin files, or an old
# sentinel with no hashes, stops with an ERROR and leaves the directory alone: move it aside and re-run.
#   1 uv 0.13.6 into $VM_TOOLS, from the release archive, refused unless its sha256 matches. ~/.local/bin is never touched.
#   2 CPython 3.11.15 through uv, the cluster's version, kept under $VM_TOOLS (the system python has no ensurepip).
#   3 .venv           CPU torch 2.11.0 and torchvision from the PyTorch CPU index, then requirements-serve.txt. The check fails
#                     unless torch is 2.11.0 and torchvision a CPU build.
#   4 .venv_chexbert  torch 2.13.0 CPU, then requirements-chexbert.txt. Check: torch 2.13.0+cpu, f1chexbert, transformers < 5, sklearn < 1.8.
#   5 .chat_deps, .chat_deps_chexbert  the web overlays, installed with --target from the pin files, as on the cluster.
#   6 the CheXbert weights (public, StanfordAIMI/RRG_scorers, sha256-pinned) into f1chexbert's cache directory, because its own downloader
#     fails with the pinned huggingface_hub and the labeller then never loads. A file already there is checked and never replaced.
#     Steps 6 and 7 and `check` run with XDG_CACHE_HOME, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE and TRANSFORMERS_CACHE unset, so the paths
#     are the ones the service sees (its unit sets none of them).
#   7 the Hugging Face prefetch into $SCRATCH_ROOT/.hf: the gpt2 tokenizer and BiomedCLIP. The wrapper serves with HF_HUB_OFFLINE=1.
#   8 CHAT_HOME (mode 700), logs/ and logs/chat_server_vm.log (mode 600, made here so systemd's append: does not make it root's).
# The pin files are scripts/vm/requirements-*.txt. It ends with `=== END vm setup ===`, or, with N [setup] ERROR lines, exit 1.
#
# Rules. Additive: it creates its own directories and files and never deletes or overwrites anything else (the rm calls remove its own
# temporary download directory and temporary file, also when a signal ends the run). No sudo: the unit is printed for the operator to install. R7 (MIMIC-CXR): the output is [setup] and
# === lines with names, versions and counts, never a path, a file list, report text or an id; the image check prints counts only.
# No deployment: the unit pins MODE=private, BIND=127.0.0.1 and an empty CORS_ORIGINS, and holds no token.
# Runs on bash 3.2 as well (the tests use it): no associative arrays, no mapfile, no ${var,,}.
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO="$(pwd)"

CMD="${1:-install}"
CHAT_HOME="${CHAT_HOME:-$HOME/chat_sessions}"
SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/vm_scratch}"
VM_TOOLS="${VM_TOOLS:-$HOME/.local/share/cxr-chat}"
GALLERY="${GALLERY:-${CHAT_HOME}/gallery/g13d_m3_v1}"
MODELS="${MODELS:-m3eos,m3}"
DRIFT_NOTE="${DRIFT_NOTE:-CPU decode on this server; not probed against the published GPU decode.}"

UV_VERSION="0.13.6"
UV_SHA256="f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645"
UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz"
UV_DIR="${VM_TOOLS}/uv-${UV_VERSION}"
PY_VERSION="3.11.15"
CHEXBERT_SHA256="6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1"   # 1314414442 bytes
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

FAILED=0
problem() { echo "[setup] ERROR $*"; FAILED=$((FAILED + 1)); }
# a failed install or check ends the run: every later step needs it
die() { problem "$@"; echo "=== END vm setup: ${FAILED} ERROR line(s) above ==="; exit 1; }
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ > "$1"; }

# One "<sha256>  <file name>" line per pin file; $@ the pin files.
pin_lines() {
  local f
  for f in "$@"; do echo "$(sha256sum "${f}" | cut -d' ' -f1)  $(basename "${f}")"; done
}
# A sentinel: the date line, then the pin lines. $1 sentinel, then the pin files the step used.
stamp_pins() {
  local sentinel="$1"
  shift
  { date -u +%Y-%m-%dT%H:%M:%SZ; pin_lines "$@"; } > "${sentinel}"
}
# A finished step is skipped only while its pin files are the ones it was installed from. $1 step, $2 directory, $3 sentinel, then the pin files.
check_pins() {
  local step="$1" dir="$2" sentinel="$3"
  shift 3
  [ "$(tail -n +2 "${sentinel}")" = "$(pin_lines "$@")" ] || die "${step} pins changed since it was installed: move ${dir} aside and re-run"
}

# ---------------------------------------------------------------- install
step_uv() {
  if [ -x "${UV_DIR}/uv" ]; then
    echo "[setup] uv ${UV_VERSION}: present"
    return 0
  fi
  local dl actual
  dl="$(mktemp -d)"
  echo "[setup] uv ${UV_VERSION}: downloading"
  curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location --output "${dl}/uv.tar.gz" "${UV_URL}" \
    || { rm -rf "${dl}"; die "uv download failed"; }
  actual="$(sha256sum "${dl}/uv.tar.gz" | cut -d' ' -f1)"
  if [ "${actual}" != "${UV_SHA256}" ]; then
    rm -rf "${dl}"
    die "uv archive sha256 does not match the pinned value: refusing it"
  fi
  tar -xzf "${dl}/uv.tar.gz" -C "${dl}" --strip-components=1 uv-x86_64-unknown-linux-gnu/uv \
    || { rm -rf "${dl}"; die "uv archive could not be extracted"; }
  mkdir -p "${UV_DIR}"
  mv "${dl}/uv" "${UV_DIR}/uv"
  chmod 755 "${UV_DIR}/uv"
  rm -rf "${dl}"
  echo "[setup] uv ${UV_VERSION}: installed, sha256 verified"
}

# a venv and its CPU torch, then its pins; $1 venv directory, $2 torch pin file, $3 pin file, $4 name for the log
step_venv() {
  local venv="$1" torch_pins="$2" pins="$3" name="$4"
  if [ -f "${venv}/.setup_ok" ]; then
    check_pins "${name} venv" "${venv}" "${venv}/.setup_ok" "${torch_pins}" "${pins}"
    echo "[setup] ${name} venv: done, skipped"
    return 0
  fi
  [ -d "${venv}" ] || "${UV}" venv --quiet --python "${PY_VERSION}" "${venv}" || die "venv creation failed: ${name}"
  "${UV}" pip install --quiet --python "${venv}/bin/python" --no-deps --index-url "${TORCH_INDEX}" -r "${torch_pins}" \
    || die "torch install failed: ${name}"
  "${UV}" pip install --quiet --python "${venv}/bin/python" -r "${pins}" || die "install failed: ${name}"
}

step_main_venv() {
  step_venv .venv scripts/vm/requirements-serve-torch.txt scripts/vm/requirements-serve.txt main
  [ -f .venv/.setup_ok ] && return 0
  .venv/bin/python -c "
import sys, torch, torchvision, transformers, open_clip
v = torch.__version__
assert v.split('+')[0] == '2.11.0' and v.endswith('+cpu') and torch.version.cuda is None, v
assert torchvision.__version__.endswith('+cpu'), torchvision.__version__
print('[setup] main venv: python', sys.version.split()[0], 'torch', v, 'torchvision', torchvision.__version__, 'transformers', transformers.__version__, 'open_clip', open_clip.__version__)" \
    || die "venv check failed: main (torch must be 2.11.0, torchvision a CPU build)"
  stamp_pins .venv/.setup_ok scripts/vm/requirements-serve-torch.txt scripts/vm/requirements-serve.txt
}

step_chexbert_venv() {
  step_venv .venv_chexbert scripts/vm/requirements-chexbert-torch.txt scripts/vm/requirements-chexbert.txt chexbert
  [ -f .venv_chexbert/.setup_ok ] && return 0
  .venv_chexbert/bin/python -c "
import sys, torch, f1chexbert, sklearn, transformers
assert torch.__version__ == '2.13.0+cpu', torch.__version__
assert int(transformers.__version__.split('.')[0]) < 5
assert tuple(int(x) for x in sklearn.__version__.split('.')[:2]) < (1, 8)
print('[setup] chexbert venv: python', sys.version.split()[0], 'torch', torch.__version__, 'transformers', transformers.__version__, 'sklearn', sklearn.__version__)" \
    || die "venv check failed: chexbert"
  stamp_pins .venv_chexbert/.setup_ok scripts/vm/requirements-chexbert-torch.txt scripts/vm/requirements-chexbert.txt
}

# The web overlays, as chat_cluster_setup_h100.sh does them: --target writes only into the overlay directory.
step_overlay_main() {
  if [ -f .chat_deps/.setup_ok ]; then
    check_pins "overlay main" .chat_deps .chat_deps/.setup_ok scripts/vm/requirements-overlay.txt
    echo "[setup] overlay main: done, skipped"; return 0; fi
  "${UV}" pip install --quiet --python .venv/bin/python --target .chat_deps -r scripts/vm/requirements-overlay.txt \
    || die "overlay install failed: main"
  PYTHONPATH=.chat_deps .venv/bin/python -c "
import sys, torch, fastapi, uvicorn, httpx
try:
    import python_multipart
except ImportError:
    import multipart
print('[setup] overlay main: python', sys.version.split()[0], 'torch', torch.__version__, 'fastapi', fastapi.__version__, 'httpx', httpx.__version__)" \
    || die "overlay check failed: main"
  stamp_pins .chat_deps/.setup_ok scripts/vm/requirements-overlay.txt
}

step_overlay_chexbert() {
  if [ -f .chat_deps_chexbert/.setup_ok ]; then
    check_pins "overlay chexbert" .chat_deps_chexbert .chat_deps_chexbert/.setup_ok scripts/vm/requirements-overlay-chexbert.txt
    echo "[setup] overlay chexbert: done, skipped"; return 0; fi
  "${UV}" pip install --quiet --python .venv_chexbert/bin/python --target .chat_deps_chexbert -r scripts/vm/requirements-overlay-chexbert.txt \
    || die "overlay install failed: chexbert"
  PYTHONPATH=.chat_deps_chexbert .venv_chexbert/bin/python -c "import sys, fastapi, uvicorn, sklearn, transformers; assert int(transformers.__version__.split('.')[0]) < 5; assert tuple(int(x) for x in sklearn.__version__.split('.')[:2]) < (1, 8); print('[setup] overlay chexbert: python', sys.version.split()[0], 'transformers', transformers.__version__, 'sklearn', sklearn.__version__, 'fastapi', fastapi.__version__)" \
    || die "overlay check failed: chexbert"
  stamp_pins .chat_deps_chexbert/.setup_ok scripts/vm/requirements-overlay-chexbert.txt
}

# Where f1chexbert looks for chexbert.pth. Result in CX_TARGET; fails when the CheXbert python cannot say.
chexbert_target() {
  CX_TARGET="$(env -u XDG_CACHE_HOME -u HF_HUB_CACHE -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE .venv_chexbert/bin/python -c 'from appdirs import user_cache_dir; import os; print(os.path.join(user_cache_dir("chexbert"), "chexbert.pth"))' 2>/dev/null)" \
    && [ -n "${CX_TARGET}" ]
}
file_sha() { sha256sum "$1" | cut -d' ' -f1; }

# The CheXbert weights, fetched here because f1chexbert's own download fails (huggingface_hub rejects its force_filename). A file in
# place is verified and left alone; the download goes through a temporary HF cache and reaches the target by temp file and rename.
# Both temporaries sit in the target's directory (one filesystem, so the rename is atomic); a failure or a signal removes them.
step_chexbert_weights() {
  local dl="" got tmp="" dir
  chexbert_target || die "chexbert weights: cache directory unknown (CheXbert venv python failed)"
  if [ -f "${CX_TARGET}" ]; then
    [ "$(file_sha "${CX_TARGET}")" = "${CHEXBERT_SHA256}" ] \
      || die "chexbert weights: $(basename "${CX_TARGET}") has an unexpected sha256: move it aside and re-run"
    echo "[setup] chexbert weights: present, sha256 verified"
    return 0
  fi
  mkdir -p logs
  dir="$(dirname "${CX_TARGET}")"
  mkdir -p "${dir}" || die "chexbert weights: cache directory could not be created"
  dl="$(mktemp -d "${dir}/.dl.XXXXXX")" || die "chexbert weights: cache directory not writable"
  trap 'rm -rf "${dl:-}"; rm -f "${tmp:-}"' EXIT
  trap 'exit 1' INT TERM
  got="$(HF_HOME="${dl}" HF_HUB_OFFLINE=0 env -u XDG_CACHE_HOME -u HF_HUB_CACHE -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE .venv_chexbert/bin/python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id='StanfordAIMI/RRG_scorers', filename='chexbert.pth'))" 2>> logs/vm_setup_chexbert.log)" \
    || die "chexbert weights download failed: bash scripts/vm_remote.sh summary logs/vm_setup_chexbert.log"
  if [ -z "${got}" ] || [ "$(file_sha "${got}")" != "${CHEXBERT_SHA256}" ]; then
    die "chexbert weights: the download has an unexpected sha256: nothing written"
  fi
  tmp="$(mktemp "${dir}/.chexbert.pth.XXXXXX")" || die "chexbert weights: cache directory not writable"
  cp "${got}" "${tmp}" || die "chexbert weights: copy failed"
  mv "${tmp}" "${CX_TARGET}" || die "chexbert weights: could not move the file into place"
  tmp=""
  rm -rf "${dl}"
  trap - EXIT INT TERM
  echo "[setup] chexbert weights: downloaded, sha256 verified"
}

# The two public models the server loads at start. The prefetch runs online; the wrapper serves with HF_HUB_OFFLINE=1. The
# download's progress goes to logs/, not to this output.
step_prefetch() {
  local hf="${SCRATCH_ROOT}/.hf"
  if [ -f "${hf}/.prefetch_ok" ]; then echo "[setup] hf prefetch: done, skipped"; return 0; fi
  mkdir -p "${hf}" logs
  HF_HOME="${hf}" HF_HUB_OFFLINE=0 env -u XDG_CACHE_HOME -u HF_HUB_CACHE -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE .venv/bin/python -c "
import open_clip, transformers
transformers.AutoTokenizer.from_pretrained('gpt2')
open_clip.create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
print('[setup] hf prefetch: gpt2 tokenizer and BiomedCLIP cached')" 2>> logs/vm_setup_prefetch.log \
    || die "hf prefetch failed: bash scripts/vm_remote.sh summary logs/vm_setup_prefetch.log"
  stamp "${hf}/.prefetch_ok"
}

do_install() {
  echo "=== vm setup: install ==="
  step_uv
  UV="${UV_DIR}/uv"
  export UV_PYTHON_INSTALL_DIR="${VM_TOOLS}/python"
  if [ ! -f .venv/.setup_ok ] || [ ! -f .venv_chexbert/.setup_ok ]; then
    "${UV}" python install --no-bin "${PY_VERSION}" || die "python ${PY_VERSION} install failed"
  fi
  step_main_venv
  step_chexbert_venv
  step_overlay_main
  step_overlay_chexbert
  step_chexbert_weights
  step_prefetch
  mkdir -p "${CHAT_HOME}" && chmod 700 "${CHAT_HOME}"
  mkdir -p "${CHAT_HOME}/logs" logs
  : >> logs/chat_server_vm.log
  chmod 600 logs/chat_server_vm.log
  echo "[setup] CHAT_HOME and logs/ ready"
  echo "=== END vm setup ==="
}

# ---------------------------------------------------------------- check
# $1 file; prints the shape the cluster setup uses: the parent directory's name and the file's
need() {
  local f="$1"
  if [ -f "${f}" ]; then echo "[setup] ok $(basename "$(dirname "${f}")")/$(basename "${f}")"
  else problem "missing $(basename "$(dirname "${f}")")/$(basename "${f}")"; fi
}

do_check() {
  echo "=== vm setup: check ==="
  local f eos="${CHAT_HOME}/models/report_gen_m3_eos_s42"
  for f in outputs/h100_report_gen_m3_tower13d_s42/checkpoints/last.ckpt \
           outputs/h100_report_gen_m3_tower13d_s42/run_metadata.json \
           results/report_gen_m3_test_split_s42/hyps.txt results/report_gen_m3_test_split_s42/refs.txt \
           results/report_gen_m3_test_split_s42/chexbert_labels.json results/retrieval_floor_test_split/hyps.txt \
           "${eos}/checkpoints/last.ckpt" "${eos}/DONE" "${eos}/run_metadata.json" \
           "${GALLERY}/manifest.json" \
           .venv/.setup_ok .venv_chexbert/.setup_ok .chat_deps/.setup_ok .chat_deps_chexbert/.setup_ok \
           "${SCRATCH_ROOT}/.hf/.prefetch_ok"; do
    need "${f}"
  done
  # present and the pinned sha256; the path and the hash stay off the output
  if [ -x .venv_chexbert/bin/python ] && chexbert_target && [ -f "${CX_TARGET}" ]; then
    if [ "$(file_sha "${CX_TARGET}")" = "${CHEXBERT_SHA256}" ]; then echo "[setup] ok chexbert/chexbert.pth"
    else problem "unexpected sha256 chexbert/chexbert.pth"; fi
  else
    problem "missing chexbert/chexbert.pth"
  fi
  # The parquets hold absolute paths of MIMIC images: only counts leave this python (R7).
  if [ -x .venv/bin/python ]; then
    .venv/bin/python -c "
import os, sys
import pandas as pd
bad = 0
for name, label in (('img_meta', 'train'), ('test_meta', 'test')):
    try:
        paths = pd.read_parquet(os.path.join(sys.argv[1], name + '.parquet'), columns=['image'])['image'].tolist()
    except Exception:
        print('[setup] images unreadable (%s)' % label)
        bad += 1
        continue
    found = sum(1 for p in paths if os.path.isfile(str(p)))
    print('[setup] images present %d/%d (%s)' % (found, len(paths), label))
    bad += len(paths) - found
sys.exit(1 if bad else 0)" "${GALLERY}" || problem "images missing or unreadable (counts above)"
  else
    problem "missing .venv/bin/python: no image check"
  fi
  if [ "${FAILED}" -ne 0 ]; then
    echo "=== END vm setup: ${FAILED} ERROR line(s) above ==="
    exit 1
  fi
  echo "=== END vm setup ==="
}

# ---------------------------------------------------------------- unit
# `unit` writes its result to stdout and every refusal to stderr, so a redirect into a unit file never holds an error.
unit_die() { echo "[setup] ERROR $*" >&2; exit 1; }
# a value that systemd would read as more than text, or that the template's quotes cannot hold, is refused (the value is not printed)
unit_value() {
  local name="$1" value="$2"
  [[ "${value}" != *$'\n'* ]] || { unit_die "${name} holds a newline"; }
  [[ "${value}" != *'"'* ]] || { unit_die "${name} holds a double quote"; }
  [[ "${value}" != *'\'* ]] || { unit_die "${name} holds a backslash"; }
}

# systemd reads % as a specifier: double it. Result in DOUBLED. (Plain loops, not ${v//%/%%}: bash 5.2 gives & a meaning in a replacement.)
double_percent() {
  local s="$1" out="" pct='%'
  while [[ "${s}" == *"${pct}"* ]]; do
    out="${out}${s%%"${pct}"*}%%"
    s="${s#*"${pct}"}"
  done
  DOUBLED="${out}${s}"
}

# The default memory limits: $1 percent of MemTotal, as whole MiB with the suffix M. Result in MEM_DEFAULT; empty when MemTotal is unreadable.
mem_default() {
  local kb
  kb="$(awk '/^MemTotal:/ { print $2; exit }' "${MEMINFO:-/proc/meminfo}" 2>/dev/null || true)"
  MEM_DEFAULT=""
  if [[ "${kb}" =~ ^[0-9]+$ ]] && [ $((kb * $1 / 102400)) -gt 0 ]; then MEM_DEFAULT="$((kb * $1 / 102400))M"; fi
}

do_unit() {
  local user group tpl="scripts/vm/cxr-chat.service.in" shape='^[a-z0-9,]{1,32}$' size='^[0-9]+[KMG]$'
  local threads memory_high memory_max
  user="$(id -un)"
  group="$(id -gn)"
  [ -f "${tpl}" ] || { unit_die "unit template missing"; }
  unit_value USER "${user}"
  unit_value GROUP "${group}"
  unit_value REPO "${REPO}"
  unit_value CHAT_HOME "${CHAT_HOME}"
  unit_value SCRATCH_ROOT "${SCRATCH_ROOT}"
  unit_value MODELS "${MODELS}"
  unit_value DRIFT_NOTE "${DRIFT_NOTE}"
  [[ "${MODELS}" =~ ${shape} ]] || { unit_die "MODELS must be 1 to 32 characters: a-z, 0-9 and commas"; }
  case "${CHAT_HOME}" in /*) ;; *) unit_die "CHAT_HOME must be an absolute path" ;; esac
  case "${SCRATCH_ROOT}" in /*) ;; *) unit_die "SCRATCH_ROOT must be an absolute path" ;; esac
  [ "${CHAT_HOME}" = "${HOME}/chat_sessions" ] \
    || { unit_die "CHAT_HOME must be \$HOME/chat_sessions: the tunnel and vm_remote.sh healthz read ~/chat_sessions/endpoint"; }
  threads="${THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)}"
  if ! [[ "${threads}" =~ ^[0-9]{1,3}$ ]] || [ $((10#${threads})) -lt 1 ] || [ $((10#${threads})) -gt 256 ]; then
    unit_die "THREADS must be a whole number from 1 to 256"
  fi
  threads=$((10#${threads}))
  memory_high="${MEMORY_HIGH:-}"
  memory_max="${MEMORY_MAX:-}"
  if [ -z "${memory_high}" ]; then mem_default 75; memory_high="${MEM_DEFAULT}"; fi
  if [ -z "${memory_max}" ]; then mem_default 90; memory_max="${MEM_DEFAULT}"; fi
  if [ -z "${memory_high}" ] || [ -z "${memory_max}" ]; then
    unit_die "no MemTotal to take the memory limits from: set MEMORY_HIGH and MEMORY_MAX"
  fi
  [[ "${memory_high}" =~ ${size} ]] || { unit_die "MEMORY_HIGH must be a number and K, M or G, such as 6000M"; }
  [[ "${memory_max}" =~ ${size} ]] || { unit_die "MEMORY_MAX must be a number and K, M or G, such as 7000M"; }
  # One pass over the template: a value that contains a marker is never filled in again.
  local rest out="" token value
  rest="$(cat "${tpl}"; printf x)"
  rest="${rest%x}"
  while [[ "${rest}" == *@* ]]; do
    out="${out}${rest%%@*}"
    rest="${rest#*@}"
    token="${rest%%@*}"
    value=""
    case "${token}" in
      USER) value="${user}" ;; GROUP) value="${group}" ;; REPO) value="${REPO}" ;; CHAT_HOME) value="${CHAT_HOME}" ;;
      SCRATCH_ROOT) value="${SCRATCH_ROOT}" ;; MODELS) value="${MODELS}" ;; DRIFT_NOTE) value="${DRIFT_NOTE}" ;;
      THREADS) value="${threads}" ;; MEMORY_HIGH) value="${memory_high}" ;; MEMORY_MAX) value="${memory_max}" ;;
      *) token="" ;;
    esac
    if [ -n "${token}" ] && [[ "${rest}" == *@* ]]; then
      double_percent "${value}"
      out="${out}${DOUBLED}"
      rest="${rest#*@}"
    else
      out="${out}@"
    fi
  done
  printf '%s%s' "${out}" "${rest}"
}

case "${CMD}" in install | check | unit)
  if [ "$(id -u)" -eq 0 ]; then
    msg="[setup] ERROR run as the service user, not root (sudo goes only in front of install and systemctl, outside this script)"
    if [ "${CMD}" = unit ]; then echo "${msg}" >&2; else echo "${msg}"; fi
    exit 1
  fi ;;
esac

case "${CMD}" in
  install) do_install ;;
  check) do_check ;;
  unit) do_unit ;;
  *) echo "usage: bash scripts/vm_setup.sh [install|check|unit]" >&2; exit 2 ;;
esac
