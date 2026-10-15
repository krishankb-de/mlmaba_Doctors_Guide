#!/bin/bash
# ============================================================================
# CHAT_UI_PLAN.md P9-B — publish the GitHub Pages copy of the chat page on the gh-pages branch of origin.
#   bash scripts/pages_publish.sh
# Run it from a checkout whose app/static/ has no change that is not committed: what is published is that commit's page.
#
#   1. scripts/pages_build.py builds the copy into a fresh temporary directory, outside every checkout.
#   2. In a temporary worktree (it shares this repository's objects, and nothing of the caller's working tree, index or stash), one commit on
#      the gh-pages branch holds exactly the build: the branch's first commit, with no parent, while origin has no gh-pages; after that a
#      commit on top of origin's gh-pages, so that the push is a fast-forward. A build that gh-pages holds already is not committed again.
#   3. git push origin gh-pages, never forced. When origin refuses it (someone published meanwhile), the script fails, and the local gh-pages
#      goes back to where it was: nothing is left behind.
#   4. The temporary worktree and directory go, however the script ends.
# No other branch is read, written or checked out, and the caller's working tree, index and stash are never touched. It prints the commit and
# the address GitHub Pages serves the branch at. It does not switch Pages on: that is done once, in the repository's Settings (Pages, deploy
# from the gh-pages branch, root), and only by the user or on their word.
#
# PYTHON (default python3) runs the build, which needs nothing but the standard library.
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BRANCH="gh-pages"

ERROR_SHOWN=0   # lets the EXIT trap tell a refusal that has said why from a command that failed without a word of ours
say() { echo "[pages] $*"; }
fail() { ERROR_SHOWN=1; echo "ERROR $*" >&2; exit 1; }

# The address GitHub Pages serves gh-pages at, from origin's own URL (the configured one: no insteadOf applied); empty for an origin that is
# not on github.com. A repository named <owner>.github.io is the owner's site, at the root.
pages_url() {
  local url path owner repo
  url="$(git -C "${REPO_ROOT}" config --get remote.origin.url || true)"
  case "${url}" in
    https://github.com/*) path="${url#https://github.com/}" ;;
    git@github.com:*) path="${url#git@github.com:}" ;;
    ssh://git@github.com/*) path="${url#ssh://git@github.com/}" ;;
    *) return 0 ;;
  esac
  path="${path%/}"
  path="${path%.git}"
  owner="$(printf '%s' "${path%%/*}" | tr '[:upper:]' '[:lower:]')"
  repo="${path#*/}"
  case "${owner}/${repo}" in */*/*|/*|*/) return 0 ;; esac
  [ -n "${owner}" ] && [ "${repo}" != "${path}" ] || return 0
  if [ "$(printf '%s' "${repo}" | tr '[:upper:]' '[:lower:]')" = "${owner}.github.io" ]; then
    echo "https://${owner}.github.io/"
  else
    echo "https://${owner}.github.io/${repo}/"
  fi
}

git -C "${REPO_ROOT}" rev-parse --verify --quiet HEAD > /dev/null || fail "this checkout has no commit to publish the page of"
git -C "${REPO_ROOT}" config --get remote.origin.url > /dev/null || fail "this repository has no remote named origin"
# --no-optional-locks: a plain status refreshes the index and writes it back, and the caller's index is never touched. --ignored: the build copies
# what is on disk, so a page file that git ignores would be published although no commit holds it. Dotfiles (.DS_Store) are left out, as the
# build leaves them behind.
[ -z "$(git -C "${REPO_ROOT}" --no-optional-locks status --porcelain --untracked-files=all --ignored -- app/static ':(exclude,glob)app/static/**/.*')" ] \
  || fail "app/static has files that are not committed, ignored ones included: commit or remove them first, so that what is published is a commit's page"
# The list is read whole before it is searched: grep -q, leaving a pipe at its first match, can kill git with SIGPIPE, and under pipefail the
# pipeline would then fail and the check be skipped.
WORKTREES="$(git -C "${REPO_ROOT}" worktree list --porcelain)"
if grep -qx "branch refs/heads/${BRANCH}" <<< "${WORKTREES}"; then
  fail "gh-pages is checked out in a worktree of this repository: switch that worktree to another branch first"
fi
SOURCE="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
OLD="$(git -C "${REPO_ROOT}" rev-parse --verify --quiet "refs/heads/${BRANCH}" || true)"   # the local gh-pages, put back unless the push goes through

WORK="$(mktemp -d "${TMPDIR:-/tmp}/pages_publish.XXXXXX")"
WT="${WORK}/worktree"
SITE="${WORK}/site"
ADDED=0    # the temporary worktree exists
MOVED=0    # the local gh-pages was moved, or made, by this run
PUSHED=0

cleanup() {
  local rc=$?
  if [ "${ADDED}" -eq 1 ]; then
    git -C "${REPO_ROOT}" worktree remove --force "${WT}" > /dev/null 2>&1 || true
  fi
  if [ "${MOVED}" -eq 1 ] && [ "${PUSHED}" -eq 0 ]; then
    if [ -n "${OLD}" ]; then
      git -C "${REPO_ROOT}" update-ref "refs/heads/${BRANCH}" "${OLD}" || true
    else
      git -C "${REPO_ROOT}" update-ref -d "refs/heads/${BRANCH}" || true
    fi
  fi
  rm -rf "${WORK}"
  if [ "${rc}" -ne 0 ] && [ "${ERROR_SHOWN}" -eq 0 ]; then
    echo "ERROR the publish stopped (exit ${rc}): nothing was published, and nothing it made is left" >&2
  fi
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

"${PYTHON}" "${REPO_ROOT}/scripts/pages_build.py" "${SITE}"

git -C "${REPO_ROOT}" worktree add --quiet --detach --no-checkout "${WT}" "${SOURCE}"
ADDED=1
REMOTE_RC=0
git -C "${WT}" ls-remote --exit-code origin "refs/heads/${BRANCH}" > /dev/null || REMOTE_RC=$?
case "${REMOTE_RC}" in
  0)   # origin has gh-pages: its tip, fetched into the worktree's own FETCH_HEAD (no ref of the caller moves), is what the commit goes on
    git -C "${WT}" fetch --quiet --no-tags --refmap= origin "refs/heads/${BRANCH}"
    BASE="$(git -C "${WT}" rev-parse --verify 'FETCH_HEAD^{commit}')"
    if [ -n "${OLD}" ] && ! git -C "${WT}" merge-base --is-ancestor "${OLD}" "${BASE}"; then
      fail "the local gh-pages has commits that origin's gh-pages has not: push them or let them go first; nothing was overwritten"
    fi
    MOVED=1
    git -C "${WT}" update-ref "refs/heads/${BRANCH}" "${BASE}"
    ;;
  2)   # origin has no gh-pages yet: the first publish
    [ -z "${OLD}" ] || fail "origin has no gh-pages, but this repository has one: push it or delete it first; nothing was overwritten"
    BASE=""
    ;;
  *) fail "origin's branches cannot be read (git ls-remote exit ${REMOTE_RC})" ;;
esac

# The worktree's HEAD names gh-pages (still to be born on a first publish), and its index and tree hold the build and nothing else.
git -C "${WT}" symbolic-ref HEAD "refs/heads/${BRANCH}"
git -C "${WT}" read-tree --empty
cp -R "${SITE}/." "${WT}/"
git -C "${WT}" add --all --force .
TREE="$(git -C "${WT}" write-tree)"
if [ -n "${BASE}" ] && [ "${TREE}" = "$(git -C "${WT}" rev-parse "${BASE}^{tree}")" ]; then
  say "gh-pages holds this build already: nothing to publish"
  exit 0
fi
MOVED=1
git -C "${WT}" commit --quiet -m "Pages build of ${SOURCE}"
NEW="$(git -C "${WT}" rev-parse HEAD)"
say "committed ${NEW} on gh-pages, the Pages copy of ${SOURCE}"

git -C "${WT}" push origin gh-pages || fail "origin refused gh-pages (its own words are above): the local gh-pages is put back where it was"
PUSHED=1
say "pushed gh-pages to origin"
URL="$(pages_url)"
if [ -n "${URL}" ]; then
  say "Pages URL: ${URL}"
else
  say "Pages URL: unknown, origin is not a github.com repository"
fi
