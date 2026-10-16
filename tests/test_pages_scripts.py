"""CHAT_UI_PLAN.md P9-B: the GitHub Pages copy of the chat page, built and published for real.

scripts/pages_build.py runs on this checkout's own app/static/ into a temporary directory, and on a synthetic tree (a copy of the script beside an
app/static/ of the test's own) for what the real page must never hold: a broken import, an absolute /static/ reference, a symlink, a file that
no page is made of. scripts/pages_publish.sh runs under bash in a temporary repository made of this checkout's page and the two scripts, against a
local bare repository that stands in for origin: nothing here reaches GitHub and nothing runs `gh` (publishing is the controller's step, after
the user's OK). Synthetic data only (R7).
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC = REPO_ROOT / "app" / "static"
BUILD = REPO_ROOT / "scripts" / "pages_build.py"
PUBLISH = REPO_ROOT / "scripts" / "pages_publish.sh"
BASH = "/bin/bash" if os.path.exists("/bin/bash") else (shutil.which("bash") or "bash")   # /bin/bash is 3.2 on the Mac: the oldest it must run in
PAGES_TAG = '<meta name="cxr-pages" content="1">'
ORIGIN_URL = "https://github.com/Owner-Name/chat_pages_test.git"   # what origin is called; insteadOf sends every git call to the bare repository
PAGES_URL = "https://owner-name.github.io/chat_pages_test/"


def run_build(out: Path, script: Path = BUILD) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), str(out)], capture_output=True, text=True, timeout=60)


def files_of(root: Path) -> Dict[str, bytes]:
    """{path relative to root: bytes} of every file under root, dotfiles included."""
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def page_sources() -> List[str]:
    """The page's files, relative to app/static: its code at the top, its pictures and fonts in their folders (img/, fonts/)."""
    return sorted(p.relative_to(STATIC).as_posix() for p in STATIC.rglob("*") if p.is_file() and not p.name.startswith("."))


# ---- the build, on this checkout's page --------------------------------------------------------------------------------------

def test_the_build_is_app_static_beside_an_index_that_is_relative_and_carries_the_pages_tag(tmp_path):
    before = files_of(STATIC)
    out = tmp_path / "site" / "chat_pages_test"   # a sub-path, as GitHub Pages serves a project's site
    done = run_build(out)
    assert done.returncode == 0, done.stdout + done.stderr
    built = files_of(out)
    names = page_sources()
    assert "index.html" in names and "app.js" in names
    assert set(built) == {"index.html", ".nojekyll"} | {"static/" + n for n in names if n != "index.html"}, sorted(built)
    for name in names:
        if name != "index.html":
            assert built["static/" + name] == (STATIC / name).read_bytes(), name   # byte for byte
    assert built[".nojekyll"] == b""
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    expected = page.replace('="/static/', '="static/').replace('<meta charset="utf-8">\n', '<meta charset="utf-8">\n  ' + PAGES_TAG + "\n", 1)
    assert built["index.html"].decode("utf-8") == expected
    assert 'href="static/styles.css"' in expected and 'src="static/app.js"' in expected and "/static/" not in expected
    assert files_of(STATIC) == before, "the page the API serves is not changed at all (R8)"
    assert PAGES_TAG not in page, "and it never carries the Pages tag itself"


def test_the_build_says_what_it_built_on_one_line(tmp_path):
    done = run_build(tmp_path / "site")
    assert done.returncode == 0 and done.stderr == ""
    assert done.stdout.splitlines() == ["[pages] built {} files: index.html, .nojekyll and static/ ({} files)".format(
        len(page_sources()) + 1, len(page_sources()) - 1)]


@pytest.mark.parametrize("where", ["repository", "app_static", "the_repository_itself", "another_checkout"])
def test_the_build_refuses_a_directory_inside_a_checkout_and_writes_nothing(tmp_path, where):
    """A build inside a checkout could be committed: this repository, any directory under it, and any other checkout of this project (a .git
    beside hybrid_xmamba/) are refused before anything is made."""
    other = tmp_path / "other_checkout"
    (other / ".git").mkdir(parents=True)
    (other / "hybrid_xmamba").mkdir()
    out = {"repository": REPO_ROOT / "pages_site_refused", "app_static": STATIC / "pages_site_refused", "the_repository_itself": REPO_ROOT,
           "another_checkout": other / "deep" / "site"}[where]
    done = run_build(out)
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stderr.splitlines() == ["ERROR pages_build: OUT_DIR is inside a checkout of this project, where the build could be committed: "
                                        "build outside it"] and done.stdout == ""   # no path (P9-B fix 1, R7)
    if where != "the_repository_itself":
        assert not out.exists()


def test_the_build_refuses_a_directory_that_is_not_empty_or_a_file_and_leaves_them_as_they_were(tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    (full / "keep.txt").write_text("mine\n")
    done = run_build(full)
    assert done.returncode == 2
    assert done.stderr.splitlines() == ["ERROR pages_build: OUT_DIR must be an empty directory, or one that is not there yet"]
    assert files_of(full) == {"keep.txt": b"mine\n"}
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    (hidden / ".git").mkdir()
    assert run_build(hidden).returncode == 2, "a dotfile is content too"
    a_file = tmp_path / "a_file"
    a_file.write_text("x\n")
    assert run_build(a_file).returncode == 2 and a_file.read_text() == "x\n"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_build(empty).returncode == 0, "an empty directory that is there already is fine"


# ---- the build, on a synthetic tree ------------------------------------------------------------------------------------------

INDEX = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Synthetic page</title>
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="/static/styles.css">
  <script type="module" src="/static/app.js"></script>
</head>
<body><main id="conversation"></main></body>
</html>
"""
APP = """// app/static/app.js: a comment that names app/static/ and a word from './nowhere.js' is no import
import * as api from './api.js';
import {
  draw,
} from './render.js';
export const x = api.a + draw();
"""
PAGE = {"index.html": INDEX, "app.js": APP, "api.js": "export const a = 1;\n", "render.js": "export const draw = () => 2;\n",
        "styles.css": "/* no url(): only its own /static/ files */\nbody { margin: 0; }\n"}


def synthetic(tmp_path: Path, **files) -> Path:
    """A tree of the test's own: this checkout's scripts/pages_build.py beside an app/static/ that holds PAGE with `files` laid over it (a
    value of None leaves that file out). -> the copy of the script."""
    root = tmp_path / "tree"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(str(BUILD), str(root / "scripts" / "pages_build.py"))
    static = root / "app" / "static"
    static.mkdir(parents=True)
    for name, text in dict(PAGE, **files).items():
        if text is not None:
            (static / name).parent.mkdir(parents=True, exist_ok=True)
            (static / name).write_text(text)
    return root / "scripts" / "pages_build.py"


def test_a_synthetic_page_builds_its_imports_resolve_and_a_dotfile_stays_behind(tmp_path):
    script = synthetic(tmp_path, **{".DS_Store": "finder\n", "icons/mark.css": "i { color: red; }\n"})
    out = tmp_path / "site"
    done = run_build(out, script)
    assert done.returncode == 0, done.stdout + done.stderr
    assert sorted(files_of(out)) == [".nojekyll", "index.html", "static/api.js", "static/app.js", "static/icons/mark.css", "static/render.js",
                                     "static/styles.css"]


NO_FILE = "ERROR pages_build: 1 reference to no file of the copy"
NOT_RELATIVE = "ERROR pages_build: 1 import that is not relative"
ABSOLUTE_LEFT = "ERROR pages_build: 1 absolute /static/ reference left"


@pytest.mark.parametrize("files, errors, names", [
    ({"app.js": "import { a } from './missing.js';\n"}, [NO_FILE], ["missing"]),
    ({"app.js": "import {\n  a,\n  b,\n} from './gone.js';\n"}, [NO_FILE], ["gone"]),
    ({"app.js": "export { a } from './gone.js';\n"}, [NO_FILE], ["gone"]),
    ({"app.js": "import './side.js';\n"}, [NO_FILE], ["side"]),
    ({"app.js": "const m = import('./later.js');\n"}, [NO_FILE], ["later"]),
    ({"app.js": "import { a } from '/static/api.js';\n"}, [ABSOLUTE_LEFT, NOT_RELATIVE], ["api.js"]),
    ({"app.js": "import x from 'lodash';\n"}, [NOT_RELATIVE], ["lodash"]),
    ({"app.js": "import x from 'https://cdn.example/x.js';\n"}, [NOT_RELATIVE], ["cdn.example", "x.js"]),
    ({"app.js": "import { a } from '../outside.js';\n"}, [NO_FILE], ["outside"]),
    ({"app.js": "import { a } from './api.js';\nexport const u = '/static/api.js';\n"}, [ABSOLUTE_LEFT], ["api.js"]),
    ({"styles.css": "@import url('/static/more.css');\n"}, [ABSOLUTE_LEFT, NO_FILE], ["more.css"]),
    ({"styles.css": "body { background: url(gone.png); }\n"}, [NO_FILE], ["gone.png"]),
    ({"app.js": "import { a } from './one.js';\nimport { b } from './two.js';\n"}, ["ERROR pages_build: 2 references to no file of the copy"],
     ["one.js", "two.js"]),
    ({"index.html": INDEX.replace("/static/styles.css", "/static/gone.css")}, [NO_FILE], ["gone.css"]),
    ({"index.html": INDEX.replace('href="data:,"', 'href="https://cdn.example/icon.png"')}, [NO_FILE], ["cdn.example", "icon.png"]),
])
def test_the_build_checks_its_own_copy_and_says_how_many_of_each_kind_never_which(tmp_path, files, errors, names):
    """Every module import, every file index.html and the stylesheets name, resolves to a file inside the copy, and no absolute /static/
    reference is left: one that is not fails the build (exit 1). The ERROR lines give a count and the kind, one line per kind; a file's name, a
    reference or a path never reaches the output (P9-B fix 1, R7)."""
    out = tmp_path / "site"
    done = run_build(out, synthetic(tmp_path, **files))
    assert done.returncode == 1, done.stdout + done.stderr
    assert done.stderr.splitlines() == errors and done.stdout == ""
    for name in names + ["app.js", "styles.css", "index.html", str(tmp_path)]:
        assert name not in done.stderr, name


@pytest.mark.parametrize("files, error", [
    ({"notes.txt": "a note\n"}, "ERROR pages_build: 1 file of another kind refused: a page is made of .html, .js and .css, img/*.png and fonts/*.woff2 and *.txt, and nothing else is published"),
    ({"img/scan.jpg": "not really a picture\n", "fonts/face.ttf": "not a woff2\n"},
     "ERROR pages_build: 2 files of another kind refused: a page is made of .html, .js and .css, img/*.png and fonts/*.woff2 and *.txt, and nothing else is published"),   # the folders take their own kinds only
    ({"scan.png": "not really a picture\n", "deep/notes.txt": "a note\n"},
     "ERROR pages_build: 2 files of another kind refused: a page is made of .html, .js and .css, img/*.png and fonts/*.woff2 and *.txt, and nothing else is published"),
    ({"index.html": INDEX.replace("<head>\n", "<head>\n  " + PAGES_TAG + "\n")},
     "ERROR pages_build: the page's index.html carries the Pages tag itself: the page the API serves must not"),
    ({"index.html": INDEX.replace('="/static/', '="static/')},
     "ERROR pages_build: the page's index.html names no /static/ file: it is not the page this script knows how to copy"),
    ({"index.html": INDEX.replace('<meta charset="utf-8">\n', "")},
     "ERROR pages_build: the page's index.html has no <meta charset=\"utf-8\"> line for the Pages tag to follow"),
])
def test_the_build_refuses_a_page_it_cannot_copy_safely_before_it_writes_anything(tmp_path, files, error):
    """Only the kinds of file a page is made of (.html, .js, .css) are copied (R7: a picture or a note that strayed into app/static is never
    published, and its name is never printed), and the API's index.html must be the page this script knows how to turn into the Pages copy."""
    out = tmp_path / "site"
    done = run_build(out, synthetic(tmp_path, **files))
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stderr.splitlines() == [error]
    for name in ("notes", "scan", str(tmp_path)):
        assert name not in done.stderr, name
    assert not out.exists()


def test_the_build_refuses_a_symlink_in_app_static_whatever_it_points_to(tmp_path):
    script = synthetic(tmp_path)
    secret = tmp_path / "outside.js"
    secret.write_text("export const s = 'from outside app/static';\n")
    (script.parent.parent / "app" / "static" / "linked.js").symlink_to(secret)
    (script.parent.parent / "app" / "static" / "linked_dir").symlink_to(tmp_path)
    out = tmp_path / "site"
    done = run_build(out, script)
    assert done.returncode == 2
    assert done.stderr.splitlines() == ["ERROR pages_build: 2 symlinks refused: the build copies regular files only"]
    for name in ("linked", "outside", str(tmp_path)):
        assert name not in done.stderr, name   # no name and no path (P9-B fix 1, R7)
    assert not out.exists()


# ---- the publish, against a bare repository that stands in for origin ---------------------------------------------------------

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Repos:
    """A bare origin and a caller's repository made of this checkout's page and the two scripts, with a branch `main` and a branch `work` on
    both, and a caller's checkout that is far from clean: a stash entry, a staged file, a changed file and an untracked one, none of them under
    app/static. git runs with a configuration of the test's own (HOME, no system or global file), and every temporary directory the publish
    makes is under `tmp`, so that what is left behind can be seen."""

    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.home.mkdir()
        (self.home / ".gitconfig").write_text("[init]\n\tdefaultBranch = main\n[user]\n\tname = P9-B test\n\temail = p9b@example.invalid\n"
                                              "[commit]\n\tgpgsign = false\n")
        self.tmp = root / "tmpdir"
        self.tmp.mkdir()
        self.env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        self.env.update(HOME=str(self.home), GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=str(self.home / ".gitconfig"), TMPDIR=str(self.tmp),
                        PYTHON=sys.executable, LC_ALL="C")
        self.origin = root / "origin.git"
        self.git(root, "init", "--quiet", "--bare", str(self.origin))
        self.caller = root / "caller"
        (self.caller / "scripts").mkdir(parents=True)
        for script in (BUILD, PUBLISH):
            shutil.copyfile(str(script), str(self.caller / "scripts" / script.name))
        shutil.copytree(str(STATIC), str(self.caller / "app" / "static"), ignore=shutil.ignore_patterns(".*"))
        (self.caller / "README.txt").write_text("the caller's own file\n")
        self.git(self.caller, "init", "--quiet")
        self.git(self.caller, "add", "-A")
        self.git(self.caller, "commit", "--quiet", "-m", "the page")
        self.git(self.caller, "checkout", "--quiet", "-b", "work")
        (self.caller / "work.txt").write_text("on the work branch\n")
        self.git(self.caller, "add", "work.txt")
        self.git(self.caller, "commit", "--quiet", "-m", "work")
        self.git(self.caller, "checkout", "--quiet", "main")
        self.git(self.caller, "remote", "add", "origin", ORIGIN_URL)
        self.git(self.caller, "config", "url.{}.insteadOf".format(self.origin), ORIGIN_URL)
        self.git(self.caller, "push", "--quiet", "origin", "main", "work")
        # far from clean, and none of it under app/static
        (self.caller / "README.txt").write_text("a change that is stashed\n")
        self.git(self.caller, "stash", "push", "--quiet", "-m", "p9b stash")
        (self.caller / "staged.txt").write_text("staged, not committed\n")
        self.git(self.caller, "add", "staged.txt")
        (self.caller / "README.txt").write_text("a change that is not staged\n")
        (self.caller / "untracked.txt").write_text("nobody added this\n")

    def git(self, where: Path, *args: str) -> str:
        done = subprocess.run(["git", "-C", str(where)] + list(args), env=self.env, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, (args, done.stdout, done.stderr)
        return done.stdout

    def commit_page_change(self, name: str = "styles.css", text: str = "/* changed */\n") -> None:
        """A committed change to the page, made without touching what the caller has staged (a commit of that path only)."""
        with open(str(self.caller / "app" / "static" / name), "a") as handle:
            handle.write(text)
        self.git(self.caller, "commit", "--quiet", "-m", "a page change", "--", "app/static/" + name)

    def publish(self) -> subprocess.CompletedProcess:
        return subprocess.run([BASH, "scripts/pages_publish.sh"], cwd=str(self.caller), env=self.env, capture_output=True, text=True, timeout=180)

    def state(self, with_pages: bool = False) -> Dict[str, object]:
        """The caller's checkout as a user would see it, and its refs: every file of the working tree (bytes), the index file, HEAD, the stash,
        the worktrees and every ref but gh-pages's two (with_pages: those too). git runs with --no-optional-locks: a plain status rewrites the
        index when it refreshes it."""
        tree = {str(p.relative_to(self.caller)): sha(p) for p in sorted(self.caller.rglob("*"))
                if p.is_file() and ".git" not in p.relative_to(self.caller).parts}
        refs = self.git(self.caller, "--no-optional-locks", "for-each-ref", "--format=%(refname) %(objectname)").splitlines()
        fetched = self.caller / ".git" / "FETCH_HEAD"   # the publish fetches into its worktree's own: the caller's is never written
        return {"tree": tree, "index": sha(self.caller / ".git" / "index"), "head": (self.caller / ".git" / "HEAD").read_text(),
                "fetch_head": sha(fetched) if fetched.exists() else None,
                "stash": self.git(self.caller, "--no-optional-locks", "stash", "list", "--format=%H %gs"),
                "status": self.git(self.caller, "--no-optional-locks", "status", "--porcelain=v2", "--branch", "--untracked-files=all"),
                "worktrees": self.git(self.caller, "--no-optional-locks", "worktree", "list", "--porcelain"),
                "refs": [r for r in refs if with_pages or "gh-pages" not in r]}

    def origin_refs(self) -> Dict[str, str]:
        lines = self.git(self.origin, "for-each-ref", "--format=%(refname) %(objectname)").splitlines()
        return dict(line.split(" ", 1) for line in lines)

    def expected_tree(self) -> Dict[str, str]:
        """{path: blob id} of a fresh build of the caller's page, as git would store it."""
        out = self.root / "expected_{}".format(len(list(self.root.glob("expected_*"))))
        done = run_build(out, self.caller / "scripts" / "pages_build.py")
        assert done.returncode == 0, done.stderr
        return {rel: self.git(self.caller, "hash-object", str(out / rel)).strip() for rel in files_of(out)}

    def pages_tree(self, rev: str = "gh-pages") -> Dict[str, str]:
        rows = self.git(self.origin, "ls-tree", "-r", "--full-tree", rev).splitlines()
        return {row.split("\t", 1)[1]: row.split()[2] for row in rows}

    def parents(self, rev: str) -> List[str]:
        return self.git(self.origin, "rev-list", "--parents", "-n", "1", rev).split()[1:]


@pytest.fixture
def repos(tmp_path):
    return Repos(tmp_path)


def test_the_first_publish_makes_gh_pages_an_orphan_branch_that_holds_exactly_the_build(repos):
    before, origin_before = repos.state(), repos.origin_refs()
    done = repos.publish()
    assert done.returncode == 0, done.stdout + done.stderr
    tip = repos.origin_refs()["refs/heads/gh-pages"]
    assert repos.parents(tip) == [], "the first commit of gh-pages has no parent: nothing of the source branches comes along"
    assert repos.pages_tree() == repos.expected_tree()
    assert {k: v for k, v in repos.origin_refs().items() if k != "refs/heads/gh-pages"} == origin_before, "no other branch of origin moved"
    assert repos.state() == before, "the caller's files, index, HEAD, stash, worktrees and other refs are what they were"
    local = repos.git(repos.caller, "rev-parse", "refs/heads/gh-pages").strip()
    assert local == tip, "the commit was added to the local gh-pages branch, which was pushed"
    source = repos.git(repos.caller, "rev-parse", "HEAD").strip()
    assert repos.git(repos.origin, "log", "-1", "--format=%s", tip).strip() == "Pages build of " + source
    lines = done.stdout.splitlines()
    assert "[pages] committed {} on gh-pages, the Pages copy of {}".format(tip, source) in lines, lines
    assert "[pages] pushed gh-pages to origin" in lines
    assert "[pages] Pages URL: " + PAGES_URL in lines
    assert list(repos.tmp.iterdir()) == [], "no temporary directory or worktree is left"


def test_a_second_publish_adds_one_fast_forward_commit_and_no_other_branch_moves(repos):
    assert repos.publish().returncode == 0
    first = repos.origin_refs()["refs/heads/gh-pages"]
    repos.commit_page_change()
    before, origin_before = repos.state(), repos.origin_refs()
    done = repos.publish()
    assert done.returncode == 0, done.stdout + done.stderr
    tip = repos.origin_refs()["refs/heads/gh-pages"]
    assert tip != first and repos.parents(tip) == [first], "one commit, on top of the last: a fast-forward"
    assert repos.pages_tree() == repos.expected_tree()
    assert {k: v for k, v in repos.origin_refs().items() if k != "refs/heads/gh-pages"} == \
        {k: v for k, v in origin_before.items() if k != "refs/heads/gh-pages"}
    assert repos.state() == before
    assert list(repos.tmp.iterdir()) == []


def test_a_publish_of_what_gh_pages_holds_already_commits_and_pushes_nothing(repos):
    assert repos.publish().returncode == 0
    before, origin_before = repos.state(with_pages=True), repos.origin_refs()
    done = repos.publish()
    assert done.returncode == 0, done.stdout + done.stderr
    assert "[pages] gh-pages holds this build already: nothing to publish" in done.stdout.splitlines()
    assert repos.origin_refs() == origin_before and repos.state(with_pages=True) == before
    assert list(repos.tmp.iterdir()) == []


@pytest.mark.parametrize("published_before", [False, True], ids=["first", "later"])
def test_a_push_that_origin_refuses_leaves_nothing_behind(repos, published_before):
    """origin turns the push down (here a pre-receive hook; in life, someone published meanwhile): the script fails, and the caller's
    repository is as it was, its local gh-pages included, with no worktree and no temporary directory left; origin has nothing new."""
    if published_before:
        assert repos.publish().returncode == 0
        repos.commit_page_change()
    hook = repos.origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'refused by the test' >&2\nexit 1\n")
    hook.chmod(0o755)
    before, origin_before = repos.state(with_pages=True), repos.origin_refs()
    done = repos.publish()
    assert done.returncode == 1, done.stdout + done.stderr
    errors = [line for line in done.stderr.splitlines() if line.startswith("ERROR")]
    assert errors == ["ERROR origin refused gh-pages (its own words are above): the local gh-pages is put back where it was"], done.stderr
    assert "refused by the test" in done.stderr, "git's own message is shown"
    assert repos.origin_refs() == origin_before
    assert repos.state(with_pages=True) == before
    assert list(repos.tmp.iterdir()) == []


NOT_COMMITTED = ("ERROR app/static has files that are not committed, ignored ones included: commit or remove them first, so that what is "
                 "published is a commit's page")


def test_a_page_with_changes_that_are_not_committed_is_refused_before_anything_is_made(repos):
    with open(str(repos.caller / "app" / "static" / "styles.css"), "a") as handle:
        handle.write("/* not committed */\n")
    before, origin_before = repos.state(with_pages=True), repos.origin_refs()
    done = repos.publish()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [NOT_COMMITTED]
    assert repos.origin_refs() == origin_before and repos.state(with_pages=True) == before
    assert list(repos.tmp.iterdir()) == []


def test_a_page_file_git_ignores_is_refused_too_before_anything_is_made(repos):
    """The build copies what is on disk, so a page file that git ignores would be published although no commit holds it (P9-B fix 1)."""
    exclude = repos.caller / ".git" / "info" / "exclude"
    exclude.write_text((exclude.read_text() if exclude.exists() else "") + "app/static/ignored.js\n")
    (repos.caller / "app" / "static" / "ignored.js").write_text("export const ignored = 1;\n")
    assert repos.git(repos.caller, "--no-optional-locks", "status", "--porcelain", "--", "app/static") == "", "git itself sees nothing to commit"
    before, origin_before = repos.state(with_pages=True), repos.origin_refs()
    done = repos.publish()
    assert done.returncode == 1
    assert done.stderr.splitlines() == [NOT_COMMITTED]
    assert repos.origin_refs() == origin_before and repos.state(with_pages=True) == before
    assert list(repos.tmp.iterdir()) == []


def test_a_gh_pages_checked_out_in_another_worktree_is_refused_before_anything_is_made(repos):
    assert repos.publish().returncode == 0   # a local gh-pages now
    repos.git(repos.caller, "worktree", "add", "--quiet", str(repos.root / "pages_checkout"), "gh-pages")
    before, origin_before = repos.state(with_pages=True), repos.origin_refs()
    done = repos.publish()
    assert done.returncode == 1
    assert done.stderr.splitlines() == ["ERROR gh-pages is checked out in a worktree of this repository: switch that worktree to another branch first"]
    assert repos.origin_refs() == origin_before and repos.state(with_pages=True) == before
    assert list(repos.tmp.iterdir()) == []


def test_a_finder_dotfile_in_app_static_neither_stops_the_publish_nor_is_published(repos):
    (repos.caller / "app" / "static" / ".DS_Store").write_text("finder\n")
    done = repos.publish()
    assert done.returncode == 0, done.stdout + done.stderr
    assert repos.pages_tree() == repos.expected_tree() and ".DS_Store" not in "".join(repos.pages_tree())


@pytest.mark.parametrize("url, pages", [
    ("git@github.com:Owner-Name/chat_pages_test.git", PAGES_URL),
    ("https://github.com/Owner-Name/owner-name.github.io", "https://owner-name.github.io/"),
    ("/srv/git/chat_pages_test.git", None),
])
def test_the_pages_url_follows_origin(repos, url, pages):
    repos.git(repos.caller, "remote", "set-url", "origin", url)
    repos.git(repos.caller, "config", "url.{}.insteadOf".format(repos.origin), url)
    done = repos.publish()
    assert done.returncode == 0, done.stdout + done.stderr
    expected = "[pages] Pages URL: " + pages if pages else "[pages] Pages URL: unknown, origin is not a github.com repository"
    assert expected in done.stdout.splitlines(), done.stdout
