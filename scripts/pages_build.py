"""The GitHub Pages copy of the chat page (CHAT_UI_PLAN.md P9-B).

    python scripts/pages_build.py OUT_DIR

The page the API serves, app/static/, built so that it works from any sub-path (https://<owner>.github.io/<repo>/) and talks to the API server its
Settings name (the drawer's "API server"):

    OUT_DIR/index.html   app/static/index.html with every /static/... reference made relative (static/...) and <meta name="cxr-pages" content="1">
    OUT_DIR/static/      every other file of app/static, byte for byte
    OUT_DIR/.nojekyll    GitHub Pages then serves the files as they are

The tag is how the page knows it is the Pages copy: there, an empty API server is no server (the page's own origin is github.io), so it asks
nothing of anyone until one is set (app/static/app.js, PAGES_META). The page the API serves is not changed at all (R8): app/static/ is only read.

Refused before anything is written (exit 2): an OUT_DIR inside this repository or inside any other checkout of this project, so that a build can
never be committed; an OUT_DIR that is not empty. Only app/static/ is copied, and of it only regular files of the kinds a page is made of (.html,
.js, .css), and in their own two folders its pictures (img/*.png) and its fonts with their licences (fonts/*.woff2, fonts/*.txt): a symlink, or a
file of any other kind or in any other place, refuses the build, so that nothing that strayed in from CHAT_HOME, outputs/ or results/ is ever
published (R7). Dotfiles (.DS_Store) stay behind.

The copy is then checked (exit 1): every module import, and every file that index.html or a stylesheet names, resolves to a file inside the copy,
and no absolute /static/ reference is left.

An ERROR line says what kind of problem it is and how many there are, one line per kind: never a file's name, a reference or a path (R7), so that
nothing of what strayed into app/static reaches the output. scripts/pages_publish.sh builds with this script and publishes the result to the
gh-pages branch.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC = REPO_ROOT / "app" / "static"
PAGES_TAG = '<meta name="cxr-pages" content="1">'    # what app/static/app.js looks for (PAGES_META)
KINDS = (".html", ".js", ".css")                     # what a page's code is made of: read and checked, as text
ASSETS = {"img": (".png",), "fonts": (".woff2", ".txt")}   # and what it shows, each kind in its own folder: copied as it is
CHARSET = re.compile(r'^([ \t]*)<meta charset="utf-8">[ \t]*(\r?\n)', re.M)
STATIC_REF = re.compile(r'(\b(?:href|src)[ \t]*=[ \t]*["\'])/static/', re.I)
# A reference that is still absolute: /static/ right after a quote, a backquote, an opening parenthesis (a CSS url) or an = (an unquoted
# attribute). A comment that names app/static/ is no reference.
ABSOLUTE_STATIC = re.compile(r'(?:["\'`(]|=)[ \t]*/static/')
HTML_REF = re.compile(r'\b(?:href|src)[ \t]*=[ \t]*(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'>]+))', re.I)
IMPORT_FROM = re.compile(r'^[ \t]*(?:import|export)\b[\s\w$,{}*]*?\bfrom[ \t]*(["\'])([^"\'\n]+)\1', re.M)
IMPORT_BARE = re.compile(r'^[ \t]*import[ \t]*(["\'])([^"\'\n]+)\1', re.M)
IMPORT_DYNAMIC = re.compile(r'\bimport[ \t]*\([ \t]*(["\'])([^"\'\n]+)\1[ \t]*\)')
CSS_URL = re.compile(r'url\([ \t]*(["\']?)([^"\')\s]+)\1[ \t]*\)')
CSS_IMPORT = re.compile(r'@import[ \t]+(["\'])([^"\']+)\1')
SCHEME = re.compile(r'^[a-z][a-z0-9+.-]*:', re.I)


class Refused(Exception):
    """The build cannot be made: nothing has been written. Its lines say why, a kind and a count each, never a name or a path (R7)."""

    def __init__(self, *lines: str):
        super().__init__(*lines)
        self.lines = lines


def counted(n: int, one: str, many: str) -> str:
    """'1 symlink' or '2 symlinks': how many of a kind."""
    return "{} {}".format(n, one if n == 1 else many)


def checkout_holding(path: Path) -> Optional[Path]:
    """This repository, or the other checkout of this project (a .git beside hybrid_xmamba/), that holds `path`; None for a path outside all."""
    for directory in (path,) + tuple(path.parents):
        if directory == REPO_ROOT or ((directory / ".git").exists() and (directory / "hybrid_xmamba").is_dir()):
            return directory
    return None


def page_files() -> List[Path]:
    """The files of app/static to copy, relative to it, sorted. A dotfile is left behind; symlinks, and files a page is not made of, refuse the
    build, with how many of each there are."""
    found, links, others = [], 0, 0
    for top, dirs, files in os.walk(str(STATIC)):   # a symlinked directory is listed, never entered
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        links += sum(1 for name in dirs + files if not name.startswith(".") and os.path.islink(os.path.join(top, name)))
        for name in sorted(files):
            if name.startswith(".") or os.path.islink(os.path.join(top, name)):
                continue
            rel = (Path(top) / name).relative_to(STATIC)
            if rel.suffix not in KINDS and rel.suffix not in ASSETS.get(str(rel.parent), ()):
                others += 1
                continue
            found.append(rel)
    refused = []
    if links:
        refused.append(counted(links, "symlink", "symlinks") + " refused: the build copies regular files only")
    if others:
        refused.append(counted(others, "file of another kind", "files of another kind")
                       + " refused: a page is made of .html, .js and .css, img/*.png and fonts/*.woff2 and *.txt, and nothing else is published")
    if refused:
        raise Refused(*refused)
    return found


def pages_index(page: str) -> str:
    """app/static/index.html made into the Pages copy's: its /static/ references relative, the Pages tag after the charset."""
    if PAGES_TAG in page:
        raise Refused("the page's index.html carries the Pages tag itself: the page the API serves must not")
    page, made_relative = STATIC_REF.subn(r"\1static/", page)
    if not made_relative:
        raise Refused("the page's index.html names no /static/ file: it is not the page this script knows how to copy")
    page, tagged = CHARSET.subn(lambda m: m.group(0) + m.group(1) + PAGES_TAG + m.group(2), page, count=1)
    if not tagged:
        raise Refused('the page\'s index.html has no <meta charset="utf-8"> line for the Pages tag to follow')
    return page


def build(out: Path) -> List[Path]:
    """Writes the copy into `out` (an empty or missing directory). -> the files written, relative to `out`."""
    files = page_files()
    if Path("index.html") not in files:
        raise Refused("the page has no index.html")
    index = pages_index((STATIC / "index.html").read_bytes().decode("utf-8"))   # bytes, so that its line endings stay as they are
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for rel in files:
        if rel == Path("index.html"):
            continue
        target = out / "static" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(STATIC / rel), str(target))
        written.append(Path("static") / rel)
    (out / "index.html").write_bytes(index.encode("utf-8"))
    (out / ".nojekyll").write_bytes(b"")
    return [Path("index.html"), Path(".nojekyll")] + written


def references(path: Path, text: str) -> Iterator[str]:
    """What a file of the copy loads: index.html's href and src, a module's imports, a stylesheet's url() and @import."""
    if path.suffix == ".html":
        for m in HTML_REF.finditer(text):
            yield next(group for group in m.groups() if group is not None)
    elif path.suffix == ".js":
        for pattern in (IMPORT_FROM, IMPORT_BARE, IMPORT_DYNAMIC):
            for m in pattern.finditer(text):
                yield m.group(2)
    elif path.suffix == ".css":
        for pattern in (CSS_URL, CSS_IMPORT):
            for m in pattern.finditer(text):
                yield m.group(2)


def problems_of(out: Path) -> List[str]:
    """What is wrong with the copy in `out`: one line per kind of problem, with how many there are; none when it is right."""
    absolute = not_relative = no_file = 0
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.suffix in KINDS):
        text = path.read_text(encoding="utf-8")
        absolute += len(ABSOLUTE_STATIC.findall(text))
        for ref in references(path, text):
            if ref.lower().startswith("data:") or ref.startswith("#"):
                continue   # inline (the page's empty icon), or a place in the page itself
            if path.suffix == ".js" and not ref.startswith(("./", "../")):
                not_relative += 1
                continue
            target = (path.parent / ref.split("#", 1)[0].split("?", 1)[0]).resolve()
            if SCHEME.match(ref) or ref.startswith("/") or out not in target.parents or not target.is_file():
                no_file += 1
    found = []
    if absolute:
        found.append(counted(absolute, "absolute /static/ reference", "absolute /static/ references") + " left")
    if not_relative:
        found.append(counted(not_relative, "import that is not relative", "imports that are not relative"))
    if no_file:
        found.append(counted(no_file, "reference to no file of the copy", "references to no file of the copy"))
    return found


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/pages_build.py", allow_abbrev=False,
                                     description="Build the GitHub Pages copy of the chat page (CHAT_UI_PLAN.md P9-B).")
    parser.add_argument("out_dir", metavar="OUT_DIR", help="an empty directory outside every checkout of this project (made when missing)")
    args = parser.parse_args(argv)
    out = Path(args.out_dir).expanduser().resolve()
    try:
        if checkout_holding(out) is not None:
            raise Refused("OUT_DIR is inside a checkout of this project, where the build could be committed: build outside it")
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise Refused("OUT_DIR must be an empty directory, or one that is not there yet")
        written = build(out)
    except Refused as exc:
        for line in exc.lines:
            sys.stderr.write("ERROR pages_build: {}\n".format(line))
        return 2
    found = problems_of(out)
    for problem in found:
        sys.stderr.write("ERROR pages_build: {}\n".format(problem))
    if found:
        return 1
    statics = sum(1 for rel in written if rel.parts[0] == "static")
    print("[pages] built {} files: index.html, .nojekyll and static/ ({} files)".format(len(written), statics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
