"""CHAT_UI_PLAN.md P9-B: the GitHub Pages copy of the page, in a real Chrome, talking to an API on another origin.

The copy is scripts/pages_build.py's own, built under a sub-path (GitHub Pages serves a project's site under /<repo>/) and served from a free
loopback port by a plain static server (python -m http.server). The API is the tiny app with the tiny gallery on another free port, its CORS
letting the copy's origin in (create_app's cors_origins, which the CLI's --cors-origin sets). Both origins are loopback, so Chrome sends no
Private Network Access preflight here: tests/test_app_api.py covers that answer. Neither port is 8000 or 8010. The harness is conftest.py's: a
CORS failure is a console error and a failed request there, so UI.assert_clean fails on one.

    venv/bin/python -m pytest tests/e2e -m e2e -q
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, List

import pytest

pytest.importorskip("playwright.sync_api", reason="the e2e tests need `pip install -r requirements-e2e.txt`")

from scripts.chat_ui_cdp import free_port, stop_group  # noqa: E402
from tests.e2e.conftest import REPO_ROOT, TinyServer  # noqa: E402
from tests.e2e.test_ui_playwright import GALLERY_STAGES, LABELS_JS, NEIGHBOURS_LOADED, PICTURES_JS, PICTURES_LOADED  # noqa: E402

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

FIRST_RUN = "Set the API server in Settings, for example http://localhost:8000 (your tunnel)."
SITE = "chat_pages"   # the sub-path the copy is served under, as GitHub Pages serves a project's site


class PagesServer(TinyServer):
    """The tiny API with the tiny gallery, letting one origin in, and that origin: the Pages copy, built fresh under /chat_pages/ and served by
    python -m http.server from a port of its own, in a process group of its own. The two stop together."""

    def __init__(self, home: str, **kwargs: Any) -> None:
        self.pages_port = free_port()
        self.pages_origin = "http://127.0.0.1:{}".format(self.pages_port)
        self.pages_url = "{}/{}/".format(self.pages_origin, SITE)
        root = Path(home).parent / "site"
        built = subprocess.run([sys.executable, str(Path(REPO_ROOT) / "scripts" / "pages_build.py"), str(root / SITE)],
                               capture_output=True, text=True, timeout=60)
        assert built.returncode == 0, built.stdout + built.stderr
        self.static_log = open(str(Path(home).parent / "static_server.log"), "a")
        self.static = subprocess.Popen([sys.executable, "-m", "http.server", str(self.pages_port), "--bind", "127.0.0.1", "--directory", str(root)],
                                       stdout=self.static_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.time() + 30
            while True:
                try:
                    with urllib.request.urlopen(self.pages_url, timeout=2):
                        break
                except OSError:
                    if time.time() > deadline or self.static.poll() is not None:
                        raise RuntimeError("the static server of the Pages copy did not start")
                    time.sleep(0.1)
            super().__init__(home, cors_origin=self.pages_origin, **kwargs)
        except BaseException:
            stop_group(self.static)
            self.static_log.close()
            raise

    def stop(self) -> None:
        try:
            super().stop()
        finally:
            stop_group(self.static)
            self.static_log.close()


def _notice(page: Any) -> Any:
    return page.evaluate("(() => { const n = document.querySelector('#notice'); return n && !n.hidden ? n.querySelector('p').textContent : null; })()")


@pytest.mark.parametrize("ui", [{"tiny_gallery": True, "server": PagesServer}], indirect=True, ids=["pages_copy"])
def test_the_pages_copy_asks_nothing_until_its_api_server_is_set_and_then_runs_a_whole_chat_on_that_server(ui, images):
    """The Pages copy's first run shows the notice and asks nothing of anyone: no /v1/ request and no /healthz, on any origin. The API server
    set in Settings, a turn streams from it across origins, its pictures load (the card's two and the similar X-rays), its labels show, an
    export downloads, and the sidebar lists the chat; a reload of the copy opens that chat again from the API server, pictures and all."""
    server = ui.server
    page = ui.page
    asked: List[str] = []   # every URL the page asked for, whole (UI.requests keeps the path only)
    page.on("request", lambda request: asked.append(request.url))
    page.goto(server.pages_url)
    page.wait_for_function("(() => { const n = document.querySelector('#notice'); return !!n && !n.hidden; })()")
    page.wait_for_timeout(1500)   # what the page asks at its start, it has asked by now
    assert _notice(page) == FIRST_RUN
    assert asked and all(url.startswith(server.pages_url) for url in asked), asked   # its own files, and nothing else
    assert not [url for url in asked if "/v1/" in url or url.endswith("/healthz")], asked
    assert page.evaluate("document.querySelector('#mode-badge').textContent") == ""
    ui.shot("p9b_first_run_1280x900_light.png")

    page.click("#settings")
    page.get_by_label("API server").fill(server.url)   # http://127.0.0.1:<port>/, kept as a browser names the origin
    page.click("#drawer-save")
    page.wait_for_function("document.querySelector('#drawer').hidden && document.querySelector('#mode-badge').textContent !== ''")
    page.wait_for_function("document.querySelector('#notice').hidden")
    assert ui.drawer()["stored"]["api_server"] == server.url.rstrip("/")
    # The first server is no change (P9-B fix 2): the address the copy had (none) is kept, and home becomes the server's newest chat, here none.
    # Waited for rather than read once, so that the check does not race the router.
    page.wait_for_function("location.hash === '#/new'", timeout=10000)

    card = ui.turn(images["xray_a.png"])
    assert card["status"] == "done" and card["stages"] == GALLERY_STAGES and card["spinning"] == 0, card
    page.wait_for_function(PICTURES_LOADED)
    page.wait_for_function(NEIGHBOURS_LOADED)
    shown = page.evaluate(PICTURES_JS)
    assert all(tile["img"]["blob"] for tile in shown["tiles"])   # fetched with the page's auth from the API server, shown as object URLs
    labels = page.evaluate(LABELS_JS)
    assert labels["chips"] == 14 and labels["positive"] >= 1 and labels["notes"] == []
    sid = ui.session_id()
    sidebar = ui.sidebar()
    assert [(s["id"], s["current"]) for s in sidebar] == [(sid, True)] and sidebar[0]["meta"].endswith("1 turn"), sidebar

    with page.expect_download() as download:
        page.click("#exports button[data-format='json']")
    saved = download.value
    assert saved.suggested_filename == "session-{}.json".format(sid)
    assert json.loads(Path(saved.path()).read_text(encoding="utf-8"))["session"]["id"] == sid

    ui.reload()   # the copy again: it keeps its API server, and opens the newest chat from it
    page.wait_for_function(PICTURES_LOADED)
    page.wait_for_function(NEIGHBOURS_LOADED)
    again = ui.card()
    assert again["n"] == 1 and again["status"] == "done" and page.evaluate("location.hash") == "#/s/{}".format(sid)
    assert page.evaluate(LABELS_JS)["chips"] == 14

    api = server.url.rstrip("/")
    assert {url.split("/", 3)[2] for url in asked} == {server.pages_origin.split("//")[1], api.split("//")[1]}   # the two origins, no third
    assert any(url.startswith(api + "/v1/messages/") and "/image" in url for url in asked)
    assert any(url.startswith(api + "/v1/gallery/images/") for url in asked)
    ui.shot("p9b_turn_across_origins_1280x900_light.png")
    ui.assert_clean()
