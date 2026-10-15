"""CHAT_UI_PLAN.md P9-F fix 1 (3a): a turn that does not compare, recorded by value on the code before compare mode existed (1f4fe6d), so that a test
can hold the head's turn to it value for value and not only key for key (R2). Not a test file: tests/test_app_compare.py runs it on the head, and
the recording was made by running it on an extracted copy of 1f4fe6d. It imports nothing of the app itself (the caller hands it create_app), so
the same file runs on either tree.

One private tiny turn, every stage running: the synthetic gallery and the keyword labeller (tiny_gallery), and a reference, so that the score
stage runs too. What it records: the turn's SSE events, GET /v1/messages/<id>, the JSON and the Markdown export, and GET /v1/models. What two runs
of one turn cannot share is blanked by normalise: the ids, the clocks and timings, the commit the code came from, and torch's thread count (a
process setting). Synthetic data only (R7): the tiny engine and the tiny gallery.

It also records the environment it ran in (P9-F fix 2): the tiny decoder's beam ties and the similarity floats can differ across torch builds
and CPU architectures, so the recording is held to a run in its own environment only, exactly; skip_reason says why it is not, anywhere else.
"""
import io
import json
import platform
import re
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from PIL import Image

OPTIONS = {"max_new_tokens": 16, "reference": "Findings: The heart is enlarged. Small left pleural effusion. Impression: Cardiomegaly."}
VOLATILE = {"ms", "total_ms", "prefill_ms", "per_token_ms", "ts", "created_at", "updated_at", "git_sha", "git_dirty", "git_source", "threads"}
ID = re.compile(r"\b([a-z])_[0-9a-f]{22}\b")                  # app/ids.py new_id: a letter, then 22 hex digits that sort by time
TIME = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC")  # how the Markdown export writes a time
# The Markdown export's provenance line names only what is not None: the commit is there on a checkout and not on an extracted copy, so it goes;
# the line's timing and thread count are blanked as their JSON keys are.
MD_COMMIT = re.compile(r" · git_(?:sha|dirty|source)=[^ ·\n]+")
MD_NUMBERS = re.compile(r"\b(total_ms|threads)=[\d.]+")


def png() -> bytes:
    """The turn's X-ray: seeded noise, the same bytes on every run."""
    pixels = (np.random.default_rng(0).random((320, 320)) * 255).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(pixels).convert("L").save(out, "PNG")
    return out.getvalue()


def _frames(text: str) -> Iterator[Dict[str, Any]]:
    for frame in text.replace("\r\n", "\n").split("\n\n"):
        event, data = "message", []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip(" "))
        if data:
            yield {"event": event, "data": json.loads("\n".join(data))}


def normalise(value: Any) -> Any:
    """value with each id replaced by its letter and its place among that letter's ids (<m1>, <m2>, <s1>), every VOLATILE key's value blanked,
    and the Markdown's times, commit and timings blanked or dropped as above."""
    seen: Dict[str, Dict[str, str]] = {}

    def placeholder(match: Any) -> str:
        ids = seen.setdefault(match.group(1), {})
        return ids.setdefault(match.group(0), "<{}{}>".format(match.group(1), len(ids) + 1))

    def blank(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: (None if k in VOLATILE else blank(x)) for k, x in v.items()}
        if isinstance(v, list):
            return [blank(x) for x in v]
        if isinstance(v, str):
            return MD_NUMBERS.sub(r"\1=<n>", MD_COMMIT.sub("", TIME.sub("<time>", v)))
        return v

    return blank(json.loads(ID.sub(placeholder, json.dumps(value))))


def environment() -> Dict[str, str]:
    """The torch version and the CPU this process runs on, read the way a model card reads them (app/engine.py _make_card)."""
    import torch
    return {"torch": str(torch.__version__), "cpu": platform.processor() or platform.machine()}


def skip_reason(recorded: Dict[str, str], running: Dict[str, str]) -> Optional[str]:
    """None when the pin applies, the running environment being the one the turn was recorded in; else why it is skipped, naming both."""
    if running == recorded:
        return None
    return "R2 pin recorded on torch {} / {}; this is torch {} / {}".format(recorded["torch"], recorded["cpu"], running["torch"], running["cpu"])


def record_turn(create_app: Any, home: str) -> Dict[str, Any]:
    """Run the turn on create_app's app (a fresh CHAT_HOME at `home`) and -> what it recorded, normalised, with the environment it ran in."""
    from fastapi.testclient import TestClient
    with TestClient(create_app(engine="tiny", home=str(home), tiny_gallery=True)) as client:
        session_id = client.post("/v1/sessions", json={}).json()["id"]
        r = client.post("/v1/sessions/{}/messages".format(session_id), files={"image": ("chest.png", png(), "image/png")},
                        data={"text": "", "options": json.dumps(OPTIONS)})
        assert r.status_code == 200, r.text
        recorded = {"environment": environment(), "sse": list(_frames(r.text)),
                    "message": client.get("/v1/messages/{}".format(r.headers["x-message-id"])).json(),
                    "export_json": client.get("/v1/sessions/{}/export".format(session_id)).json(),
                    "export_md": client.get("/v1/sessions/{}/export?format=md".format(session_id)).text,
                    "models": client.get("/v1/models").json()}
    return normalise(recorded)


def stages_run(recorded: Dict[str, Any]) -> List[str]:
    """The stages of the recorded turn that ran (ended with a detail), in their order: what the pin covers."""
    return [e["data"]["stage"] for e in recorded["sse"] if e["event"] == "stage_end" and "detail" in e["data"]]
