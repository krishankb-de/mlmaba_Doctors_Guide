"""CHAT_UI_PLAN.md P4-B: the recorded turn the browser tests replay, tests/frontend/fixtures/turn_tiny.json.

It is one fresh tiny turn (create_app(engine="tiny"), max_new_tokens 16, png_bytes()) stored as the list of
{"event", "data"} objects the stream yields: the shape app/static/api.js hands to app/static/state.js. The tiny
engine has random weights and a toy vocabulary, so the file holds no MIMIC data and may be committed.

UPDATE_FIXTURES=1 records it again. Without it the test compares the file with a fresh turn, so a server change the
reducers have not seen (a new stage, a renamed field) fails here instead of passing against stale data.
"""
import json
import os
import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.engine import TinyEngine
from app.labels import RuleLabeler
from app.server import create_app
from app.tiny import TINY_VOCAB
from tests.app_helpers import iter_sse, png_bytes

FIXTURE = Path(__file__).resolve().parent / "frontend" / "fixtures" / "turn_tiny.json"
STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
STAGES = ["preprocess", "encode", "retrieve", "generate", "label", "score"]
UPDATE_HINT = "record it again with: UPDATE_FIXTURES=1 venv/bin/python -m pytest tests/test_app_frontend_fixture.py"


def _fresh_turn(home):
    """One tiny turn as [{"event", "data"}, ...], the objects api.js streamTurn yields."""
    with TestClient(create_app(engine="tiny", home=str(home))) as client:
        session_id = client.post("/v1/sessions", json={}).json()["id"]
        with client.stream("POST", "/v1/sessions/{}/messages".format(session_id),
                           files={"image": ("chest.png", png_bytes(), "image/png")},
                           data={"text": "", "options": json.dumps({"max_new_tokens": 16})}) as r:
            assert r.status_code == 200, r.read()
            return list(iter_sse(r.iter_text()))


def _dump(events):
    """One event per line, so a re-recording is a readable diff."""
    return "[\n" + ",\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n]\n"


def _fields(value):
    """A payload's field names, nested: what the reducers read, without the values that differ from run to run."""
    if isinstance(value, dict):
        return {key: _fields(sub) for key, sub in value.items()}
    if isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        return [_fields(value[0])]
    return None


def _recorded():
    assert FIXTURE.is_file(), "missing {}: {}".format(FIXTURE, UPDATE_HINT)
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_fixture_has_the_events_of_a_fresh_turn(tmp_path):
    fresh = _fresh_turn(tmp_path)
    if os.environ.get("UPDATE_FIXTURES", "") not in ("", "0"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(_dump(fresh), encoding="utf-8")
    recorded = _recorded()
    assert [e["event"] for e in recorded] == [e["event"] for e in fresh], UPDATE_HINT
    assert [_fields(e["data"]) for e in recorded] == [_fields(e["data"]) for e in fresh], UPDATE_HINT   # field names


def test_fixture_is_a_complete_turn_with_the_six_stages_in_contract_order():
    log = _recorded()
    assert [e["data"]["seq"] for e in log] == list(range(1, len(log) + 1))
    assert log[0]["event"] == "message_start" and log[-1]["event"] == "message_stop"
    assert log[-1]["data"]["status"] == "done"
    # A skipped stage emits only its stage_end, so the ends are the whole list (P3 skips retrieve, label and score).
    assert [e["data"]["stage"] for e in log if e["event"] == "stage_end"] == STAGES


def test_fixture_holds_only_tiny_engine_output():   # DUA: nothing MIMIC-derived is ever committed
    log = _recorded()
    card = log[0]["data"]["model"]
    assert card["name"] == "tiny" and card["checkpoint"] is None and card["checkpoint_sha256"] is None
    words = re.findall(r"[^\s.,]+", log[-1]["data"]["report"])
    # A heuristic: a trained model's report strays outside the toy vocabulary.
    assert words and set(words) <= set(TINY_VOCAB)
    assert not re.search(r'"/(Users|home|sc|tmp|var|private|opt)/', json.dumps(log))   # no path of the recording machine


# ---- P9-F: a recorded compare turn, tests/frontend/fixtures/turn_compare.json ------------------------------------------------------
# Two tiny engines named after the published models (create_app's tiny_models: random weights, a seed each, no checkpoint) and the keyword
# labeller, so that both reports are labelled. UPDATE_FIXTURES=1 records it again, as it does turn_tiny.json.

COMPARE_FIXTURE = FIXTURE.parent / "turn_compare.json"
COMPARE_MODELS = ("hybrid_150m_m3_rrg", "hybrid_150m_v2_rrg")


def _fresh_compare_turn(home):
    with TestClient(create_app(engine="tiny", home=str(home), tiny_models=COMPARE_MODELS, labeler=RuleLabeler())) as client:
        session_id = client.post("/v1/sessions", json={}).json()["id"]
        with client.stream("POST", "/v1/sessions/{}/messages".format(session_id),
                           files={"image": ("chest.png", png_bytes(), "image/png")},
                           data={"text": "", "options": json.dumps({"max_new_tokens": 16, "compare": True})}) as r:
            assert r.status_code == 200, r.read()
            return list(iter_sse(r.iter_text()))


def _recorded_compare():
    assert COMPARE_FIXTURE.is_file(), "missing {}: {}".format(COMPARE_FIXTURE, UPDATE_HINT)
    return json.loads(COMPARE_FIXTURE.read_text(encoding="utf-8"))


def test_the_compare_fixture_has_the_events_of_a_fresh_compare_turn(tmp_path):
    fresh = _fresh_compare_turn(tmp_path)
    if os.environ.get("UPDATE_FIXTURES", "") not in ("", "0"):
        COMPARE_FIXTURE.write_text(_dump(fresh), encoding="utf-8")
    recorded = _recorded_compare()
    assert [(e["event"], e["data"].get("index")) for e in recorded] == [(e["event"], e["data"].get("index")) for e in fresh], UPDATE_HINT
    assert [_fields(e["data"]) for e in recorded] == [_fields(e["data"]) for e in fresh], UPDATE_HINT


def test_the_compare_fixture_is_a_whole_compare_turn_with_both_blocks_and_both_label_sets():
    log = _recorded_compare()
    assert [e["data"]["seq"] for e in log] == list(range(1, len(log) + 1))
    start, stop = log[0]["data"], log[-1]["data"]
    assert start["compare"]["model"] == COMPARE_MODELS[1] and stop["status"] == "done" and stop["compare"]["model"] == COMPARE_MODELS[1]
    assert [e["data"]["stage"] for e in log if e["event"] == "stage_end"] == STAGES
    blocks = [e["data"]["index"] for e in log if e["event"] == "content_block_start"]
    assert blocks == [0, 1]
    label = next(e["data"]["detail"] for e in log if e["event"] == "stage_end" and e["data"]["stage"] == "label")
    assert set(label["compare"]) == {"chexbert_14", "positives"} and len(label["compare"]["chexbert_14"]) == 14
    assert stop["report"] != stop["compare"]["report"]


def test_the_compare_fixture_holds_only_tiny_engine_output():   # DUA: nothing MIMIC-derived is ever committed
    log = _recorded_compare()
    for card in (log[0]["data"]["model"], log[0]["data"]["compare"]["card"]):
        assert card["checkpoint"] is None and card["checkpoint_sha256"] is None and card["train_experiment"] is None
        assert card["drift_note"] == TinyEngine.drift_note
    for report in (log[-1]["data"]["report"], log[-1]["data"]["compare"]["report"]):
        words = re.findall(r"[^\s.,]+", report)
        assert words and set(words) <= set(TINY_VOCAB)
    assert not re.search(r'"/(Users|home|sc|tmp|var|private|opt)/', json.dumps(log))


def test_the_data_layer_modules_are_served_as_javascript_and_revalidated(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as client:
        for name in ("api.js", "state.js"):   # a browser refuses a module script with any other content type
            r = client.get("/static/" + name)
            assert r.status_code == 200 and "javascript" in r.headers["content-type"], name
            assert r.headers["cache-control"] == "no-cache" and r.content == (STATIC / name).read_bytes(), name
