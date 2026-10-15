"""CHAT_UI_PLAN.md P9-F: compare mode. One X-ray decoded by a Mamba-3 decoder and by the 13D decoder, the two reports and their labels side by
side in one card.

On the tiny stack only (R7): create_app's tiny_models serves the tiny engine under the published models' names, each name its own seed, so the
two reports differ; the keyword labeller and, where a test needs one, the synthetic gallery. No checkpoint is loaded, and no text is MIMIC's.
"""
import json
import re
import threading
from pathlib import Path

import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import cli, server
from app.engine import TinyEngine
from app.labels import CHEXBERT_14, RuleLabeler
from app.pipeline import INCUMBENT_MODEL, MAMBA3_MODELS, compare_available, compare_partner
from app.redact import PUBLIC_ERROR_MESSAGE, redact_card, redact_event
from app.schemas import Options
from app.server import NO_CACHE_MSG, NO_COMPARE_MSG, create_app
from app.store import _compare_lines
from tests.app_helpers import iter_sse, png_bytes, start_live_server
from tests.r2_pin import environment, record_turn, skip_reason, stages_run

M3, M3EOS, D13 = "hybrid_150m_m3_rrg", "hybrid_150m_m3_rrg_eos", "hybrid_150m_v2_rrg"
# What an Options dump held before P9-F, in its order: a turn that does not compare is dumped exactly so (R2).
NON_COMPARE_KEYS = ["model", "decode", "beam_size", "max_new_tokens", "cached_decode", "compile", "k_images", "k_reports", "label",
                    "reference", "display_repair", "stop_on_repeat", "test_row"]


def _app(tmp_path, names=(M3, D13), **kw):
    return create_app(engine="tiny", home=str(tmp_path), tiny_models=names, **kw)


def _post(c, sid, options, image=True, text="", headers=None):
    files = {"image": ("x.png", png_bytes(), "image/png")} if image else None
    return c.post("/v1/sessions/{}/messages".format(sid), files=files, data={"text": text, "options": json.dumps(options)}, headers=headers)


def _turn(c, sid, options, image=True, text="", headers=None):
    r = _post(c, sid, options, image=image, text=text, headers=headers)
    assert r.status_code == 200, r.text
    return list(iter_sse([r.text]))


# ---- the option -----------------------------------------------------------------------------------------------------------------

def test_compare_is_an_option_that_is_off_by_default_and_is_dumped_only_when_it_is_on():
    assert Options().compare is False and Options(compare=True).compare is True
    assert list(Options().model_dump()) == NON_COMPARE_KEYS                       # R2: a turn that does not compare dumps as before
    assert list(Options(compare=False).model_dump()) == NON_COMPARE_KEYS
    assert "compare" not in json.loads(Options().model_dump_json())
    on = Options(compare=True, model=M3).model_dump()
    assert list(on) == NON_COMPARE_KEYS + ["compare"] and on["compare"] is True and on["model"] == M3
    assert json.loads(Options(compare=True).model_dump_json())["compare"] is True
    assert "compare" in Options.model_fields                                       # so the reference names it among the options
    with pytest.raises(ValidationError):
        Options(compare="sometimes")


# ---- the tiny stack's two engines -------------------------------------------------------------------------------------------------

def test_tiny_models_serves_the_tiny_engine_under_each_name_each_with_a_seed_of_its_own(tmp_path):
    with TestClient(_app(tmp_path / "a")) as c:
        engines, default = c.app.state.engines, c.app.state.default_model
    assert list(engines) == [M3, D13] and default == M3                            # the first is the default, as --models has it
    for name, engine in engines.items():
        card = engine.card()
        assert isinstance(engine, TinyEngine) and engine.name == card["name"] == name
        assert card["checkpoint"] is None and card["checkpoint_sha256"] is None and card["drift_note"] == TinyEngine.drift_note
    first, second = engines.values()
    weights = lambda e: torch.cat([p.detach().flatten() for p in e.decoder.parameters()])   # noqa: E731
    assert not torch.equal(weights(first), weights(second))                       # two models, not one model twice
    assert torch.equal(weights(first), weights(TinyEngine()))                     # seed 0 is the tiny engine of every other test
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "b"))) as c:
        assert list(c.app.state.engines) == ["tiny"] and c.app.state.default_model == "tiny"   # without it, the one tiny engine as before


FRONTEND_FIXTURES = Path(__file__).resolve().parent / "frontend" / "fixtures"


def test_a_tiny_engine_under_another_name_is_marked_a_test_model_and_the_default_tiny_card_is_as_it_was():   # P9-F fix 2
    """The tiny engine served under a published model's name says on its card that it is a test model, so that the page marks it as it marks
    the engine called tiny (the banner's badge, the card's notice). The default tiny card is the card it always was: no such key."""
    renamed = TinyEngine(name=M3).card()
    assert renamed["test_model"] is True and renamed["name"] == M3 and list(renamed)[-1] == "test_model"
    default = TinyEngine().card()
    assert "test_model" not in default
    recorded = json.loads((FRONTEND_FIXTURES / "turn_tiny.json").read_text(encoding="utf-8"))[0]["data"]["model"]
    assert list(default) == list(recorded)                                   # its keys, in their order, as the turn recorded before P9-F has them
    assert redact_card(renamed, "public") == renamed and redact_card(renamed, "public")["test_model"] is True   # not sensitive: kept as it is


def test_the_real_card_builder_never_marks_a_test_model(tmp_path, monkeypatch):   # P9-F fix 2
    import sys
    from types import SimpleNamespace

    import scripts.evaluate_report_generation as erg
    from app.engine import RealEngine
    from app.tiny import TinyTokenizer, TinyTower, tiny_decoder, tiny_prefix_mapper
    module = SimpleNamespace(image_encoder=TinyTower(), prefix_mapper=tiny_prefix_mapper(), decoder=tiny_decoder(), prefix_k=4)
    monkeypatch.setattr(erg, "load_report_generation_module", lambda checkpoint, model_config, device="cpu", **kw: module)   # nothing is loaded
    # The GPT-2 tokenizer is stubbed in sys.modules, as tests/test_report_eos_eval.py stubs it: reaching the real transformers.AutoTokenizer
    # imports sentencepiece, a SWIG module whose import warns on Python 3.14, for nothing this test needs (fix round 2, item 4).
    tokenizers = SimpleNamespace(from_pretrained=lambda name, **kw: TinyTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizers))
    checkpoint = tmp_path / "run" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a real checkpoint" * 50)
    threads = torch.get_num_threads()
    try:
        engines = [RealEngine(str(checkpoint), M3), RealEngine(str(checkpoint), M3, name=M3EOS)]   # under its config's name, and another
    finally:
        torch.set_num_threads(threads)                                       # RealEngine sets it for the whole process
    assert all(engine.module is module and isinstance(engine.tokenizer, TinyTokenizer) for engine in engines)   # the stubs, and nothing else
    cards = [engine.card() for engine in engines]
    assert [card["name"] for card in cards] == [M3, M3EOS] and all("test_model" not in card for card in cards)


def test_two_tiny_engines_write_different_reports_of_the_same_xray(tmp_path):
    with TestClient(_app(tmp_path)) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        reports = [_turn(c, sid, {"max_new_tokens": 24, "model": name})[-1]["data"]["report"] for name in (M3, D13)]
    assert reports[0] and reports[1] and reports[0] != reports[1]


def test_tiny_models_is_for_the_tiny_engine_only_and_names_each_model_once(tmp_path):
    with pytest.raises(ValueError):
        create_app(engine="real", home=str(tmp_path / "a"), models=(M3,), tiny_models=(M3, D13))
    for bad in ((), (M3, M3), (M3, "")):
        with pytest.raises(ValueError):
            create_app(engine="tiny", home=str(tmp_path / "b"), tiny_models=bad)


# ---- the pair rule ----------------------------------------------------------------------------------------------------------------

def test_the_pair_rules_names_are_the_command_lines_aliases():
    assert MAMBA3_MODELS == (cli.MODEL_ALIASES["m3eos"], cli.MODEL_ALIASES["m3"]) and INCUMBENT_MODEL == cli.MODEL_ALIASES["13d"]
    assert set(MAMBA3_MODELS) | {INCUMBENT_MODEL} == set(server.MODEL_NAMES)


@pytest.mark.parametrize("model, served, partner", [
    (M3, (M3, D13), D13),                     # a Mamba-3 model is compared with 13D
    (M3EOS, (M3EOS, M3, D13), D13),
    (M3, (M3EOS, M3, D13), D13),              # whichever is the default
    (D13, (M3EOS, M3, D13), M3EOS),           # 13D: the default, when that is a Mamba-3 model
    (D13, (M3, M3EOS, D13), M3),
    (D13, (D13, M3, M3EOS), M3EOS),           # else m3eos, when it is served
    (D13, (D13, M3), M3),                     # else m3
    (M3, (M3, M3EOS), None),                  # no 13D: no partner
    (M3EOS, (M3EOS,), None),
    (D13, (D13,), None),                      # no Mamba-3 model: none either
    ("tiny", ("tiny",), None),                # a model that is neither
])
def test_the_pair_rule_for_every_case(model, served, partner):
    assert compare_partner(model, served, served[0]) == partner


@pytest.mark.parametrize("served, available", [
    ((M3, D13), True), ((D13, M3EOS), True), ((M3EOS, M3, D13), True),
    ((M3EOS, M3), False), ((M3,), False), ((D13,), False), (("tiny",), False),
])
def test_compare_is_available_exactly_when_a_mamba3_model_and_13d_are_both_served(served, available):
    assert compare_available(served) is available
    assert all((compare_partner(m, served, served[0]) is not None) is available for m in served if m != "tiny")


# ---- the route: the refusal, the cache, /v1/models -------------------------------------------------------------------------------

def test_a_compare_turn_without_a_partner_is_refused_before_it_is_accepted(tmp_path):
    for home, kw in (("tiny", {}), ("m3s", {"tiny_models": (M3EOS, M3)}), ("d13", {"tiny_models": (D13,)})):
        with TestClient(create_app(engine="tiny", home=str(tmp_path / home), **kw)) as c:
            sid = c.post("/v1/sessions", json={}).json()["id"]
            r = _post(c, sid, {"compare": True, "max_new_tokens": 16})
            assert r.status_code == 422, home
            assert r.json() == {"type": "error", "error": {"type": "validation_error", "message": NO_COMPARE_MSG}}
            assert c.get("/v1/sessions/{}".format(sid)).json()["messages"] == []          # nothing was stored: no turn
            assert _turn(c, sid, {"max_new_tokens": 16})[-1]["data"]["status"] == "done"   # and a turn that does not compare runs
    assert NO_COMPARE_MSG == "Compare needs a Mamba-3 model and the 13D model on this server."


def test_a_compare_turn_whose_partner_has_no_decode_cache_is_refused_in_the_engines_own_words(tmp_path):
    app = _app(tmp_path)
    app.state.engines[D13]._card["cached_decode_available"] = False   # as the published 13D decoder (Mamba-1 has no step())
    with TestClient(app) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        r = _post(c, sid, {"compare": True, "max_new_tokens": 16})      # cached_decode defaults to true, and block 1 decodes as block 0 does
        assert r.status_code == 422 and r.json()["error"]["message"] == NO_CACHE_MSG.format(D13)
        frames = _turn(c, sid, {"compare": True, "max_new_tokens": 16, "cached_decode": False})
        assert frames[-1]["data"]["status"] == "done"


def test_the_compare_checks_come_after_compile_and_before_the_test_row(tmp_path):
    app = _app(tmp_path)
    app.state.engines[D13]._card["cached_decode_available"] = False
    with TestClient(app) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        refused = lambda options: _post(c, sid, options).json()["error"]["message"]   # noqa: E731
        assert refused({"compare": True, "compile": True, "test_row": 0}) == "compile is not enabled on this server."
        assert refused({"compare": True, "test_row": 0}) == NO_CACHE_MSG.format(D13)
        assert "gallery" in refused({"compare": True, "cached_decode": False, "test_row": 0})   # the test row's own check comes next
    with TestClient(create_app(engine="tiny", home=str(tmp_path / "plain"))) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        assert _post(c, sid, {"compare": True, "test_row": 0}).json()["error"]["message"] == NO_COMPARE_MSG


def test_the_compare_checks_are_for_a_turn_that_runs_a_model_and_a_question_with_compare_is_answered_as_any_question(tmp_path):   # P9-F fix 2
    """A question or a note runs no model (the fixed answer), so compare has nothing to check for it: it is accepted, and answered as round 1
    streams it, on a server that serves no partner and on one whose partner has no decode cache. A turn that runs a model is checked as before:
    an image, or text that runs the chat's last X-ray again (a command, or no text at all)."""
    plain = create_app(engine="tiny", home=str(tmp_path / "plain"))              # no partner served
    no_cache = _app(tmp_path / "no_cache")
    no_cache.state.engines[D13]._card["cached_decode_available"] = False         # the partner cannot decode as cached_decode asks
    for app, refusal in ((plain, NO_COMPARE_MSG), (no_cache, NO_CACHE_MSG.format(D13))):
        with TestClient(app) as c:
            sid = c.post("/v1/sessions", json={}).json()["id"]
            _turn(c, sid, {"max_new_tokens": 16})                                # the chat has an X-ray, so text alone is accepted
            for text in ("is this pneumonia?", "thanks"):                        # a question, a note
                asked = _turn(c, sid, {"compare": True, "cached_decode": True, "max_new_tokens": 16}, image=False, text=text)
                usual = _turn(c, sid, {"cached_decode": True, "max_new_tokens": 16}, image=False, text=text)
                assert [f["event"] for f in asked] == ["message_start", "warning", "message_stop"], (refusal, text)
                assert asked[1]["data"]["code"] == "not_a_command" and asked[0]["data"]["options"]["compare"] is True
                stable = _stable(asked, _ids(asked))
                del stable[0]["data"]["options"]["compare"]
                assert stable == _stable(usual, _ids(usual)), (refusal, text)       # the usual answer, as without compare
            for text in ("beam 2", ""):                                          # the chat's last X-ray runs again: checked
                r = _post(c, sid, {"compare": True, "cached_decode": True, "max_new_tokens": 16}, image=False, text=text)
                assert (r.status_code, r.json()["error"]["message"]) == (422, refusal), text
            r = _post(c, sid, {"compare": True, "cached_decode": True, "max_new_tokens": 16})   # an image: checked
            assert (r.status_code, r.json()["error"]["message"]) == (422, refusal)


def test_models_says_compare_exactly_when_a_mamba3_model_and_13d_are_both_served_and_each_card_names_its_partner(tmp_path):
    cases = [((M3, D13), {M3: D13, D13: M3}), ((M3EOS, M3, D13), {M3EOS: D13, M3: D13, D13: M3EOS}), ((D13, M3), {D13: M3, M3: D13})]
    for n, (names, partners) in enumerate(cases):
        with TestClient(_app(tmp_path / str(n), names)) as c:
            listed = c.get("/v1/models").json()
        assert listed["features"] == {"retrieval": False, "labels": False, "compare": True}, names
        assert {card["name"]: card["compare_with"] for card in listed["models"]} == partners
    for n, kw in enumerate(({}, {"tiny_models": (M3EOS, M3)}, {"tiny_models": (D13,)})):
        with TestClient(create_app(engine="tiny", home=str(tmp_path / "none{}".format(n)), **kw)) as c:
            listed = c.get("/v1/models").json()
        assert listed["features"] == {"retrieval": False, "labels": False}   # no compare: the answer is what it always was (R2)
        assert all("compare_with" not in card for card in listed["models"])


# ---- one turn, two decoders: the events ---------------------------------------------------------------------------------------------

STAGES = ["preprocess", "encode", "retrieve", "generate", "label", "score"]


class CountingLabeler(RuleLabeler):
    """The keyword labeller, with every call it gets kept: the label stage labels both reports in one call."""

    def __init__(self):
        self.calls = []

    def label(self, texts):
        self.calls.append(list(texts))
        return super().label(texts)


VOLATILE = {"ms", "total_ms", "prefill_ms", "per_token_ms", "ts", "created_at", "updated_at"}   # timings and clocks


def _stable(value, ids):
    """`value` with every timing and clock blanked and each id in `ids` replaced by its placeholder: what two runs of one turn share."""
    text = json.dumps(value)
    for real, placeholder in ids.items():
        text = text.replace(real, placeholder)

    def blank(v):
        if isinstance(v, dict):
            return {k: (None if k in VOLATILE else blank(x)) for k, x in v.items()}
        if isinstance(v, list):
            return [blank(x) for x in v]
        return v

    return blank(json.loads(text))


def _stable_md(text, ids):
    """A Markdown export as _stable leaves a JSON one: its times and its total_ms said as text are blanked too."""
    text = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC", "<time>", _stable(text, ids))
    return re.sub(r"total_ms=[\d.]+", "total_ms=<ms>", text)


def _ids(frames):
    start = frames[0]["data"]
    return {start["message_id"]: "<message>", start["user_message_id"]: "<user>", start["session_id"]: "<session>"}


def _blocks(frames):
    """index -> the events of that content block, in order."""
    out = {}
    for f in frames:
        if f["event"].startswith("content_block_"):
            out.setdefault(f["data"]["index"], []).append(f)
    return out


def _stage_end(frames, stage):
    return next(f["data"] for f in frames if f["event"] == "stage_end" and f["data"]["stage"] == stage)


def test_a_compare_turn_decodes_block_0_then_block_1_labels_both_in_one_call_and_ends_with_both(tmp_path):
    labeler = CountingLabeler()
    app = _app(tmp_path, labeler=labeler)
    engines = app.state.engines
    with TestClient(app) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        frames = _turn(c, sid, {"compare": True, "max_new_tokens": 16})
        alone = _turn(c, sid, {"model": D13, "max_new_tokens": 16})[-1]["data"]   # the partner on the same X-ray, in a turn of its own
        stored = c.get("/v1/messages/{}".format(frames[0]["data"]["message_id"])).json()
    names = [f["event"] for f in frames]
    start, stop = frames[0]["data"], frames[-1]["data"]
    assert [f["data"]["seq"] for f in frames] == list(range(1, len(frames) + 1))
    assert names[0] == "message_start" and names[-1] == "message_stop" and stop["status"] == "done"
    assert [f["data"]["stage"] for f in frames if f["event"] == "stage_end"] == STAGES   # the same six stages, in their order
    # message_start: the partner and its card, and the resolved options say compare
    assert start["compare"] == {"model": D13, "card": engines[D13].card()} and start["model"] == engines[M3].card()
    assert start["options"]["compare"] is True and start["options"]["model"] == M3
    # encode ran both towers; generate decoded block 0, then block 1, inside the one stage
    encode = _stage_end(frames, "encode")["detail"]
    assert set(encode["compare"]) == set(encode) - {"compare"}
    generate_at = [i for i, f in enumerate(frames) if f["event"] in ("stage_start", "stage_end") and f["data"]["stage"] == "generate"]
    inside = [(f["event"], f["data"]["index"]) for f in frames[generate_at[0] + 1:generate_at[1]]]
    deltas = {index: len(block) - 2 for index, block in _blocks(frames).items()}
    assert inside == ([("content_block_start", 0)] + [("content_block_delta", 0)] * deltas[0] + [("content_block_stop", 0)]
                      + [("content_block_start", 1)] + [("content_block_delta", 1)] * deltas[1] + [("content_block_stop", 1)])
    assert _blocks(frames)[1][0]["data"]["content_block"] == {"type": "report", "text": ""}
    generate = _stage_end(frames, "generate")["detail"]
    assert generate["compare"]["tokens"] == deltas[1] == 16 and generate["tokens"] == deltas[0] == 16
    assert {k: generate["compare"][k] for k in ("decode", "beam_size", "cached_decode", "stopped")} == {
        "decode": "beam", "beam_size": 3, "cached_decode": True, "stopped": "budget"}   # the same decode options as block 0
    # block 1 is what the partner writes of this X-ray with these options
    assert stop["compare"] == {"model": D13, "report": alone["report"], "display_report": alone["display_report"],
                               "truncated_mid_sentence": alone["truncated_mid_sentence"]}
    assert _blocks(frames)[1][-2]["data"]["delta"]["text"] == stop["compare"]["report"] != stop["report"]
    # one labeller call, both reports in it, and the label detail says block 1's labels under compare
    assert labeler.calls == [[stop["report"], stop["compare"]["report"]], [alone["report"]]]   # the second is the partner's own turn
    label = _stage_end(frames, "label")["detail"]
    other = RuleLabeler().label([stop["compare"]["report"]])[0]
    assert label["compare"] == {"chexbert_14": dict(zip(CHEXBERT_14, other)), "positives": [n for n, v in zip(CHEXBERT_14, other) if v]}
    assert label["chexbert_14"] == dict(zip(CHEXBERT_14, RuleLabeler().label([stop["report"]])[0]))
    # the message row stays block 0's: block 1 lives in the event log
    assert stored["report"] == stop["report"] and stored["display_report"] == stop["display_report"]
    assert stored["provenance"]["name"] == M3 and stored["options"]["compare"] is True
    assert [(e["event"], e["data"]) for e in stored["events"]] == [(f["event"], f["data"]) for f in frames]


def test_a_non_compare_turn_is_the_turn_it_always_was_on_a_server_that_can_compare(tmp_path):
    """R2: a turn that does not compare streams, stores and exports exactly what it does on a server that serves no partner."""
    runs = {}
    for name, names in (("two", (M3, D13)), ("one", (M3,))):
        with TestClient(_app(tmp_path / name, names, labeler=RuleLabeler())) as c:
            sid = c.post("/v1/sessions", json={}).json()["id"]
            frames = _turn(c, sid, {"max_new_tokens": 16})
            explicit = _turn(c, sid, {"max_new_tokens": 16, "compare": False})
            mid = frames[0]["data"]["message_id"]
            both = dict(_ids(frames), **{explicit[0]["data"]["message_id"]: "<message2>", explicit[0]["data"]["user_message_id"]: "<user2>"})
            runs[name] = {
                "frames": _stable(frames, _ids(frames)), "explicit": _stable(explicit, _ids(explicit)),
                "message": _stable({k: v for k, v in c.get("/v1/messages/{}".format(mid)).json().items() if k != "events"}, both),
                "json": _stable(c.get("/v1/sessions/{}/export".format(sid)).json(), both),
                "md": _stable_md(c.get("/v1/sessions/{}/export?format=md".format(sid)).text, both),
            }
    two, one = runs["two"], runs["one"]
    assert two == one
    assert two["explicit"] == two["frames"]                                     # compare: false is no compare at all
    assert '"compare":' not in json.dumps([two["frames"], two["message"], two["json"]])   # keys only: the tiny vocabulary has "compared"
    assert {f["data"]["index"] for f in two["frames"] if f["event"].startswith("content_block_")} == {0}
    assert "Compare mode" not in two["md"]


R2_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "r2_turn_1f4fe6d.json"


def test_a_turn_that_does_not_compare_equals_the_turn_recorded_on_the_code_before_compare_mode_value_for_value(tmp_path):   # P9-F fix 1 (3a)
    """R2, by value: one private tiny turn with every stage (tests/r2_pin.py), recorded on an extracted copy of 1f4fe6d, the code before compare mode,
    and committed; the head's turn must be that turn: its SSE events, GET /v1/messages, both exports and /v1/models, with only ids, clocks, timings,
    the commit and the thread count blanked (one normaliser for both runs). The test above compares the head with itself; this compares it with
    the code it changed. Exactly, in the environment it was recorded in; in another it skips, naming both (fix 2: the test below)."""
    assert R2_FIXTURE.is_file(), "missing {}: record it on an extracted copy of 1f4fe6d with tests/r2_pin.py record_turn".format(R2_FIXTURE)
    text = R2_FIXTURE.read_text(encoding="utf-8")
    recorded = json.loads(text)
    assert stages_run(recorded) == STAGES                                       # every stage ran on the base: each is held to it
    assert not re.search(r'"/(Users|home|sc|tmp|var|private|opt)/', text)      # no path of the recording machine
    reason = skip_reason(recorded["environment"], environment())
    if reason is not None:
        pytest.skip(reason)
    assert record_turn(create_app, str(tmp_path / "home")) == recorded


def test_the_r2_pin_compares_in_the_environment_it_was_recorded_in_and_skips_in_any_other():   # P9-F fix 2
    """The tiny decoder's beam ties and the similarity floats can differ across torch builds and CPU architectures, so the pin holds where it
    was recorded: the same torch and CPU compare, exactly; another torch, or another CPU, skips, with a reason that names both environments."""
    here = {"torch": "2.11.0", "cpu": "arm"}
    assert skip_reason(here, dict(here)) is None                                 # the same environment: compare
    assert skip_reason(here, {"torch": "2.5.1", "cpu": "arm"}) == "R2 pin recorded on torch 2.11.0 / arm; this is torch 2.5.1 / arm"
    assert skip_reason(here, {"torch": "2.11.0", "cpu": "x86_64"}) == "R2 pin recorded on torch 2.11.0 / arm; this is torch 2.11.0 / x86_64"
    assert skip_reason(here, {"torch": "2.5.1", "cpu": "x86_64"}) == "R2 pin recorded on torch 2.11.0 / arm; this is torch 2.5.1 / x86_64"
    # Both read the way the cards read them: the running one as a card built here says it, the recorded one as the recorded turn's cards do.
    card = TinyEngine().card()
    assert environment() == {"torch": card["torch"], "cpu": card["cpu"]}
    recorded = json.loads(R2_FIXTURE.read_text(encoding="utf-8"))
    cards = [recorded["sse"][0]["data"]["model"], recorded["message"]["provenance"]] + recorded["models"]["models"]
    assert all(recorded["environment"] == {"torch": c["torch"], "cpu": c["cpu"]} for c in cards)


# ---- public mode: each new field by the rule of its block-0 counterpart (R1) --------------------------------------------------------

PUBLIC = {"Authorization": "Bearer t", "X-Client-Id": "a"}
CHECKPOINTS = {M3: "/sc/home/someone/outputs/m3_run/checkpoints/last.ckpt",   # made-up paths: a card's checkpoint holds the user name, which
               D13: "/sc/home/someone/outputs/d13_run/checkpoints/last.ckpt"}   # public mode cuts to the file name


def _planted(app):
    for name, path in CHECKPOINTS.items():
        app.state.engines[name]._card["checkpoint"] = path
    return app


def test_public_mode_redacts_every_new_field_by_the_rule_of_its_block_0_counterpart(tmp_path):
    with TestClient(_planted(_app(tmp_path / "private", labeler=RuleLabeler()))) as c:
        private = _turn(c, c.post("/v1/sessions", json={}).json()["id"], {"compare": True, "max_new_tokens": 16})
    with TestClient(_planted(_app(tmp_path / "public", labeler=RuleLabeler(), mode="public", token="t"))) as c:
        sid = c.post("/v1/sessions", json={}, headers=PUBLIC).json()["id"]
        public = _turn(c, sid, {"compare": True, "max_new_tokens": 16}, headers=PUBLIC)
        stored = c.get("/v1/messages/{}".format(public[0]["data"]["message_id"]), headers=PUBLIC).json()
    mine, theirs = private[0]["data"], public[0]["data"]
    # message_start.compare.card is cut as message_start.model is: the checkpoint path to its file name, every other key kept
    assert theirs["compare"] == {"model": D13, "card": redact_card(mine["compare"]["card"], "public")}
    assert theirs["model"] == redact_card(mine["model"], "public")
    assert theirs["model"]["test_model"] is True and theirs["compare"]["card"]["test_model"] is True   # fix 2: a test model, said in public too
    assert (mine["compare"]["card"]["checkpoint"], theirs["compare"]["card"]["checkpoint"]) == (CHECKPOINTS[D13], "last.ckpt")
    # the rest is model output, its labels and its stage facts, which public mode keeps for block 0: it keeps them for block 1 too
    for stage in ("encode", "generate", "label"):
        assert _stable(_stage_end(public, stage)["detail"]["compare"], {}) == _stable(_stage_end(private, stage)["detail"]["compare"], {}), stage
    assert "neighbor_agreement" not in _stage_end(public, "label")["detail"]                     # block 0's own public rule, unchanged
    deltas = lambda frames: [f["data"]["delta"] for f in _blocks(frames)[1] if f["event"] == "content_block_delta"]   # noqa: E731
    assert deltas(public) == deltas(private) and len(deltas(public)) == 16
    assert public[-1]["data"]["compare"] == private[-1]["data"]["compare"] and public[-1]["data"]["compare"]["report"]
    assert stored["provenance"]["checkpoint"] == "last.ckpt" and stored["provenance"]["name"] == M3   # the row is block 0's, cut as before


def test_redact_event_cuts_the_partners_card_as_the_turns_own_and_sweeps_r1_keys_wherever_they_sit():
    card = {"name": D13, "checkpoint": "/sc/home/someone/run/checkpoints/last.ckpt", "nested": {"checkpoint": "/sc/home/someone/x.ckpt"}}
    start = {"message_id": "m", "model": dict(card, name=M3), "options": {}, "image": None, "compare": {"model": D13, "card": card}}
    out = redact_event("message_start", start, "public")
    assert out["compare"] == {"model": D13, "card": redact_card(card, "public")}
    assert out["compare"]["card"]["checkpoint"] == "last.ckpt" and out["compare"]["card"]["nested"]["checkpoint"] == "x.ckpt"
    assert redact_event("message_start", start, "private") == start and start["compare"]["card"] is card   # private: as it was, input untouched
    planted = {"stage": "label", "ms": 1.0, "detail": {"chexbert_14": {}, "positives": [],
                                                       "compare": {"chexbert_14": {}, "positives": [], "neighbor_agreement": [], "reference": "x"}}}
    assert redact_event("stage_end", planted, "public")["detail"]["compare"] == {"chexbert_14": {}, "positives": []}   # the catch-all, at depth


# ---- cancel, in either decode ------------------------------------------------------------------------------------------------------

@pytest.fixture
def live(tmp_path):
    base, stop = start_live_server(_app(tmp_path, tiny_step_delay_s=0.02, labeler=RuleLabeler()))
    yield base
    stop()


def _cancelled_in_block(base, block):
    """A compare turn of 150 tokens a block, cancelled at block `block`'s first snapshot: -> (its frames, the cancel's status, the stored turn)."""
    sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]
    frames, cancels = [], []
    with httpx.stream("POST", base + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"compare": True, "max_new_tokens": 150})}, timeout=60) as r:
        for frame in iter_sse(r.iter_text()):
            frames.append(frame)
            if frame["event"] == "content_block_delta" and frame["data"]["index"] == block and not cancels:
                cancels.append(httpx.post(base + "/v1/messages/{}/cancel".format(frames[0]["data"]["message_id"])).status_code)
    return frames, cancels, httpx.get(base + "/v1/messages/{}".format(frames[0]["data"]["message_id"])).json()


NOT_ENDED = {"model": D13, "report": None, "display_report": None, "truncated_mid_sentence": False}


def _compare_section(md):
    """The compare lines of a Markdown export's only turn: from "Compare mode" to the provenance line."""
    return "Compare mode" + md.split("Compare mode", 1)[1].split("\n\nProvenance:", 1)[0]


def test_cancel_during_block_0_stops_its_decode_and_block_1_never_starts(live):
    frames, cancels, stored = _cancelled_in_block(live, 0)
    assert cancels == [200] and frames[-1]["event"] == "message_stop"
    stop = frames[-1]["data"]
    assert (stop["status"], stop["report"], stop["compare"]) == ("aborted", None, NOT_ENDED)   # aborted as today, block 1 with it
    blocks = _blocks(frames)
    assert list(blocks) == [0] and len(blocks[0]) - 1 < 150 and blocks[0][-1]["event"] == "content_block_delta"   # cut mid-decode
    assert not any(f["event"] == "stage_end" and f["data"]["stage"] in ("generate", "label") for f in frames)
    assert stored["status"] == "aborted" and stored["report"] is None
    assert [(e["event"], e["data"]) for e in stored["events"]] == [(f["event"], f["data"]) for f in frames]   # what streamed is kept
    md = httpx.get(live + "/v1/sessions/{}/export?format=md".format(frames[0]["data"]["session_id"])).text
    assert _compare_section(md) == "Compare mode: the turn ended before {} decoded.".format(D13)   # fix 1: only what happened


def test_cancel_during_block_1_stops_its_decode_after_block_0_has_ended(live):
    frames, cancels, stored = _cancelled_in_block(live, 1)
    assert cancels == [200]
    stop = frames[-1]["data"]
    assert (stop["status"], stop["report"], stop["compare"]) == ("aborted", None, NOT_ENDED)   # a turn that did not finish: neither report
    blocks = _blocks(frames)
    assert [f["event"] for f in blocks[0]] == ["content_block_start"] + ["content_block_delta"] * 150 + ["content_block_stop"]
    assert blocks[1][0]["event"] == "content_block_start" and len(blocks[1]) - 1 < 150 and blocks[1][-1]["event"] == "content_block_delta"
    assert not any(f["event"] == "stage_end" and f["data"]["stage"] in ("generate", "label") for f in frames)
    assert stored["status"] == "aborted" and [(e["event"], e["data"]) for e in stored["events"]] == [(f["event"], f["data"]) for f in frames]
    md = httpx.get(live + "/v1/sessions/{}/export?format=md".format(frames[0]["data"]["session_id"])).text
    # block 1 started, and the turn did not finish: block 0's rule, no report, said as what happened
    assert _compare_section(md) == "Compare mode: {} began decoding the same X-ray too, and the turn was stopped.".format(D13)


class HeldLabeler(RuleLabeler):
    """The keyword labeller, which holds its call until release is set: a test presses Stop while the label stage runs."""

    def __init__(self):
        self.release = threading.Event()

    def label(self, texts):
        assert self.release.wait(30), "the test never released the labeller"
        return super().label(texts)


def _label_lines(turn_md):
    """A turn's label lines in its Markdown export: block 0's table, with its heading, and block 1's positives line."""
    return [line for line in turn_md.split("\n") if line.startswith(("CheXbert-14 labels", "| ", "Positive CheXbert-14 labels"))]


def test_a_compare_turn_stopped_after_its_label_stage_exports_block_1s_positives_as_block_0s_labels_are(tmp_path):   # P9-F fix 2
    """Stop pressed while the label stage runs: the stage ends with both blocks' labels, and the turn ends aborted at the next stage's check (a
    reference, so that the score stage runs and checks). Block 0's label table is exported whatever the status, so block 1's positives are too,
    each in the very lines a finished turn exports (the same turn run to the end next), and the status said as block 0's rule says it."""
    labeler = HeldLabeler()
    base, stop_server = start_live_server(_app(tmp_path, labeler=labeler))
    options = {"compare": True, "max_new_tokens": 16, "reference": "Findings: Small left pleural effusion. Impression: Effusion."}
    try:
        sid = httpx.post(base + "/v1/sessions", json={}).json()["id"]
        frames, cancels = [], []
        with httpx.stream("POST", base + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                          data={"text": "", "options": json.dumps(options)}, timeout=60) as r:
            for frame in iter_sse(r.iter_text()):
                frames.append(frame)
                if frame["event"] == "stage_start" and frame["data"]["stage"] == "label":
                    cancels.append(httpx.post(base + "/v1/messages/{}/cancel".format(frames[0]["data"]["message_id"])).status_code)
                    labeler.release.set()
        finished = httpx.post(base + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                              data={"text": "", "options": json.dumps(options)}, timeout=60)
        md = httpx.get(base + "/v1/sessions/{}/export?format=md".format(sid)).text
    finally:
        labeler.release.set()
        stop_server()
    assert cancels == [200] and finished.status_code == 200
    stop = frames[-1]["data"]
    assert (frames[-1]["event"], stop["status"], stop["report"], stop["compare"]) == ("message_stop", "aborted", None, NOT_ENDED)
    label = _stage_end(frames, "label")["detail"]                                # the label stage ended, with block 1's labels
    assert [f["data"]["stage"] for f in frames if f["event"] == "stage_end"] == STAGES[:-1]   # and the score stage never started
    stopped, done = md.split("\n## Turn ")[1:]
    assert "\n\nStatus: aborted\n\n" in stopped and "\n\nStatus: " not in done   # block 0's status line; the turn run to the end has none
    positives = "Positive CheXbert-14 labels ({}): {}".format(D13, ", ".join(label["compare"]["positives"]) or "none")
    assert _compare_section(stopped) == ("Compare mode: {} began decoding the same X-ray too, and the turn was stopped.\n\n".format(D13)
                                         + positives)
    assert _label_lines(stopped) == _label_lines(done)                          # both blocks' labels, as the finished turn exports them
    assert len(_label_lines(stopped)) == 1 + 2 + 14 + 1 and _label_lines(stopped)[-1] == positives   # heading, table head, 14 rows, positives


def test_a_question_turn_with_compare_runs_no_model_streams_what_a_question_without_it_does_and_exports_no_compare_section(tmp_path):
    """Fix 1 (review Important 1): text with no image that is no command gets the fixed answer and runs no model. With compare on, it must not
    say that two models ran: its stream is a question's, but for the options it was asked with, and its Markdown has no compare section."""
    with TestClient(_app(tmp_path, labeler=RuleLabeler())) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        _turn(c, sid, {"compare": True, "max_new_tokens": 16})                   # an image first: the session has one, so text alone is accepted
        asked = _turn(c, sid, {"compare": True, "max_new_tokens": 16}, image=False, text="is this pneumonia?")
        plain = _turn(c, sid, {"max_new_tokens": 16}, image=False, text="is this pneumonia?")
        md = c.get("/v1/sessions/{}/export?format=md".format(sid)).text
    assert [f["event"] for f in asked] == ["message_start", "warning", "message_stop"]
    assert "compare" not in asked[0]["data"] and "compare" not in asked[-1]["data"]
    assert asked[0]["data"]["options"]["compare"] is True                        # what the turn was asked with, as every option is said
    stable = _stable(asked, _ids(asked))
    del stable[0]["data"]["options"]["compare"]
    assert stable == _stable(plain, _ids(plain))                                 # and nothing else differs from a question's
    turns = md.split("\n## Turn ")
    assert len(turns) == 4 and "Compare mode" in turns[1] and "Compare mode" not in turns[2] + turns[3]


def _failing_partner(app, steps=3):
    """The partner's decode writes `steps` snapshots and then raises, with a text that must reach no event (it could hold report text)."""
    def generate(enc, opts, on_snapshot, cancel):
        for step in range(steps):
            on_snapshot(step, " ".join(["edema"] * (step + 1)))
        raise RuntimeError("SECRET-PARTNER-TEXT")

    app.state.engines[D13].generate = generate
    return app


def test_an_error_in_block_1_ends_the_turn_as_an_error_with_neither_report_and_keeps_what_streamed(tmp_path):   # P9-F fix 1 (3b)
    for mode, headers, message in (("private", None, "Internal error (RuntimeError)"), ("public", PUBLIC, PUBLIC_ERROR_MESSAGE)):
        kw = {"mode": "public", "token": "t"} if mode == "public" else {}
        with TestClient(_failing_partner(_app(tmp_path / mode, labeler=RuleLabeler(), **kw))) as c:
            sid = c.post("/v1/sessions", json={}, headers=headers).json()["id"]
            frames = _turn(c, sid, {"compare": True, "max_new_tokens": 16}, headers=headers)
            stored = c.get("/v1/messages/{}".format(frames[0]["data"]["message_id"]), headers=headers).json()
        blocks = _blocks(frames)
        assert [f["event"] for f in blocks[0]] == ["content_block_start"] + ["content_block_delta"] * 16 + ["content_block_stop"], mode
        assert [f["event"] for f in blocks[1]] == ["content_block_start"] + ["content_block_delta"] * 3, mode   # cut where it failed
        [error] = [f["data"] for f in frames if f["event"] == "error"]
        assert error["error"] == {"type": "model_error", "message": message}, mode
        assert [f["event"] for f in frames[-2:]] == ["error", "message_stop"] and not any(
            f["event"] == "stage_end" and f["data"]["stage"] in ("generate", "label") for f in frames), mode
        stop = frames[-1]["data"]
        assert (stop["status"], stop["report"], stop["display_report"], stop["compare"]) == ("error", None, None, NOT_ENDED), mode
        assert "SECRET-PARTNER-TEXT" not in json.dumps(frames) + json.dumps(stored), mode
        assert stored["status"] == "error" and stored["report"] is None
        assert [(e["event"], e["data"]) for e in stored["events"]] == [(f["event"], f["data"]) for f in frames], mode   # stored as streamed


# ---- replay: the session, the poll path and the exports --------------------------------------------------------------------------

def test_a_dropped_compare_stream_is_picked_up_by_polling_and_the_log_replays_the_whole_turn(live):
    sid = httpx.post(live + "/v1/sessions", json={}).json()["id"]
    streamed = []
    with httpx.stream("POST", live + "/v1/sessions/{}/messages".format(sid), files={"image": ("x.png", png_bytes(), "image/png")},
                      data={"text": "", "options": json.dumps({"compare": True, "max_new_tokens": 40})}, timeout=60) as r:
        for frame in iter_sse(r.iter_text()):
            streamed.append(frame)
            if frame["event"] == "content_block_delta" and frame["data"]["index"] == 1:
                break                                                      # the tunnel drops in block 1
    mid = streamed[0]["data"]["message_id"]
    polled, status = [], "running"
    for _ in range(600):                                                   # as the page polls: ?after=<the last seq it has>
        message = httpx.get(live + "/v1/messages/{}?after={}".format(mid, (polled or streamed)[-1]["data"]["seq"])).json()
        polled += [{"event": e["event"], "data": e["data"]} for e in message["events"]]
        status = message["status"]
        if status != "running":
            break
        threading.Event().wait(0.05)
    whole = streamed + polled
    assert status == "done" and [f["data"]["seq"] for f in whole] == list(range(1, len(whole) + 1))
    assert whole[-1]["event"] == "message_stop" and whole[-1]["data"]["compare"]["report"]
    assert [f["event"] for f in _blocks(whole)[1]] == ["content_block_start"] + ["content_block_delta"] * 40 + ["content_block_stop"]
    stored = httpx.get(live + "/v1/messages/{}".format(mid)).json()["events"]
    assert [(e["event"], e["data"]) for e in stored] == [(f["event"], f["data"]) for f in whole]
    session = httpx.get(live + "/v1/sessions/{}".format(sid)).json()
    user, assistant = session["messages"]
    assert user["image_urls"]["thumb"].endswith("variant=thumb") and assistant["id"] == mid
    assert assistant["report"] == whole[-1]["data"]["report"] and assistant["provenance"]["name"] == M3   # the row: block 0's


def test_both_exports_carry_block_1(tmp_path):
    with TestClient(_app(tmp_path, labeler=RuleLabeler())) as c:
        sid = c.post("/v1/sessions", json={}).json()["id"]
        frames = _turn(c, sid, {"compare": True, "max_new_tokens": 16, "display_repair": True})
        exported = c.get("/v1/sessions/{}/export".format(sid)).json()
        md = c.get("/v1/sessions/{}/export?format=md".format(sid)).text
    events = exported["messages"][1]["events"]
    assert [(e["event"], e["data"]) for e in events] == [(f["event"], f["data"]) for f in frames]   # the JSON export is the event log
    stop = frames[-1]["data"]["compare"]
    positives = _stage_end(frames, "label")["detail"]["compare"]["positives"]
    section = md.split("Compare mode", 1)[1].split("Provenance:", 1)[0]
    assert section.startswith(": the same X-ray, decoded by {} too.".format(D13))
    assert "Report ({}):\n\n> {}".format(D13, stop["report"]) in section
    assert ("Display report ({}, sentence repaired):".format(D13) in section) == (stop["display_report"] != stop["report"])
    assert "Positive CheXbert-14 labels ({}): {}".format(D13, ", ".join(positives) or "none") in section
    assert md.index("Report:") < md.index("Compare mode") < md.index("Provenance:")   # after block 0's report and labels, in its turn
    assert md.count("Compare mode") == 1


def test_a_compare_turn_that_ended_before_its_partner_started_exports_one_line_saying_so(tmp_path):
    """Fix 1 (the review's Markdown Minor): a compare turn that failed or was stopped before block 1 started (no content_block_start of index 1 in
    its log) never decoded with the partner, and says only that."""
    app = _app(tmp_path, labeler=RuleLabeler())
    with TestClient(app) as c:
        for status in ("aborted", "error"):
            sid = c.post("/v1/sessions", json={}).json()["id"]
            user_id, message_id = app.state.store.start_turn(sid, "", "private", {"compare": True}, "ab" * 32, "x.png")
            for event, data in (("message_start", {"message_id": message_id, "compare": {"model": D13, "card": {"name": D13}}}),
                                ("message_stop", {"message_id": message_id, "status": status, "report": None, "compare": NOT_ENDED})):
                app.state.store.append_event(message_id, event, data)
            app.state.store.finish_turn(message_id, status)
            md = c.get("/v1/sessions/{}/export?format=md".format(sid)).text
            assert _compare_section(md) == "Compare mode: the turn ended before {} decoded.".format(D13), status


def test_the_compare_lines_of_an_export_say_what_happened_in_every_case():   # P9-F fix 1, fix 2
    part = {"model": D13, "started": True, "report": "Findings: Small effusion.", "display_report": "Findings: Small effusion.",
            "positives": ["Pleural Effusion"]}
    lines = lambda status, **change: "\n".join(_compare_lines(dict(part, **change), status)).strip()   # noqa: E731
    assert lines("done") == ("Compare mode: the same X-ray, decoded by {0} too.\n\nReport ({0}):\n\n> Findings: Small effusion.\n\n"
                             "Positive CheXbert-14 labels ({0}): Pleural Effusion").format(D13)
    assert lines("done", positives=[]).endswith("Positive CheXbert-14 labels ({}): none".format(D13))
    # Block 1 started and the turn did not finish: the status in words, and then what block 0's part shows of such a turn (fix 2). It has no
    # report (a turn that did not finish has neither), and block 0's label table is printed whatever the status, so block 1's positives are too,
    # once the label stage has run; before it, the status line alone, as fix 1 had it.
    for status, words in (("aborted", "the turn was stopped"), ("error", "the turn ended in an error"), ("running", "the turn is still running")):
        said = "Compare mode: {} began decoding the same X-ray too, and {}.".format(D13, words)
        assert lines(status, report=None) == said + "\n\nPositive CheXbert-14 labels ({}): Pleural Effusion".format(D13), status
        assert lines(status, report=None, positives=[]) == said + "\n\nPositive CheXbert-14 labels ({}): none".format(D13), status
        assert lines(status, report=None, positives=None) == said, status
    for status in ("aborted", "error"):
        assert lines(status, started=False, report=None) == "Compare mode: the turn ended before {} decoded.".format(D13)
    assert lines("running", started=False, report=None) == ("Compare mode: {} decodes the same X-ray once the first report is done; the turn is "
                                                            "still running.".format(D13))


# ---- the reference (OpenAPI) -------------------------------------------------------------------------------------------------------

def test_the_reference_documents_the_option_its_refusal_its_second_block_and_the_models_fields(tmp_path):
    with TestClient(create_app(engine="tiny", home=str(tmp_path))) as c:
        spec = c.get("/openapi.json").json()
    turn = spec["paths"]["/v1/sessions/{session_id}/messages"]["post"]
    ref = turn["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].rsplit("/", 1)[1]
    options = spec["components"]["schemas"][ref]["properties"]["options"]["description"]
    about = options.split("`compare`", 1)[1]
    assert "13D" in about and "Mamba-3" in about and "index" in about and NO_COMPARE_MSG in about
    assert "`compare`" in turn["responses"]["200"]["description"] and "`index` 1" in turn["responses"]["200"]["description"]
    models = spec["paths"]["/v1/models"]["get"]["description"]
    assert "`features.compare`" in models and "`compare_with`" in models
