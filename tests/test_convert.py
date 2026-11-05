"""convert_panama.py: knobs, switches, look, labels and author from the plan and flags."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from parametric_nam.convert_panama import ask_look, main  # noqa: E402
from parametric_nam.format import LOOKS, nam_weights_to_state  # noqa: E402

PLAIN = pathlib.Path(__file__).resolve().parents[1] / "fixtures/plain.nam"


def _session(tmp_path):
    torch = pytest.importorskip("torch")
    plain = json.loads(PLAIN.read_text())
    layers = plain["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain["weights"])
    ckpt = tmp_path / "last.ckpt"
    torch.save({"state_dict": {"_net." + k: torch.tensor(v) for k, v in state.items()}}, ckpt)
    cfg = tmp_path / "model.json"
    cfg.write_text(json.dumps({"net": {"config": {"layers_configs": layers, "head_scale": float(hs)}}}))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"name": "Deluxe", "knobs": ["Volume", "Bright"],
                                "switches": {"Bright": ["off", "on"]}, "look": "blackface",
                                "labels": {"Volume": "VOL"}, "author": "Jane"}))
    return ckpt, cfg, plan


def test_convert_writes_panel_from_plan_and_flags_override(tmp_path):
    ckpt, cfg, plan = _session(tmp_path)
    out = tmp_path / "amp.nam"
    assert main(["--ckpt", str(ckpt), "--model-config", str(cfg), "--plan", str(plan), "--out", str(out),
                 "--label", "Volume=VOLUME", "--note", "hi"]) == 0
    model = json.loads(out.read_text())
    md = model["metadata"]
    assert md["panel"] == {"look": "blackface", "labels": {"Volume": "VOLUME"}}
    assert md["modeled_by"] == "Jane"
    assert md["author_note"] == "hi"
    assert md["name"] == "Deluxe"
    assert model["config"]["knobs"][1]["labels"] == ["off", "on"]


def test_convert_without_a_look_writes_generic_when_not_interactive(tmp_path, monkeypatch):
    ckpt, cfg, _ = _session(tmp_path)
    out = tmp_path / "amp.nam"
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    assert main(["--ckpt", str(ckpt), "--model-config", str(cfg), "--knobs", "Gain", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["metadata"]["panel"] == {"look": "generic"}


def test_convert_refuses_an_unknown_look(tmp_path):
    ckpt, cfg, _ = _session(tmp_path)
    with pytest.raises(SystemExit):
        main(["--ckpt", str(ckpt), "--model-config", str(cfg), "--knobs", "Gain", "--look", "orange",
              "--out", str(tmp_path / "x.nam")])


def test_ask_look_defaults_to_generic():
    assert ask_look(input=lambda _: "") == "generic"
    assert ask_look(input=lambda _: "2") == LOOKS[1]
    assert ask_look(input=lambda _: "tweed") == "tweed"
