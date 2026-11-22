"""parametric_nam/snapshot.py: one knob setting of a parametric file as a plain NAM WaveNet."""
import copy
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from parametric_nam.snapshot import snapshot  # noqa: E402
from ref_wavenet import play  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
SETTINGS = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.9, 0.1, 0.7], [1.0, 1.0, 1.0]]


def _scaled(model, seed=0, amount=0.5):
    """The fixture plus a random knob_scale, as a knob-scaling trainer would write it."""
    m = copy.deepcopy(model)
    rng = np.random.default_rng(seed)
    N = len(m["config"]["knobs"])
    m["knob_scale"] = [[list(rng.normal(0, amount, L["channels"] * N)) for _ in L["dilations"]]
                       for L in m["config"]["layers"]]
    return m


@pytest.fixture
def model():
    return json.loads((FIX / "widened_rand.nam").read_text())


@pytest.fixture
def audio():
    return np.random.default_rng(1).normal(0, 0.3, 2000)


def _esr(a, b):
    return float(np.sum((a - b) ** 2) / np.sum(b ** 2))


@pytest.mark.parametrize("scaled", [False, True])
def test_a_snapshot_sounds_like_the_parametric_file_at_that_setting(model, audio, scaled):
    m = _scaled(model) if scaled else model
    for g in SETTINGS:
        snap = snapshot(m, g)
        assert snap["architecture"] == "WaveNet" and "knob_scale" not in snap
        assert all(L["condition_size"] == 1 for L in snap["config"]["layers"])
        assert snap["config"]["layers"][0]["input_size"] == 1
        assert _esr(play(snap, audio), play(m, audio, g)) < 1e-20


def test_knob_scale_changes_the_sound(model, audio):
    g = [0.9, 0.1, 0.7]
    assert _esr(play(_scaled(model), audio, g), play(model, audio, g)) > 1e-3


def test_gain_and_invert_go_into_the_head_scale(model, audio):
    g = [0.5, 0.5, 0.5]
    base = play(snapshot(model, g), audio)
    louder = play(snapshot(model, g, gain_db=6.0, invert=True), audio)
    assert np.allclose(louder, -base * 10 ** (6 / 20))


def test_wrong_knob_count_and_plain_files_are_refused(model):
    with pytest.raises(ValueError):
        snapshot(model, [0.5])
    with pytest.raises(ValueError):
        snapshot(json.loads((FIX / "plain.nam").read_text()), [])


def test_snapshot_of_a_bad_knob_scale_is_refused(model):
    m = _scaled(model)
    m["knob_scale"][0][0] = m["knob_scale"][0][0][:-1]
    with pytest.raises(ValueError):
        snapshot(m, [0.5, 0.5, 0.5])
