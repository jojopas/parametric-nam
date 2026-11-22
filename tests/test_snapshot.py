"""parametric_nam/snapshot.py: one knob setting of a parametric file as a plain NAM WaveNet,
and knob_scale written by format.py."""
import copy
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from parametric_nam.format import nam_weights_to_state, state_to_parametric_nam  # noqa: E402
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


def _panama_state(plain_model, knobs, scale):
    """A PANAMA-shaped state from the plain fixture: global mixer (and optionally scale) per layer."""
    layers = plain_model["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain_model["weights"])
    rng = np.random.default_rng(2)
    for b, lc in enumerate(layers):
        for l, _ in enumerate(lc["dilations"]):
            p = f"_blocks.{b}._layers.{l}."
            state[p + "_global_mixer.weight"] = rng.normal(0, 0.3, (lc["channels"], knobs))
            state[p + "_global_mixer.bias"] = rng.normal(0, 0.1, lc["channels"])
            if scale:
                state[p + "_global_scale.weight"] = rng.normal(0, 0.3, (lc["channels"], knobs))
    return state, hs, layers


def test_format_writes_knob_scale_from_global_scale_tensors(audio):
    plain = json.loads((FIX / "plain.nam").read_text())
    state, hs, layers = _panama_state(plain, 2, scale=True)
    m = state_to_parametric_nam(state, layers, hs, [{"name": "Gain"}, {"name": "Tone"}])
    ks = m["knob_scale"]
    assert [len(a) for a in ks] == [len(L["dilations"]) for L in layers]
    assert ks[0][1] == list(state["_blocks.0._layers.1._global_scale.weight"].ravel())
    # and a PANAMA-style forward of that state agrees with the file
    g = np.array([0.3, 0.8])
    x = audio
    ref = _panama_forward(state, layers, hs, x, g)
    assert _esr(play(m, x, g), ref) < 1e-20


def test_format_without_global_scale_writes_no_knob_scale():
    plain = json.loads((FIX / "plain.nam").read_text())
    state, hs, layers = _panama_state(plain, 2, scale=False)
    assert "knob_scale" not in state_to_parametric_nam(state, layers, hs, [{"name": "Gain"}, {"name": "Tone"}])


def test_format_refuses_knob_scale_on_only_some_layers():
    plain = json.loads((FIX / "plain.nam").read_text())
    state, hs, layers = _panama_state(plain, 2, scale=True)
    del state["_blocks.1._layers.0._global_scale.weight"]
    with pytest.raises(ValueError):
        state_to_parametric_nam(state, layers, hs, [{"name": "Gain"}, {"name": "Tone"}])


def _panama_forward(state, layers, hs, x, g):
    """PANAMA's layer with knob scaling, from named tensors:
    z = (conv(x) + global_mixer(g)) * (1 + global_scale(g)) + input_mixer(audio)."""
    x = np.asarray(x, dtype=np.float64)[None, :]
    y, head = x, None
    for b, lc in enumerate(layers):
        f = {"Tanh": np.tanh}[lc["activation"]]
        xa = state[f"_blocks.{b}._rechannel.weight"][:, :, 0] @ y
        for l, d in enumerate(lc["dilations"]):
            p = f"_blocks.{b}._layers.{l}."
            W, bb = state[p + "_conv.weight"], state[p + "_conv.bias"]
            K = W.shape[2]
            T = xa.shape[1] - (K - 1) * d
            zc = sum(W[:, :, k] @ xa[:, k * d:k * d + T] for k in range(K)) + bb[:, None]
            zg = state[p + "_global_mixer.weight"] @ g + state[p + "_global_mixer.bias"]
            s = 1 + state[p + "_global_scale.weight"] @ g
            z = (zc + zg[:, None]) * s[:, None] + state[p + "_input_mixer.weight"][:, :, 0] @ x[:, -T:]
            post = f(z)
            xa = xa[:, -T:] + state[p + "_1x1.weight"][:, :, 0] @ post + state[p + "_1x1.bias"][:, None]
            head = post if head is None else head[:, -T:] + post
        hb = state.get(f"_blocks.{b}._head_rechannel.bias")
        head = state[f"_blocks.{b}._head_rechannel.weight"][:, :, 0] @ head + (0 if hb is None else hb[:, None])
        y = xa
    return hs * head[0]
