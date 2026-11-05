"""The parametric .nam layout (parametric_nam/format.py): numpy only."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from parametric_nam.format import (ARCHITECTURE, expected_weight_count, nam_weights_to_state,  # noqa: E402
                                 state_to_parametric_nam, strip_state_prefix)

PLAIN = pathlib.Path(__file__).resolve().parents[1] / "fixtures/plain.nam"
KNOBS = [{"name": "Gain"}, {"name": "Bass", "default": 0.25}, {"name": "Treble"}]


@pytest.fixture
def plain():
    return json.loads(PLAIN.read_text())


def test_plain_weights_walk_exactly(plain):
    state, head_scale = nam_weights_to_state(plain["config"]["layers"], plain["weights"])
    assert head_scale == plain["weights"][-1]
    assert state["_blocks.0._rechannel.weight"].shape == (3, 1, 1)
    assert state["_blocks.0._layers.1._conv.weight"].shape == (3, 3, 3)
    assert state["_blocks.1._head_rechannel.bias"].shape == (1,)
    assert "_blocks.0._head_rechannel.bias" not in state  # head_bias false on the first block
    total = sum(v.size for v in state.values()) + 1
    assert total == len(plain["weights"])


def test_truncated_weights_are_refused(plain):
    with pytest.raises(ValueError):
        nam_weights_to_state(plain["config"]["layers"], plain["weights"][:-5])
    with pytest.raises(ValueError):
        nam_weights_to_state(plain["config"]["layers"], plain["weights"] + [0.0])


def test_widened_layout(plain):
    layers = plain["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain["weights"])
    out = state_to_parametric_nam(state, layers, hs, KNOBS, metadata={"name": "x"})
    assert out["architecture"] == ARCHITECTURE
    assert [k["name"] for k in out["config"]["knobs"]] == ["Gain", "Bass", "Treble"]
    assert out["config"]["knobs"][1]["default"] == 0.25
    assert out["config"]["layers"][0]["input_size"] == 4
    assert all(lc["condition_size"] == 4 for lc in out["config"]["layers"])
    assert out["config"]["layers"][1]["input_size"] == 3  # later blocks take the previous block's channels
    assert len(out["weights"]) == expected_weight_count(layers, 3)
    assert out["weights"][-1] == hs
    assert out["sample_rate"] == 48000.0
    # the original layers are untouched
    assert layers[0]["condition_size"] == 1


def test_zero_knob_weights_keep_every_plain_weight(plain):
    layers = plain["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain["weights"])
    out = state_to_parametric_nam(state, layers, hs, KNOBS)
    # walking the widened file back as if it were plain, with the widened layer configs,
    # must give the same tensors plus zero knob columns
    back, hs2 = nam_weights_to_state(out["config"]["layers"], out["weights"])
    assert hs2 == hs
    rech = back["_blocks.0._rechannel.weight"]
    assert np.array_equal(rech[:, :1], state["_blocks.0._rechannel.weight"])
    assert not rech[:, 1:].any()
    for k, v in state.items():
        if k.endswith("_input_mixer.weight"):
            w = back[k]
            assert np.array_equal(w[:, :1], v)
            assert not w[:, 1:].any()
        elif "_rechannel.weight" in k and k.startswith("_blocks.0."):
            continue
        else:
            assert np.array_equal(back[k], v), k


def test_global_mixer_is_folded(plain):
    layers = plain["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain["weights"])
    p = "_blocks.0._layers.0."
    gw = np.arange(9, dtype=float).reshape(3, 3)
    gb = np.array([1.0, 2.0, 3.0])
    state[p + "_global_mixer.weight"] = gw
    state[p + "_global_mixer.bias"] = gb
    out = state_to_parametric_nam(state, layers, hs, KNOBS)
    back, _ = nam_weights_to_state(out["config"]["layers"], out["weights"])
    assert np.array_equal(back[p + "_input_mixer.weight"][:, 1:, 0], gw)
    assert np.array_equal(back[p + "_conv.bias"], state[p + "_conv.bias"] + gb)


def test_knob_count_must_match_the_mixer(plain):
    layers = plain["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain["weights"])
    state["_blocks.0._layers.0._global_mixer.weight"] = np.zeros((3, 2))
    with pytest.raises(ValueError):
        state_to_parametric_nam(state, layers, hs, KNOBS)


def test_knob_validation(plain):
    layers = plain["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain["weights"])
    with pytest.raises(ValueError):
        state_to_parametric_nam(state, layers, hs, [])
    with pytest.raises(ValueError):
        state_to_parametric_nam(state, layers, hs, [{"name": f"k{i}"} for i in range(9)])
    with pytest.raises(ValueError):
        state_to_parametric_nam(state, layers, hs, [{"default": 0.5}])


def test_strip_state_prefix():
    st = {"_net._net._blocks.0._layers.0._conv.weight": np.ones((2, 2, 3)), "_net._sample_rate": np.array(48000.0)}
    out = strip_state_prefix(st)
    assert list(out) == ["_blocks.0._layers.0._conv.weight"]
    with pytest.raises(ValueError):
        strip_state_prefix({"foo": np.zeros(1)})


def test_switch_labels_are_written_and_checked():
    plain = json.loads(PLAIN.read_text())
    layers = plain["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain["weights"])
    knobs = [{"name": "Gain"}, {"name": "Power", "default": 1.0, "labels": ["half", "full"]}]
    out = state_to_parametric_nam(state, layers, hs, knobs)
    assert out["config"]["knobs"][1] == {"name": "Power", "default": 1.0, "labels": ["half", "full"]}
    assert "labels" not in out["config"]["knobs"][0]
    with pytest.raises(ValueError):
        state_to_parametric_nam(state, layers, hs, [{"name": "Power", "labels": ["only"]}])


def test_panel_metadata_is_written_and_validated(plain):
    from parametric_nam.format import LOOKS, panel_metadata
    layers = plain["config"]["layers"]
    state, hs = nam_weights_to_state(layers, plain["weights"])
    out = state_to_parametric_nam(state, layers, hs, [{"name": "Gain"}],
                                  metadata={"panel": panel_metadata("tweed", {"Gain": "VOLUME"}),
                                            "modeled_by": "Jane", "author_note": "65 reissue"})
    assert out["metadata"]["panel"] == {"look": "tweed", "labels": {"Gain": "VOLUME"}}
    assert out["metadata"]["modeled_by"] == "Jane"
    assert out["metadata"]["author_note"] == "65 reissue"
    assert panel_metadata(None, None) == {}
    assert panel_metadata("generic", None) == {"look": "generic"}
    assert LOOKS == ("generic", "blackface", "tweed", "vox", "mesa", "plexi")
    with pytest.raises(ValueError):
        panel_metadata("orange", None)
    with pytest.raises(ValueError):
        state_to_parametric_nam(state, layers, hs, [{"name": "Gain"}], metadata={"panel": {"look": "orange"}})
