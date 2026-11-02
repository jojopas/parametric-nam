"""Regenerate fixtures/.

  make_fixtures.py            widened_zero.nam / widened_rand.nam from plain.nam (numpy only)
  make_fixtures.py --panama   also panama_demo.* from the PANAMA demo checkpoint
                              (needs PANAMA_DIR and its environment; renders the
                              PyTorch reference a C++ host can null-test against)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from parametric_nam.format import nam_weights_to_state, state_to_parametric_nam  # noqa: E402

import os

FIX = REPO / "fixtures"
PANAMA = pathlib.Path(os.environ.get("PANAMA_DIR", REPO.parent / "PANAMA"))
KNOBS3 = [{"name": "Gain", "default": 0.5}, {"name": "Bass", "default": 0.5}, {"name": "Treble", "default": 0.5}]


def widened() -> None:
    plain = json.loads((FIX / "plain.nam").read_text())
    layers = plain["config"]["layers"]
    state, head_scale = nam_weights_to_state(layers, plain["weights"])
    md = {"name": "widened test model", "loudness": plain["metadata"].get("loudness")}
    zero = state_to_parametric_nam(state, layers, head_scale, KNOBS3, metadata=md)
    (FIX / "widened_zero.nam").write_text(json.dumps(zero))

    rng = np.random.default_rng(7)
    rand = dict(state)
    for b, lc in enumerate(layers):
        mid = 2 * lc["channels"] if lc.get("gated", False) else lc["channels"]
        for l in range(len(lc["dilations"])):
            p = f"_blocks.{b}._layers.{l}."
            rand[p + "_global_mixer.weight"] = rng.normal(0, 0.5, (mid, len(KNOBS3)))
            rand[p + "_global_mixer.bias"] = rng.normal(0, 0.1, (mid,))
    (FIX / "widened_rand.nam").write_text(json.dumps(state_to_parametric_nam(rand, layers, head_scale, KNOBS3, metadata=md)))
    print("wrote widened_zero.nam widened_rand.nam")


def panama_demo() -> None:
    import soundfile as sf
    import torch

    sys.path.insert(0, str(PANAMA))
    from nam.train.lightning_module import LightningModule  # PANAMA's fork

    from parametric_nam.convert_panama import load_ckpt_state, panama_layers

    cfg_path = PANAMA / "default_config_files/models/wavenet.json"
    model_config = json.loads(cfg_path.read_text())
    knobs = [{"name": f"K{i}", "default": 0.5} for i in range(6)]
    layers, head_scale = panama_layers(model_config)
    state = load_ckpt_state(PANAMA / "demo_ckpt.ckpt")
    (FIX / "panama_demo.nam").write_text(json.dumps(state_to_parametric_nam(state, layers, head_scale, knobs,
                                                                      metadata={"name": "PANAMA demo"})))

    model = LightningModule.load_from_checkpoint(PANAMA / "demo_ckpt.ckpt", **LightningModule.parse_config(model_config))
    model.eval()
    sr = 48000
    rng = np.random.default_rng(3)
    t = np.arange(sr) / sr
    x = (0.3 * np.sin(2 * np.pi * 110 * t) * np.exp(-t) + 0.05 * rng.standard_normal(sr)).astype(np.float32)
    g = [0.5, 0.2, 0.5, 0.7, 0.5, 0.8]
    with torch.no_grad():
        y = model.forward(torch.tensor(x)[None, :], torch.tensor([g], dtype=torch.float32), pad_start=True)[0].numpy()
    sf.write(FIX / "panama_demo_in.wav", x, sr, subtype="FLOAT")
    sf.write(FIX / "panama_demo_out.wav", y.astype(np.float32), sr, subtype="FLOAT")
    (FIX / "panama_demo_knobs.json").write_text(json.dumps(g))
    print(f"wrote panama_demo.* (ref peak {np.abs(y).max():.3f})")


if __name__ == "__main__":
    widened()
    if "--panama" in sys.argv:
        panama_demo()
