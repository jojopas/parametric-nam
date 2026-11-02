"""Parametric .nam files: the weight layout shared by the trainer side
(PANAMA checkpoints) and the plugin (cpp/ParametricWaveNet.cpp).

A PANAMA layer adds a Linear(g) term to the WaveNet layer's pre-activation:
    z = conv(x) + input_mixer(audio) + global_mixer(g)
NAM core computes
    z = conv(x) + input_mixin(condition)
so with condition = [audio, g...] and input_mixin = [input_mixer | global_mixer.W]
and global_mixer.bias folded into conv.bias, the two are identical and NAM core
runs the model unchanged. The host feeds the knob rows (cpp/ParametricWaveNet.cpp).

Everything here is numpy + json: no torch, so hosts and tests can use it
without a PyTorch install.
"""
from __future__ import annotations

import copy
import re
from typing import Any

import numpy as np

ARCHITECTURE = "ParametricWaveNet"
LEGACY_ARCHITECTURES = ("CoroParametricWaveNet",)
MAX_KNOBS = 8
NAM_FILE_VERSION = "0.5.4"

_KEY = re.compile(r"_blocks\.(\d+)\.(.*)$")


def strip_state_prefix(state: dict[str, Any]) -> dict[str, np.ndarray]:
    """Keep only the WaveNet tensors, keyed from `_blocks.` on (drops Lightning prefixes)."""
    out: dict[str, np.ndarray] = {}
    for k, v in state.items():
        m = _KEY.search(k)
        if m is None:
            continue
        arr = np.asarray(v.detach().cpu().numpy() if hasattr(v, "detach") else v, dtype=np.float64)
        out[f"_blocks.{m.group(1)}.{m.group(2)}"] = arr
    if not out:
        raise ValueError("no WaveNet tensors (_blocks.*) in the state dict")
    return out


def _take(it: np.ndarray, pos: int, shape: tuple[int, ...]) -> tuple[np.ndarray, int]:
    n = int(np.prod(shape))
    if pos + n > it.size:
        raise ValueError("weights ran out while walking the layers")
    return it[pos:pos + n].reshape(shape), pos + n


def nam_weights_to_state(layers: list[dict], weights: list[float] | np.ndarray) -> tuple[dict[str, np.ndarray], float]:
    """Walk a plain NAM WaveNet's flat weights (no head, no FiLM) into named tensors.
    Returns (state, head_scale). Inverse of state_to_weights for the plain case."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    pos = 0
    state: dict[str, np.ndarray] = {}
    for b, lc in enumerate(layers):
        ch, cond, inp, head = lc["channels"], lc["condition_size"], lc["input_size"], lc["head_size"]
        mid = 2 * ch if lc.get("gated", False) else ch
        k = lc["kernel_size"]
        state[f"_blocks.{b}._rechannel.weight"], pos = _take(w, pos, (ch, inp, 1))
        for l, _d in enumerate(lc["dilations"]):
            p = f"_blocks.{b}._layers.{l}."
            state[p + "_conv.weight"], pos = _take(w, pos, (mid, ch, k))
            state[p + "_conv.bias"], pos = _take(w, pos, (mid,))
            state[p + "_input_mixer.weight"], pos = _take(w, pos, (mid, cond, 1))
            state[p + "_1x1.weight"], pos = _take(w, pos, (ch, ch, 1))
            state[p + "_1x1.bias"], pos = _take(w, pos, (ch,))
        state[f"_blocks.{b}._head_rechannel.weight"], pos = _take(w, pos, (head, ch, 1))
        if lc.get("head_bias", True):
            state[f"_blocks.{b}._head_rechannel.bias"], pos = _take(w, pos, (head,))
    head_scale, pos = _take(w, pos, (1,))
    if pos != w.size:
        raise ValueError(f"{w.size - pos} weights left over: not a plain WaveNet (head or FiLM?)")
    return state, float(head_scale[0])


def _knob_entry(k: dict) -> dict:
    e = {"name": k["name"], "default": float(k.get("default", 0.5))}
    if "labels" in k:
        e["labels"] = list(k["labels"])
    return e


def state_to_parametric_nam(state: dict[str, np.ndarray], layers: list[dict], head_scale: float,
                      knobs: list[dict], *, sample_rate: float = 48000.0,
                      metadata: dict | None = None) -> dict:
    """Fold a PANAMA-shaped state (with `_global_mixer`) into a parametric .nam dict.
    A state without `_global_mixer` tensors gets zero knob weights (the model ignores
    its knobs: useful to test that the plumbing changes nothing)."""
    if not 1 <= len(knobs) <= MAX_KNOBS:
        raise ValueError(f"need 1..{MAX_KNOBS} knobs, got {len(knobs)}")
    for kn in knobs:
        if not isinstance(kn.get("name"), str) or not kn["name"]:
            raise ValueError("every knob needs a name")
        if "labels" in kn and (len(kn["labels"]) < 2 or not all(isinstance(x, str) for x in kn["labels"])):
            raise ValueError(f"knob {kn['name']}: labels must be 2+ strings")
    n = len(knobs)
    out_layers = []
    flat: list[np.ndarray] = []
    for b, lc in enumerate(layers):
        lc = copy.deepcopy(lc)
        ch = lc["channels"]
        mid = 2 * ch if lc.get("gated", False) else ch
        if lc["condition_size"] != 1:
            raise ValueError("source layers must have condition_size 1 (audio only)")
        rech = state[f"_blocks.{b}._rechannel.weight"]
        if b == 0:
            if lc["input_size"] != 1:
                raise ValueError("first layer must take the audio only (input_size 1)")
            lc["input_size"] = 1 + n
            rech = np.concatenate([rech, np.zeros((ch, n, 1))], axis=1)  # knob rows feed nothing here
        lc["condition_size"] = 1 + n
        flat.append(rech.ravel())
        for l, _d in enumerate(lc["dilations"]):
            p = f"_blocks.{b}._layers.{l}."
            conv_w, conv_b = state[p + "_conv.weight"], state[p + "_conv.bias"]
            mixer = state[p + "_input_mixer.weight"]  # (mid, 1, 1)
            gw = state.get(p + "_global_mixer.weight")
            gb = state.get(p + "_global_mixer.bias")
            if gw is None:
                gw = np.zeros((mid, n))
            if gb is None:
                gb = np.zeros((mid,))
            if gw.shape != (mid, n):
                raise ValueError(f"{p}_global_mixer.weight is {gw.shape}, expected {(mid, n)}: knob count mismatch")
            mixin = np.concatenate([mixer.reshape(mid, 1), gw.reshape(mid, n)], axis=1)  # (mid, 1+n)
            flat += [conv_w.ravel(), (conv_b + gb).ravel(), mixin.ravel(),
                     state[p + "_1x1.weight"].ravel(), state[p + "_1x1.bias"].ravel()]
        flat.append(state[f"_blocks.{b}._head_rechannel.weight"].ravel())
        if lc.get("head_bias", True):
            flat.append(state[f"_blocks.{b}._head_rechannel.bias"].ravel())
        out_layers.append(lc)
    flat.append(np.asarray([head_scale], dtype=np.float64))
    weights = np.concatenate(flat)
    md = {"name": "Parametric capture"}
    if metadata:
        md.update(metadata)
    return {
        "version": NAM_FILE_VERSION,
        "metadata": md,
        "architecture": ARCHITECTURE,
        "config": {
            "knobs": [_knob_entry(k) for k in knobs],
            "layers": out_layers,
            "head": None,
            "head_scale": float(head_scale),
        },
        "weights": [float(x) for x in weights],
        "sample_rate": float(sample_rate),
    }


def expected_weight_count(layers: list[dict], n_knobs: int) -> int:
    """How many weights a parametric .nam file with these (source) layers has."""
    total = 0
    for b, lc in enumerate(layers):
        ch, head = lc["channels"], lc["head_size"]
        mid = 2 * ch if lc.get("gated", False) else ch
        k = lc["kernel_size"]
        total += ch * (lc["input_size"] + (n_knobs if b == 0 else 0))
        per_layer = mid * ch * k + mid + mid * (1 + n_knobs) + ch * ch + ch
        total += per_layer * len(lc["dilations"])
        total += head * ch + (head if lc.get("head_bias", True) else 0)
    return total + 1
