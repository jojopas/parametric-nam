#!/usr/bin/env python3
"""A parametric .nam -> plain NAM WaveNet .nam files, one per knob setting, for players that
only know standard NAM (most hardware modelers, at the time of writing).

    python3 -m parametric_nam.snapshot MODEL.nam OUTDIR --at 0.3 0.7 [--at 0.8 0.5 ...]
        [--name "My Drive"] [--gain-db 0.0] [--invert]

The knob rows of the conditioning input are constant at a fixed setting, so each layer's
input_mixin knob columns fold into its conv bias, and an optional "knob_scale" multiplies
that channel's conv weights and folded bias by (1 + S . g) (SPEC.md). What is left takes the audio only: a standard WaveNet (condition_size 1),
identical in sound to the parametric file at that setting. No numpy, no torch: stdlib only.

--gain-db and --invert bake a level and a polarity flip into the head scale, e.g. to undo a
capture loop that changed the level or flipped polarity.
"""
import argparse, copy, json, math, pathlib, sys


def snapshot(model, g, gain_db=0.0, invert=False):
    if model.get("architecture") not in ("ParametricWaveNet", "CoroParametricWaveNet"):
        raise ValueError("not a ParametricWaveNet")
    cfg = model["config"]
    if cfg.get("head") is not None or "condition_dsp" in cfg:
        raise ValueError("a head or condition_dsp is not supported")
    N = len(cfg["knobs"])
    if len(g) != N:
        raise ValueError(f"{N} knobs, {len(g)} values given")
    w = model["weights"]
    ks = model.get("knob_scale")
    layers = cfg["layers"]
    if ks is not None and len(ks) != len(layers):
        raise ValueError("knob_scale: one entry per layer array")
    out, p = [], 0

    def take(n):
        nonlocal p
        if p + n > len(w):
            raise ValueError("too few weights")
        s = w[p:p + n]; p += n
        return s

    for a, L in enumerate(layers):
        ch, K, hd = L["channels"], L["kernel_size"], L["head_size"]
        if L.get("gated", False):
            raise ValueError("gated layers are not supported")
        if L["condition_size"] != 1 + N:
            raise ValueError(f"array {a}: condition_size {L['condition_size']} is not 1 + {N} knobs")
        inW = L["input_size"]
        r = take(ch * inW)
        if a == 0:
            if inW != 1 + N:
                raise ValueError("array 0 input is not audio + knobs")
            for i in range(ch):
                row = r[i * inW:(i + 1) * inW]
                if any(v != 0.0 for v in row[1:]):
                    raise ValueError("knob rows feed the input rechannel")
                out.append(row[0])
        else:
            out.extend(r)
        if ks is not None and len(ks[a]) != len(L["dilations"]):
            raise ValueError("knob_scale: one entry per layer")
        for l in range(len(L["dilations"])):
            conv = take(ch * ch * K)       # (out i, in j, tap k) at (i*ch + j)*K + k
            bias = take(ch)
            mix = take(ch * (1 + N))       # ch x (audio, knobs)
            one = take(ch * ch + ch)
            sc = [1.0] * ch
            if ks is not None:
                row = ks[a][l]
                if len(row) != ch * N:
                    raise ValueError("knob_scale size")
                sc = [1.0 + sum(row[i * N + n] * g[n] for n in range(N)) for i in range(ch)]
            for i in range(ch):
                out.extend(v * sc[i] for v in conv[i * ch * K:(i + 1) * ch * K])
            out.extend((bias[i] + sum(mix[i * (1 + N) + 1 + n] * g[n] for n in range(N))) * sc[i]
                       for i in range(ch))
            out.extend(mix[i * (1 + N)] for i in range(ch))
            out.extend(one)
        out.extend(take(hd * ch + (hd if L.get("head_bias", False) else 0)))
    hs = take(1)[0] * 10 ** (gain_db / 20) * (-1 if invert else 1)
    out.append(hs)                         # head scale, with the capture loop's level and polarity
    if p != len(w):
        raise ValueError("too many weights")

    plain_layers = []
    for a, L in enumerate(layers):
        L = copy.deepcopy(L)
        L["condition_size"] = 1
        if a == 0:
            L["input_size"] = 1
        plain_layers.append(L)
    res = {k: copy.deepcopy(v) for k, v in model.items() if k not in ("weights", "knob_scale", "config", "architecture")}
    res["architecture"] = "WaveNet"
    res["config"] = {"layers": plain_layers, "head": None, "head_scale": cfg["head_scale"] * 10 ** (gain_db / 20) * (-1 if invert else 1)}
    res["weights"] = out
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("outdir")
    ap.add_argument("--at", type=float, nargs="+", action="append", required=True, help="knob values 0..1, one per knob")
    ap.add_argument("--name", help="base name (default: the model's metadata name)")
    ap.add_argument("--gain-db", type=float, default=0.0, help="level to add, in dB (e.g. what the capture loop lost)")
    ap.add_argument("--invert", action="store_true", help="flip polarity (e.g. the capture loop inverted it)")
    a = ap.parse_args()
    model = json.loads(pathlib.Path(a.model).read_text())
    knobs = [k["name"] for k in model["config"]["knobs"]]
    base = a.name or model.get("metadata", {}).get("name") or pathlib.Path(a.model).stem
    outdir = pathlib.Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    for g in a.at:
        if any(not 0.0 <= v <= 1.0 for v in g):
            sys.exit("knob values are 0..1")
        snap = snapshot(model, g, a.gain_db, a.invert)
        label = " ".join(f"{n} {round(v * 10, 1):g}" for n, v in zip(knobs, g))
        name = f"{base} - {label}"
        snap.setdefault("metadata", {})["name"] = name
        path = outdir / (name.replace("/", "-") + ".nam")
        path.write_text(json.dumps(snap))
        print(path)


if __name__ == "__main__":
    main()
