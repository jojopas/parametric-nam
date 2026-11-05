"""Convert a PANAMA checkpoint into a parametric .nam (architecture "ParametricWaveNet").

usage: convert_panama.py --ckpt last.ckpt --model-config model.json \
           --knobs Gain,Bass,Mid,Treble --out amp.nam [--name "Deluxe"] [--defaults 0.5,0.5,0.5,0.5]
       convert_panama.py --ckpt last.ckpt --model-config model.json --plan SESSION/plan.json --out amp.nam
           (knob names, order and switch labels from the capture plan)

The knob order is the g-vector order used when recording (capture_plan.py keeps it).
Needs torch only to read the checkpoint (run it with the PANAMA venv).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from parametric_nam.format import LOOKS, panel_metadata, state_to_parametric_nam, strip_state_prefix  # noqa: E402


def load_ckpt_state(path: pathlib.Path) -> dict:
    import torch  # local import: the plugin tests never need it

    ck = torch.load(path, map_location="cpu", weights_only=True)
    return strip_state_prefix(ck["state_dict"] if "state_dict" in ck else ck)


def panama_layers(model_config: dict) -> tuple[list[dict], float]:
    """PANAMA's model config -> plain NAM layer configs (drop its global_condition_size)."""
    net = model_config["net"]["config"]
    layers = []
    for lc in net["layers_configs"]:
        lc = {k: v for k, v in lc.items() if k != "global_condition_size"}
        layers.append(lc)
    return layers, float(net.get("head_scale", 1.0))


def ask_look(input=input) -> str:
    """Ask which look a host should draw the capture in; Enter picks generic."""
    print("Amp look for this capture:")
    for i, name in enumerate(LOOKS, 1):
        print(f"  {i}. {name}")
    while True:
        ans = input(f"look [1-{len(LOOKS)}, Enter = {LOOKS[0]}]: ").strip().lower()
        if not ans:
            return LOOKS[0]
        if ans.isdigit() and 1 <= int(ans) <= len(LOOKS):
            return LOOKS[int(ans) - 1]
        if ans in LOOKS:
            return ans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--model-config", required=True, type=pathlib.Path, help="PANAMA model config JSON")
    ap.add_argument("--knobs", help="comma-separated knob names, in g-vector order")
    ap.add_argument("--plan", type=pathlib.Path, help="capture plan.json: knob names, order and switch labels")
    ap.add_argument("--defaults", help="comma-separated 0..1 defaults (default 0.5 each)")
    ap.add_argument("--name", default="Parametric capture")
    ap.add_argument("--loudness", type=float, help="metadata.loudness (dB) if measured")
    ap.add_argument("--sample-rate", type=float, default=48000.0)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--look", help=f"amp look a host draws the capture in: {', '.join(LOOKS)}")
    ap.add_argument("--label", action="append", default=[], metavar="KNOB=LABEL",
                    help="display label for a knob (repeatable), e.g. Gain=VOLUME")
    ap.add_argument("--author", help="metadata.modeled_by")
    ap.add_argument("--note", help="metadata.author_note")
    a = ap.parse_args(argv)

    plan = json.loads(a.plan.read_text()) if a.plan else None
    if plan:
        names = plan["knobs"]
    elif a.knobs:
        names = [s.strip() for s in a.knobs.split(",") if s.strip()]
    else:
        sys.exit("give --knobs or --plan")
    defaults = [float(x) for x in a.defaults.split(",")] if a.defaults else [0.5] * len(names)
    if len(defaults) != len(names):
        sys.exit("--defaults must have one value per knob")
    knobs = [{"name": n, "default": d} for n, d in zip(names, defaults)]
    for k in knobs:
        if plan and k["name"] in plan.get("switches", {}):
            k["labels"] = plan["switches"][k["name"]]
            k["default"] = 1.0  # sessions start at the switch's last position

    layers, head_scale = panama_layers(json.loads(a.model_config.read_text()))
    state = load_ckpt_state(a.ckpt)
    md = {"name": plan["name"] if plan and a.name == "Parametric capture" else a.name}
    if a.loudness is not None:
        md["loudness"] = a.loudness
    labels = dict((plan or {}).get("labels", {}))
    for item in a.label:
        if "=" not in item:
            sys.exit(f"--label {item!r}: use KNOB=LABEL")
        k, v = item.split("=", 1)
        labels[k.strip()] = v.strip()
    look = a.look or (plan or {}).get("look")
    if look is None:
        look = ask_look() if sys.stdin.isatty() else LOOKS[0]
    try:
        md["panel"] = panel_metadata(look, labels)
    except ValueError as e:
        sys.exit(str(e))
    author = a.author or (plan or {}).get("author")
    note = a.note or (plan or {}).get("note")
    if author:
        md["modeled_by"] = author
    if note:
        md["author_note"] = note
    model = state_to_parametric_nam(state, layers, head_scale, knobs, sample_rate=a.sample_rate, metadata=md)
    a.out.write_text(json.dumps(model))
    print(f"wrote {a.out}  knobs={names}  weights={len(model['weights'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
