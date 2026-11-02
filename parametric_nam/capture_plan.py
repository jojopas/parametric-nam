"""Plan a parametric capture session and turn its takes into PANAMA training data.

  capture_plan.py new  --dir SESSION --knobs Gain,Bass,Treble [--grid 3] [--random 24] [--seed 1]
  capture_plan.py status --dir SESSION                 progress + the next setting to dial in
  capture_plan.py add  --dir SESSION --g 0.2,0.7,0.5   append a point (active-learning suggestions)
  capture_plan.py extend --dir SESSION --knobs Middle,Presence --defaults 0.5,0.5
                         [--switch Power --switch-labels half,full] [--grid 2] [--random 60] [--switch-random 40]
                                                       add knobs to a session: finished takes keep their
                                                       recordings (new knobs at the defaults, switch at its
                                                       last label) and new points are scheduled, all takes at
                                                       one switch position before the other
  capture_plan.py data-config --dir SESSION [--val-every 8]
                                                       write SESSION/data.json for PANAMA (takes must be aligned)
  capture_plan.py model-config --dir SESSION           write SESSION/model.json (knob count filled in)

Knob values are 0..1 (shown as 0-10 on the dial). The point order is the
g-vector order everywhere: here, in the recordings, in the trained model and in
the plugin's knob list. capture_record.py records the takes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

import os

HERE = pathlib.Path(__file__).resolve().parent
# a PANAMA checkout (https://github.com/ETH-DISCO/PANAMA); override with PANAMA_DIR
PANAMA = pathlib.Path(os.environ.get("PANAMA_DIR", HERE.parent.parent / "PANAMA"))
# the NAM v3 training signal (input.wav from the NAM trainer); override with NAM_INPUT
DEFAULT_INPUT = pathlib.Path(os.environ.get("NAM_INPUT", "input.wav"))


def dial(v: float) -> str:
    return f"{v * 10:.1f}"


def setting_text(plan: dict, g: list[float]) -> str:
    """'Gain 2.5   Power HALF': dial values, switch knobs by label."""
    sw = plan.get("switches", {})
    parts = []
    for k, v in zip(plan["knobs"], g):
        parts.append(f"{k} {sw[k][int(round(v))].upper()}" if k in sw else f"{k} {dial(v)}")
    return "   ".join(parts)


def latin_hypercube(n: int, n_knobs: int, rng) -> list[list[float]]:
    cols = []
    for _ in range(n_knobs):
        strata = (np.arange(n) + rng.random(n)) / n
        cols.append(rng.permutation(strata))
    return [list(map(float, np.round(p, 4))) for p in np.stack(cols, axis=1)] if n > 0 else []


def schedule(n_knobs: int, grid: int, n_random: int, seed: int) -> list[list[float]]:
    """A full grid (grid levels per knob) plus stratified random points, shuffled.
    The grid pins the corners and the middle; the random points fill the gaps the
    model would otherwise have to guess at."""
    levels = np.linspace(0.0, 1.0, grid) if grid > 1 else np.array([0.5])
    pts = [list(map(float, p)) for p in itertools.product(levels, repeat=n_knobs)]
    rng = np.random.default_rng(seed)
    pts += latin_hypercube(n_random, n_knobs, rng)  # each knob's values spread over n_random strata
    order = rng.permutation(len(pts))
    # keep the first point a centred one so the first take doubles as a level check
    pts = [pts[i] for i in order]
    centre = min(range(len(pts)), key=lambda i: sum((v - 0.5) ** 2 for v in pts[i]))
    pts.insert(0, pts.pop(centre))
    return pts


def load(d: pathlib.Path) -> dict:
    p = d / "plan.json"
    if not p.exists():
        sys.exit(f"no plan at {p}: run `capture_plan.py new` first")
    return json.loads(p.read_text())


def save(d: pathlib.Path, plan: dict) -> None:
    (d / "plan.json").write_text(json.dumps(plan, indent=1))


def cmd_new(a) -> None:
    d = pathlib.Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    if (d / "plan.json").exists() and not a.force:
        sys.exit(f"{d / 'plan.json'} exists; --force to overwrite")
    knobs = [k.strip() for k in a.knobs.split(",") if k.strip()]
    if not 1 <= len(knobs) <= 8:
        sys.exit("1..8 knobs")
    pts = schedule(len(knobs), a.grid, a.random, a.seed)
    plan = {
        "name": a.name or d.name,
        "knobs": knobs,
        "input": str(pathlib.Path(a.input).resolve()),
        "sample_rate": 48000,
        "latency_samples": None,
        "points": [{"id": f"{i + 1:04d}", "g": g, "status": "todo"} for i, g in enumerate(pts)],
    }
    save(d, plan)
    (d / "takes").mkdir(exist_ok=True)
    print(f"{len(pts)} points ({a.grid}^{len(knobs)} grid + {a.random} random) -> {d / 'plan.json'}")
    cmd_status(a)


def cmd_status(a) -> None:
    d = pathlib.Path(a.dir)
    plan = load(d)
    done = [p for p in plan["points"] if p["status"] == "done"]
    todo = [p for p in plan["points"] if p["status"] != "done"]
    print(f"{plan['name']}: {len(done)}/{len(plan['points'])} takes done; knobs {plan['knobs']}")
    if plan.get("latency_samples") is not None:
        print(f"loop latency {plan['latency_samples']} samples")
    if todo:
        nxt = todo[0]
        print("NEXT  " + nxt["id"] + "  " + setting_text(plan, nxt["g"]))


def cmd_add(a) -> None:
    d = pathlib.Path(a.dir)
    plan = load(d)
    g = [float(x) for x in a.g.split(",")]
    if len(g) != len(plan["knobs"]) or not all(0.0 <= v <= 1.0 for v in g):
        sys.exit(f"need {len(plan['knobs'])} values in 0..1")
    pid = f"{len(plan['points']) + 1:04d}"
    plan["points"].append({"id": pid, "g": g, "status": "todo"})
    save(d, plan)
    print(f"added {pid} {g}")


def extend(plan: dict, knobs: list[str], defaults: list[float], switch: str | None, labels: list[str],
           grid: int, n_random: int, n_switch: int, seed: int) -> int:
    """Add knobs (and at most one two-position switch) to a plan in place; returns the
    number of new points. Finished takes keep their recordings, sitting at the new
    knobs' defaults and the switch's last label. New points: a grid over the new knobs
    only (old knobs centred), then random points over every knob, then random points
    at the other switch position, so the switch is flipped once per session."""
    if len(knobs) != len(defaults) or any(k in plan["knobs"] for k in knobs):
        raise ValueError("one default per new knob, and no knob twice")
    if switch and (switch in plan["knobs"] or len(labels) != 2):
        raise ValueError("a switch needs two labels and a new name")
    if len(plan["knobs"]) + len(knobs) + bool(switch) > 8:
        raise ValueError("at most 8 knobs")
    n_old = len(plan["knobs"])
    plan["knobs"] = plan["knobs"] + knobs + ([switch] if switch else [])
    if switch:
        plan.setdefault("switches", {})[switch] = labels
    tail = list(defaults) + ([1.0] if switch else [])
    for p in plan["points"]:
        p["g"] = p["g"] + tail
    rng = np.random.default_rng(seed)
    n_cont = n_old + len(knobs)
    new = []
    if grid > 1 and knobs:
        levels = np.linspace(0.0, 1.0, grid)
        for combo in itertools.product(levels, repeat=len(knobs)):
            new.append([0.5] * n_old + list(map(float, combo)))
    new += latin_hypercube(n_random, n_cont, rng)
    if switch:
        new = [g + [1.0] for g in new] + [g + [0.0] for g in latin_hypercube(n_switch, n_cont, rng)]
    start = len(plan["points"])
    plan["points"] += [{"id": f"{start + i + 1:04d}", "g": g, "status": "todo"} for i, g in enumerate(new)]
    return len(new)


def cmd_extend(a) -> None:
    d = pathlib.Path(a.dir)
    plan = load(d)
    knobs = [k.strip() for k in a.knobs.split(",") if k.strip()] if a.knobs else []
    defaults = [float(x) for x in a.defaults.split(",")] if a.defaults else [0.5] * len(knobs)
    labels = [x.strip() for x in a.switch_labels.split(",")] if a.switch else []
    try:
        n = extend(plan, knobs, defaults, a.switch, labels, a.grid, a.random, a.switch_random, a.seed)
    except ValueError as e:
        sys.exit(str(e))
    save(d, plan)
    print(f"{n} new points; knobs now {plan['knobs']} -> {d / 'plan.json'}")
    cmd_status(a)


def data_config(plan: dict, takes_dir: pathlib.Path, val_every: int) -> dict:
    """PANAMA's shared_in_multi_out config from the finished takes. Every
    val_every-th take is held out for validation so the loss means something."""
    done = [p for p in plan["points"] if p["status"] == "done"]
    if len(done) < 2:
        raise ValueError("need at least 2 finished takes")
    if any(not p.get("aligned") for p in done):
        raise ValueError("some takes are not aligned yet: run `capture_record.py align --dir SESSION`")
    outs = [{"y_path": str((takes_dir / f"{p['id']}.wav").resolve()), "g_vector": p["g"]} for p in done]
    val = [o for i, o in enumerate(outs) if val_every > 0 and i % val_every == val_every - 1]
    train = [o for o in outs if o not in val] if val else outs
    if not val:  # tiny sessions: validate on the last take anyway
        val = [train.pop()] if len(train) > 1 else list(train)
    # takes are aligned to the input when saved, so no delay correction here
    return {
        "type": "shared_in_multi_out",
        "train": {"x_path": plan["input"], "outputs": train, "ny": 8192},
        "validation": {"x_path": plan["input"], "outputs": val, "ny": None},
    }


def cmd_data_config(a) -> None:
    d = pathlib.Path(a.dir)
    plan = load(d)
    try:
        cfg = data_config(plan, d / "takes", a.val_every)
    except ValueError as e:
        sys.exit(str(e))
    (d / "data.json").write_text(json.dumps(cfg, indent=1))
    print(f"{len(cfg['train']['outputs'])} train / {len(cfg['validation']['outputs'])} validation -> {d / 'data.json'}")


def model_config(plan: dict, template: dict) -> dict:
    cfg = json.loads(json.dumps(template))
    for lc in cfg["net"]["config"]["layers_configs"]:
        lc["global_condition_size"] = len(plan["knobs"])
    return cfg


def cmd_model_config(a) -> None:
    d = pathlib.Path(a.dir)
    plan = load(d)
    template = json.loads(pathlib.Path(a.template).read_text())
    (d / "model.json").write_text(json.dumps(model_config(plan, template), indent=1))
    print(f"model config for {len(plan['knobs'])} knobs -> {d / 'model.json'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new")
    p.add_argument("--dir", required=True)
    p.add_argument("--knobs", required=True)
    p.add_argument("--grid", type=int, default=3, help="levels per knob for the full grid (0 = no grid)")
    p.add_argument("--random", type=int, default=24, help="extra stratified random points")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--name")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="the shared dry signal (NAM v3 input)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_new)
    p = sub.add_parser("status")
    p.add_argument("--dir", required=True)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("add")
    p.add_argument("--dir", required=True)
    p.add_argument("--g", required=True)
    p.set_defaults(fn=cmd_add)
    p = sub.add_parser("extend")
    p.add_argument("--dir", required=True)
    p.add_argument("--knobs", default="", help="new continuous knobs, comma separated")
    p.add_argument("--defaults", default="", help="where the finished takes sat for each new knob (0..1)")
    p.add_argument("--switch", help="a two-position switch to add, e.g. Power")
    p.add_argument("--switch-labels", default="half,full", help="labels for switch value 0,1; finished takes are at 1")
    p.add_argument("--grid", type=int, default=2, help="levels per NEW knob for a grid with the old knobs centred")
    p.add_argument("--random", type=int, default=60, help="random points over all knobs at switch position 1")
    p.add_argument("--switch-random", type=int, default=40, help="random points at switch position 0")
    p.add_argument("--seed", type=int, default=2)
    p.set_defaults(fn=cmd_extend)
    p = sub.add_parser("data-config")
    p.add_argument("--dir", required=True)
    p.add_argument("--val-every", type=int, default=8)
    p.set_defaults(fn=cmd_data_config)
    p = sub.add_parser("model-config")
    p.add_argument("--dir", required=True)
    p.add_argument("--template", default=str(HERE.parent / "configs/model_wide.json"))
    p.set_defaults(fn=cmd_model_config)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
