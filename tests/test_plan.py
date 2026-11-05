"""Capture planning (parametric_nam/capture_plan.py) and take alignment."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from parametric_nam import capture_plan as cp  # noqa: E402


def test_schedule_has_grid_plus_random_and_starts_centred():
    pts = cp.schedule(3, 3, 10, seed=1)
    assert len(pts) == 27 + 10
    assert pts[0] == [0.5, 0.5, 0.5]
    assert all(len(p) == 3 and all(0.0 <= v <= 1.0 for v in p) for p in pts)
    assert [0.0, 0.0, 0.0] in pts and [1.0, 1.0, 1.0] in pts
    assert cp.schedule(3, 3, 10, seed=1) == pts  # deterministic


def test_schedule_random_points_are_stratified_per_knob():
    pts = cp.schedule(2, 0, 20, seed=3)
    assert len(pts) == 21  # grid 0 still contributes the centre point, first
    assert pts[0] == [0.5, 0.5]
    col = sorted(p[0] for p in pts[1:])
    assert all(i / 20 - 1e-3 <= v <= (i + 1) / 20 + 1e-3 for i, v in enumerate(col))  # values are rounded to 4 dp


def test_data_config_splits_validation_and_needs_aligned_takes(tmp_path):
    plan = {"knobs": ["Gain"], "input": "/x.wav",
            "points": [{"id": f"{i:04d}", "g": [i / 10], "status": "done" if i < 9 else "todo", "aligned": "click"}
                       for i in range(10)]}
    cfg = cp.data_config(plan, tmp_path, val_every=4)
    assert cfg["type"] == "shared_in_multi_out"
    assert len(cfg["train"]["outputs"]) == 7
    assert len(cfg["validation"]["outputs"]) == 2
    assert "delay" not in cfg["train"]
    assert cfg["train"]["ny"] == 8192 and cfg["validation"]["ny"] is None
    assert cfg["validation"]["outputs"][0]["y_path"].endswith("0003.wav")
    assert cfg["validation"]["outputs"][0]["g_vector"] == [0.3]
    with pytest.raises(ValueError):
        cp.data_config({**plan, "points": plan["points"][:1]}, tmp_path, 4)
    plan["points"][2].pop("aligned")
    with pytest.raises(ValueError, match="not aligned"):
        cp.data_config(plan, tmp_path, 4)


def test_model_config_sets_knob_count():
    template = {"net": {"config": {"layers_configs": [{"global_condition_size": 6}, {"global_condition_size": 6}]}}}
    out = cp.model_config({"knobs": ["a", "b", "c"]}, template)
    assert [lc["global_condition_size"] for lc in out["net"]["config"]["layers_configs"]] == [3, 3]
    assert template["net"]["config"]["layers_configs"][0]["global_condition_size"] == 6


def test_new_status_add_roundtrip(tmp_path, capsys):
    d = tmp_path / "s"
    cp.main(["new", "--dir", str(d), "--knobs", "Gain,Tone", "--grid", "2", "--random", "3", "--input", "/x.wav"])
    plan = json.loads((d / "plan.json").read_text())
    assert len(plan["points"]) == 7 and plan["knobs"] == ["Gain", "Tone"]
    cp.main(["add", "--dir", str(d), "--g", "0.1,0.9"])
    plan = json.loads((d / "plan.json").read_text())
    assert plan["points"][-1] == {"id": "0008", "g": [0.1, 0.9], "status": "todo"}
    cp.main(["status", "--dir", str(d)])
    assert "NEXT  0001" in capsys.readouterr().out


def _amp(x, lag):
    """A cranked amp: delayed, inverted, driven hard, low-passed a little."""
    y = np.concatenate([np.zeros(lag, np.float32), x[:-lag]]) if lag else x.copy()
    y = -np.tanh(6.0 * y)
    return (0.5 * (y + np.concatenate([[0], y[:-1]]))).astype(np.float32)


def test_click_alignment_through_a_cranked_amp():
    from parametric_nam.capture_record import aligned_take, click_positions, delay_from_clicks, with_clicks

    sr = 48000
    rng = np.random.default_rng(1)
    x = (0.3 * np.sin(2 * np.pi * 110 * np.arange(sr) / sr) * rng.random(sr)).astype(np.float32)
    play, start = with_clicks(x, sr)
    assert start == int(1.75 * sr) and len(play) == start + sr + int(0.6 * sr)
    assert play[click_positions(sr)[0] + 24] > 0.4
    lag = 14660
    y = _amp(play, lag) + (0.001 * rng.standard_normal(len(play))).astype(np.float32)
    d = delay_from_clicks(y, sr)
    assert d is not None and abs(d - lag) <= 3
    al = aligned_take(y, start, d, len(x))
    assert len(al) == len(x)
    # the aligned take answers the input: its envelope tracks x's
    assert np.corrcoef(np.abs(al[1000:-1000]), np.abs(x[1000:-1000]))[0, 1] > 0.6
    assert delay_from_clicks(np.zeros(len(play), np.float32), sr) is None  # no response


def test_correlation_alignment_through_a_cranked_amp():
    from parametric_nam.capture_record import delay_by_correlation, level_report

    sr = 48000
    rng = np.random.default_rng(2)
    t = np.arange(sr * 6) / sr
    x = (0.3 * np.sin(2 * np.pi * 110 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t)) * rng.random(len(t))).astype(np.float32)
    lag = 14700
    y = _amp(x, lag)
    assert abs(delay_by_correlation(x, y, sr) - lag) <= 5
    pk, rms, warn = level_report(np.full(100, 1.0, np.float32))
    assert warn and warn[0].startswith("CLIPPED")
    pk, rms, warn = level_report(np.full(100, 1e-4, np.float32))
    assert any("quiet" in w for w in warn)


def test_meter_bar_and_hold():
    from parametric_nam.capture_record import Meters, bar

    assert bar(-60.0) == "." * 24
    assert bar(0.0).endswith("!") and bar(0.0).count("!") == 1
    assert bar(-6.0).count("#") == 22 and "!" not in bar(-6.0)
    m = Meters(["IN1 DI", "IN2 amp"])
    m.feed(np.array([[0.5, 1.0], [0.1, 0.2]], dtype=np.float32))
    line = m.line()
    assert "IN1 DI" in line and "IN2 amp" in line
    assert "-6.0 dBFS" in line and "CLIP x1" in line


def test_extend_keeps_old_takes_and_groups_switch_positions(tmp_path, capsys):
    d = tmp_path / "s"
    cp.main(["new", "--dir", str(d), "--knobs", "Gain,Bass", "--grid", "2", "--random", "2", "--seed", "1"])
    plan = cp.load(d)
    for p in plan["points"][:3]:
        p["status"], p["aligned"], p["delay"] = "done", "click", 100
    cp.save(d, plan)
    cp.main(["extend", "--dir", str(d), "--knobs", "Middle,Presence", "--defaults", "0.5,0.5",
             "--switch", "Power", "--switch-labels", "half,full", "--random", "6", "--switch-random", "4",
             "--grid", "2", "--seed", "2"])
    plan = cp.load(d)
    assert plan["knobs"] == ["Gain", "Bass", "Middle", "Presence", "Power"]
    assert plan["switches"] == {"Power": ["half", "full"]}
    old = plan["points"][:6]
    assert [p["status"] for p in old[:3]] == ["done"] * 3
    assert all(p["g"][2:] == [0.5, 0.5, 1.0] for p in old), "old takes sit at the defaults, switch at full"
    assert all(p.get("delay") == 100 for p in old[:3])
    new = plan["points"][6:]
    assert len(new) == 4 + 6 + 4, "2x2 grid on the new knobs + random at full + random at half"
    assert all(len(p["g"]) == 5 and p["status"] == "todo" for p in new)
    sw = [p["g"][4] for p in new]
    assert sw == [1.0] * 10 + [0.0] * 4, "all full-power takes first, then flip the switch once"
    assert all(p["g"][:2] == [0.5, 0.5] for p in new[:4]), "grid points vary only the new knobs"
    assert {tuple(p["g"][2:4]) for p in new[:4]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    ids = [p["id"] for p in plan["points"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    cp.main(["status", "--dir", str(d)])
    out = capsys.readouterr().out
    assert "3/20 takes done" in out and "Power FULL" in out


def test_setting_text_shows_switch_label():
    plan = {"knobs": ["Gain", "Power"], "switches": {"Power": ["half", "full"]}}
    assert cp.setting_text(plan, [0.25, 0.0]) == "Gain 2.5   Power HALF"


def test_new_records_look_and_author(tmp_path):
    d = tmp_path / "s"
    cp.main(["new", "--dir", str(d), "--knobs", "Gain", "--grid", "2", "--random", "1", "--input", "/x.wav",
             "--look", "tweed", "--author", "Jane"])
    plan = json.loads((d / "plan.json").read_text())
    assert plan["look"] == "tweed" and plan["author"] == "Jane"
    with pytest.raises(SystemExit):
        cp.main(["new", "--dir", str(tmp_path / "t"), "--knobs", "Gain", "--input", "/x.wav", "--look", "orange"])
