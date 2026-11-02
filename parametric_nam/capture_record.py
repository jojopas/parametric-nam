"""Record the takes of a parametric capture session (reamp loop through the amp).

  capture_record.py devices
  capture_record.py meter --device "My Interface" --in 1,2 [--out 3 --play]
        live level meters on those inputs (DI on 1, amp return on 2); with --play the
        training signal loops out of --out so you set the master / load box against it
  capture_record.py loopback --device "My Interface" --out 3 --in 1
        dry loop test: plays clicks, reports latency and level (patch out -> in directly)
  capture_record.py take --dir SESSION --device "My Interface" --out 3 --in 1 [--id 0007]
        one take: shows the knob setting, waits for Enter, plays the input while recording
  capture_record.py run  --dir SESSION --device "My Interface" --out 3 --in 1
        every remaining take in order, Enter between them, q to stop
  capture_record.py align --dir SESSION
        align takes recorded by an older version (no calibration clicks) by correlation

--out / --in are 1-based channel numbers on the device (meter takes a list).
Every take shows live meters for the recorded input and any --watch inputs. The input signal
plays out of --out into the reamp box -> amp -> load box / DI -> back into --in.
Each take is checked for level and clipping and aligned to the input to the
sample before it is saved (calibration clicks ahead of the signal; see
"alignment" below). The raw recording is kept under SESSION/raw/. Needs sounddevice and soundfile.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from parametric_nam.capture_plan import DEFAULT_INPUT, dial, load, save, setting_text  # noqa: E402


def read_wav(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x[:, 0], sr


# ---- alignment ---------------------------------------------------------------
# The interface's duplex stream adds a long, run-dependent round trip (~300 ms
# on the interface it was developed on, with the default buffering), so every take is aligned to the
# input before it is saved. Two ways:
#  * clicks: the playback is [silence, 3 clicks, silence, input, tail]; the amp's
#    response to the clicks gives the delay to the sample (consistent across
#    takes, which is what matters), and the saved take is the input span only.
#  * correlation: for takes recorded without clicks (or if the clicks are not
#    found) a coarse envelope match over the whole take, refined by a low-passed
#    waveform cross-correlation (good to a few samples through a cranked amp).
PRE_SEC, GAP_SEC, TAIL_SEC = 0.5, 0.25, 0.6
CLICK_LEN, CLICK_AMP, N_CLICKS = 48, 0.5, 3


def click_positions(sr: int) -> list[int]:
    return [int((PRE_SEC + GAP_SEC * k) * sr) for k in range(N_CLICKS)]


def with_clicks(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """The playback signal and the sample where the input starts in it."""
    start = int((PRE_SEC + GAP_SEC * N_CLICKS + PRE_SEC) * sr)
    out = np.zeros(start + len(x) + int(TAIL_SEC * sr), dtype=np.float32)
    for c in click_positions(sr):
        out[c:c + CLICK_LEN] = (CLICK_AMP * np.sin(np.pi * np.arange(CLICK_LEN) / CLICK_LEN)).astype(np.float32)
    out[start:start + len(x)] = x
    return out, start


def delay_from_clicks(y: np.ndarray, sr: int) -> int | None:
    """Loop delay in samples from the amp's response to the clicks, or None if not found.
    The first response after the first click sets the delay; the other clicks must
    answer at the same delay (within 20 samples) or the loop is not trusted."""
    clicks = click_positions(sr)
    start = int((PRE_SEC + GAP_SEC * N_CLICKS + PRE_SEC) * sr)  # where the input begins
    if len(y) < start:
        return None
    noise = float(np.sqrt(np.mean(y[int(0.05 * sr):int(0.4 * sr)] ** 2)))
    thr = max(20.0 * noise, 0.03)
    seg = np.abs(y[clicks[0]:start])
    hit = np.flatnonzero(seg > thr)
    if not hit.size:
        return None
    d = int(hit[0])
    for c in clicks[1:]:
        lo = c + d - 20
        hi = c + d + int(0.02 * sr)
        if hi > len(y):
            return None
        w = np.flatnonzero(np.abs(y[lo:hi]) > thr)
        if not w.size or abs((lo + int(w[0])) - (c + d)) > 20:
            return None
    return d


def delay_by_correlation(x: np.ndarray, y: np.ndarray, sr: int, max_sec: float = 1.0) -> int:
    """y lags x by this many samples; works through heavy distortion and either polarity."""
    from scipy.signal import butter, fftconvolve, sosfiltfilt

    n = min(len(x), len(y))
    a = x[:n].astype(np.float64)
    b = y[:n].astype(np.float64)
    maxlag = int(max_sec * sr)

    def lag(p, q, lo, hi):
        p = p - p.mean()
        q = q - q.mean()
        cc = fftconvolve(q, p[::-1], mode="full")
        base = len(p) - 1
        seg = cc[base + max(lo, 0):base + hi + 1]
        return max(lo, 0) + int(np.argmax(np.abs(seg)))

    env = butter(2, 300 / (sr / 2), output="sos")
    coarse = lag(sosfiltfilt(env, np.abs(a)), sosfiltfilt(env, np.abs(b)), 0, maxlag)
    lp = butter(4, 1500 / (sr / 2), output="sos")
    return lag(sosfiltfilt(lp, a), sosfiltfilt(lp, b), coarse - 1200, coarse + 1200)


def aligned_take(y: np.ndarray, start: int, delay: int, n: int) -> np.ndarray:
    """The n samples of y that answer the input, which began at `start` in the playback."""
    out = np.zeros(n, dtype=np.float32)
    seg = y[start + delay:start + delay + n]
    out[:len(seg)] = seg
    return out


def estimate_delay(x: np.ndarray, y: np.ndarray, sr: int, **_) -> int:
    """Kept for the loopback test: correlation over the first seconds."""
    return delay_by_correlation(x, y, sr)


def level_report(y: np.ndarray) -> tuple[float, float, list[str]]:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if y.size else 0.0
    db = lambda v: 20 * np.log10(max(v, 1e-9))
    warn = []
    if peak >= 0.999:
        warn.append("CLIPPED: lower the return level and redo this take")
    elif db(peak) > -1.0:
        warn.append("hot: peak above -1 dBFS")
    if db(rms) < -40:
        warn.append("very quiet: is the amp on and the loop patched?")
    return db(peak), db(rms), warn


def db(v: float) -> float:
    return 20 * np.log10(max(float(v), 1e-9))


def bar(peak_db: float, width: int = 24) -> str:
    """-60..0 dBFS as a bar; the top 3 dB are drawn as a hot zone."""
    frac = min(1.0, max(0.0, (peak_db + 60.0) / 60.0))
    n = int(round(frac * width))
    hot = int(round(width * 57 / 60))
    return "".join("#" if i < min(n, hot) else ("!" if i < n else ".") for i in range(width))


class Meters:
    """Peak / hold / clip per channel, fed from the audio callback, drawn by the main thread."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.peak = np.zeros(len(labels))
        self.hold = np.full(len(labels), -120.0)
        self.hold_t = np.zeros(len(labels))
        self.clips = np.zeros(len(labels), dtype=int)

    def feed(self, block: np.ndarray) -> None:
        pk = np.max(np.abs(block), axis=0) if block.size else np.zeros(len(self.labels))
        self.peak = np.maximum(self.peak * 0.6, pk)  # fast attack, ~decays over a few frames
        self.clips += (pk >= 0.999)

    def line(self) -> str:
        now = time.time()
        parts = []
        for i, lab in enumerate(self.labels):
            p = db(self.peak[i])
            if p >= self.hold[i] or now - self.hold_t[i] > 1.5:
                self.hold[i], self.hold_t[i] = p, now
            clip = f" CLIP x{self.clips[i]}" if self.clips[i] else ""
            parts.append(f"{lab} [{bar(p)}] {self.hold[i]:6.1f} dBFS{clip}")
        return "   ".join(parts)


def stream(x: np.ndarray | None, sr: int, device: str, out_ch: int | None, in_chs: list[int],
           labels: list[str], record_ch: int | None, loop: bool = False, title: str = "") -> np.ndarray:
    """Play x out of out_ch (looping if asked) while metering in_chs live; returns what
    record_ch captured (empty if None). Runs until x ends, or Ctrl-C when looping/metering."""
    import sounddevice as sd

    meters = Meters(labels)
    n_in = max(in_chs)
    n_out = out_ch or 1
    rec_idx = in_chs.index(record_ch) if record_ch in in_chs else None
    chunks: list[np.ndarray] = []
    pos = [0]
    done = [False]
    total = len(x) if x is not None else 0

    def cb(indata, outdata, frames, t, status):
        outdata.fill(0)
        if x is not None and out_ch is not None:
            end = pos[0] + frames
            seg = x[pos[0]:end]
            if len(seg) < frames and loop and total:
                seg = np.concatenate([seg, x[: frames - len(seg)]])
                pos[0] = frames - (total - pos[0])
            else:
                pos[0] = end
            outdata[: len(seg), out_ch - 1] = seg
            if not loop and pos[0] >= total:
                done[0] = True
        sel = indata[:, [c - 1 for c in in_chs]]
        meters.feed(sel)
        if rec_idx is not None:
            chunks.append(sel[:, rec_idx].copy())

    with sd.Stream(device=device, samplerate=sr, blocksize=1024, dtype="float32",
                   channels=(n_in, n_out), callback=cb):
        try:
            while not done[0]:
                prog = f"  {pos[0] / sr:5.0f}/{total / sr:.0f} s" if (x is not None and not loop) else ""
                sys.stdout.write("\r" + title + meters.line() + prog + "   ")
                sys.stdout.flush()
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    sys.stdout.write("\n")
    y = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return y[:total] if total else y


def playrec(x: np.ndarray, sr: int, device: str, out_ch: int, in_ch: int, watch: list[int] = (),
            title: str = "") -> np.ndarray:
    ins = sorted(set([in_ch, *watch]))
    labels = [f"IN{c}" + (" amp" if c == in_ch else " DI" if c == 1 else "") for c in ins]
    return stream(x, sr, device, out_ch, ins, labels, in_ch, loop=False, title=title)


def cmd_meter(a) -> None:
    ins = [int(c) for c in a.inp.split(",")]
    labels = [f"IN{c}" + (" DI" if c == 1 else " amp" if c == 2 else "") for c in ins]
    x = None
    if a.play:
        if not a.out:
            sys.exit("--play needs --out")
        x, sr = read_wav(a.input)
    else:
        sr = 48000
    print("meters live; Ctrl-C to stop" + (" (training signal looping)" if a.play else ""))
    stream(x, sr, a.device, a.out, ins, labels, None, loop=True)


def cmd_devices(a) -> None:
    import sounddevice as sd

    print(sd.query_devices())


def cmd_loopback(a) -> None:
    sr = 48000
    t = np.arange(sr * 3) / sr
    x = np.zeros_like(t, dtype=np.float32)
    for k in range(5):
        i = int((0.5 + 0.5 * k) * sr)
        x[i:i + 48] = 0.5 * np.hanning(48)
    y = playrec(x, sr, a.device, a.out, a.inp, title="loop  ")
    d = estimate_delay(x, y, sr, seconds=3.0)
    pk, rms, warn = level_report(y)
    print(f"latency {d} samples ({d / sr * 1000:.2f} ms)  return peak {pk:.1f} dBFS rms {rms:.1f} dBFS")
    for w in warn:
        print("  !", w)


def one_take(plan: dict, d: pathlib.Path, point: dict, a) -> bool:
    import soundfile as sf

    x, sr = read_wav(plan["input"])
    if sr != plan["sample_rate"]:
        sys.exit(f"input is {sr} Hz, plan says {plan['sample_rate']}")
    setting = setting_text(plan, point["g"])
    print("\n" + "=" * 64)
    print(f"TAKE {point['id']}    {setting}")
    print("=" * 64)
    ans = input("dial it in, then Enter to record (s = skip, q = quit): ").strip().lower()
    if ans == "q":
        return False
    if ans == "s":
        return True
    play, start = with_clicks(x, sr)
    print(f"recording {len(play) / sr:.0f} s ...")
    t0 = time.time()
    y = playrec(play, sr, a.device, a.out, a.inp, watch=a.watch, title=f"{point['id']}  ")
    delay = delay_from_clicks(y, sr)
    method = "click"
    if delay is None:
        delay = delay_by_correlation(x, y[start:], sr)
        method = "xcorr"
        print("  ! clicks not found in the return; aligned by correlation instead")
    yal = aligned_take(y, start, delay, len(x))
    pk, rms, warn = level_report(yal)
    print(f"done in {time.time() - t0:.0f} s: peak {pk:.1f} dBFS, rms {rms:.1f} dBFS, "
          f"loop delay {delay} samples ({delay / sr * 1000:.1f} ms, {method})")
    ref = plan.get("latency_samples")
    if ref is not None and abs(delay - ref) > 50:
        warn.append(f"loop delay moved from {ref} to {delay} samples: check the patch (take is aligned anyway)")
    for w in warn:
        print("  !", w)
    if any(w.startswith("CLIPPED") for w in warn):
        return True
    (d / "raw").mkdir(exist_ok=True)
    sf.write(d / "raw" / f"{point['id']}.wav", y, sr, subtype="PCM_24")
    out = d / "takes" / f"{point['id']}.wav"
    sf.write(out, yal, sr, subtype="PCM_24")
    point["status"] = "done"
    point["peak_dbfs"], point["rms_dbfs"], point["delay"], point["aligned"] = round(pk, 2), round(rms, 2), delay, method
    delays = [p["delay"] for p in plan["points"] if p.get("aligned")]
    plan["latency_samples"] = int(np.median(delays))
    save(d, plan)
    print(f"saved {out.name}")
    return True


def cmd_align(a) -> None:
    """Align takes saved by an earlier version (raw input-only recordings) by correlation."""
    import soundfile as sf

    d = pathlib.Path(a.dir)
    plan = load(d)
    x, sr = read_wav(plan["input"])
    (d / "raw").mkdir(exist_ok=True)
    for p in plan["points"]:
        if p["status"] != "done" or p.get("aligned"):
            continue
        f = d / "takes" / f"{p['id']}.wav"
        y, sr2 = read_wav(str(f))
        delay = delay_by_correlation(x, y, sr)
        f.rename(d / "raw" / f.name)
        sf.write(f, aligned_take(y, 0, delay, len(x)), sr, subtype="PCM_24")
        p["delay"], p["aligned"] = delay, "xcorr"
        print(f"{p['id']}: aligned by {delay} samples ({delay / sr * 1000:.1f} ms)")
    delays = [p["delay"] for p in plan["points"] if p.get("aligned")]
    if delays:
        plan["latency_samples"] = int(np.median(delays))
    save(d, plan)


def cmd_take(a) -> None:
    d = pathlib.Path(a.dir)
    plan = load(d)
    if a.id:
        pts = [p for p in plan["points"] if p["id"] == a.id]
        if not pts:
            sys.exit(f"no point {a.id}")
    else:
        pts = [p for p in plan["points"] if p["status"] != "done"][:1]
        if not pts:
            sys.exit("all takes done")
    one_take(plan, d, pts[0], a)


def cmd_run(a) -> None:
    d = pathlib.Path(a.dir)
    plan = load(d)
    for p in [p for p in plan["points"] if p["status"] != "done"]:
        if not one_take(plan, d, p, a):
            break
    done = sum(p["status"] == "done" for p in plan["points"])
    print(f"\n{done}/{len(plan['points'])} takes done")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devices").set_defaults(fn=cmd_devices)
    p = sub.add_parser("align", help="align takes recorded before the calibration clicks existed")
    p.add_argument("--dir", required=True)
    p.set_defaults(fn=cmd_align)
    p = sub.add_parser("meter")
    p.add_argument("--device", required=True)
    p.add_argument("--in", dest="inp", required=True, help="1-based input channels, e.g. 1,2")
    p.add_argument("--out", type=int, help="with --play: output channel for the training signal")
    p.add_argument("--play", action="store_true", help="loop the training signal while metering")
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.set_defaults(fn=cmd_meter)
    for name, fn, needs_dir in [("loopback", cmd_loopback, False), ("take", cmd_take, True), ("run", cmd_run, True)]:
        p = sub.add_parser(name)
        if needs_dir:
            p.add_argument("--dir", required=True)
        p.add_argument("--device", required=True, help="sounddevice name or index (see `devices`)")
        p.add_argument("--out", type=int, required=True, help="1-based output channel")
        p.add_argument("--in", dest="inp", type=int, required=True, help="1-based input channel")
        if name == "take":
            p.add_argument("--id")
        if name in ("take", "run"):
            p.add_argument("--watch", type=lambda v: [int(c) for c in v.split(",")], default=[1],
                           help="extra inputs to meter during takes (default 1, the DI)")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
