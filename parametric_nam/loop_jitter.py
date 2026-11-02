"""Does the interface's play/record offset change between stream starts?
Patch the playback output straight into the record input with a cable, then:
  loop_jitter.py --device "My Interface" --out 3 --in 2 --runs 8
Each run opens a fresh stream, plays 5 clicks and reports the loop delay to 1/16 sample."""
import argparse, sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from parametric_nam.capture_record import playrec
from scipy.signal import resample_poly

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True); ap.add_argument("--out", type=int, required=True)
    ap.add_argument("--in", dest="inp", type=int, required=True); ap.add_argument("--runs", type=int, default=8)
    a = ap.parse_args(); sr = 48000; U = 16
    x = np.zeros(sr * 4, dtype=np.float32); pos = [int((0.5 + 0.5 * k) * sr) for k in range(5)]
    click = (0.5 * np.sin(np.pi * np.arange(48) / 48)).astype(np.float32)
    for p in pos: x[p:p + 48] = click
    ref = resample_poly(click, U, 1); results = []
    for r in range(a.runs):
        y = playrec(x, sr, a.device, a.out, a.inp, watch=(), title=f"run {r + 1}")
        y = y[:, 0] if y.ndim > 1 else y
        noise = np.sqrt(np.mean(y[: int(0.3 * sr)] ** 2)); thr = max(20 * noise, 0.02)
        ds = []
        for p in pos:
            seg = y[p:p + sr]  # up to 1 s after the click
            i = int(np.argmax(np.abs(seg) > thr))
            if not (np.abs(seg) > thr).any(): continue
            w = y[p + i - 24: p + i + 72]; wu = resample_poly(w, U, 1)
            cc = np.correlate(wu, ref, "valid"); j = int(np.argmax(cc))
            ds.append(i - 24 + j / U)
        if not ds: print(f"run {r + 1}: no click heard (peak {np.abs(y).max():.3f})"); continue
        results.append(np.median(ds)); print(f"run {r + 1}: delay {np.median(ds):.2f} samples  (clicks within run: spread {max(ds) - min(ds):.2f})")
    if len(results) > 1:
        print(f"\nacross {len(results)} stream starts: min {min(results):.2f}  max {max(results):.2f}  spread {max(results) - min(results):.2f} samples")
        print("-> deterministic to the sample" if max(results) - min(results) < 0.5 else "-> start offset jitters between streams")

if __name__ == "__main__": main()
