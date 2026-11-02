# parametric-nam

Parametric captures for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler):
one model of an amplifier that plays every setting of its knobs, not one
snapshot per setting.

This repo has three things:

1. **[A file format](SPEC.md)**, `ParametricWaveNet`. A standard NAM WaveNet
   `.nam` whose conditioning input carries the knob values next to the audio.
   NAM core runs it unmodified. A host only has to feed the knob rows.
2. **A reference loader** in [`cpp/`](cpp/), about 160 lines on top of
   [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore).
3. **Capture and training tools** in [`parametric_nam/`](parametric_nam/):
   plan the knob settings, record the amp at each one, train with
   [PANAMA](https://github.com/ETH-DISCO/PANAMA), and convert the result to
   the format.

> **Status: experimental.** The format and the conversion are exact: a
> converted model nulls against PyTorch. Model quality is still being tuned.
> Our first amp, a 4-knob Marshall Origin 20 capture, reaches an ESR of about
> 0.09 on knob settings held out of training. A good single-setting NAM
> capture is around 0.01 to 0.02. See [What we learned](#what-we-learned).

## How it works

PANAMA trains a WaveNet that takes a vector of knob values alongside the audio.
You record the amp many times, each at a different knob setting, all from the
same dry input signal. The model learns how the knobs change the sound, so it
can also play settings that were never recorded. The format stores that model
so that NAM-based players can load it. [SPEC.md](SPEC.md) has the details and
the weight fold that makes PANAMA's layers match NAM's.

## Quick start

Requirements: Python 3.10+, an audio interface, a reamp box, a load box or DI
for the amp's output, and a [PANAMA](https://github.com/ETH-DISCO/PANAMA)
checkout installed into its own virtual environment (`pip install -e .`).
Training runs on CUDA or Apple Silicon.

```sh
pip install -r requirements.txt
export NAM_INPUT=/path/to/input.wav      # the NAM v3 training signal
export PANAMA_DIR=/path/to/PANAMA

# 1. plan: a 2-level grid over 4 knobs plus 40 stratified random settings
python -m parametric_nam.capture_plan new --dir sessions/myamp \
    --knobs Gain,Bass,Middle,Treble --grid 2 --random 40

# 2. check levels, then record every setting (prompts you to dial each one)
python -m parametric_nam.capture_record devices
python -m parametric_nam.capture_record meter --device "My Interface" --in 1,2 --out 3 --play
python -m parametric_nam.capture_record run --dir sessions/myamp \
    --device "My Interface" --out 3 --in 2

# 3. train and convert
scripts/train.sh sessions/myamp 30
# -> sessions/myamp/myamp.nam
```

Each take is about 3 minutes of audio. A 56-setting session takes about three
hours at the amp.

To add knobs or a switch to a finished session without re-recording it, use
`capture_plan extend`. The finished takes keep their recordings and count as
having the new knobs at their default positions:

```sh
python -m parametric_nam.capture_plan extend --dir sessions/myamp \
    --knobs Presence --defaults 0.5 --switch Power --switch-labels half,full
```

## What we learned

From our first capture, a Marshall Origin 20 with a plexi mod, four knobs, 56
takes over two days:

- **Alignment has to be exact to the sample.** A one-sample timing error
  between takes costs about 0.1 ESR on a high-gain amp, as much as the whole
  remaining model error. The recorder prepends calibration clicks and aligns
  each take to them. `parametric_nam/loop_jitter.py` tests whether your
  interface's round trip is constant across stream starts. Run it with a cable
  from output to input before a session.
- **Check drift between days.** Re-record one setting at the start of each
  session and compare it with the first day's take. Ours matched within 0.1 dB.
- **The standard NAM network is too small for a whole knob space.** At 16
  channels it plateaued at about 0.10 ESR. Doubling to 32/16 channels
  ([`configs/model_wide.json`](configs/model_wide.json)) did better, and is the
  default here.
- **Low-gain corners need extra coverage.** Settings where the amp is nearly
  silent, or where one channel carries all the sound, had the highest error.

## Hosts

- [Coro](https://murphyresearch.ai) by Murphy Research loads `ParametricWaveNet`
  files and shows their knobs.

To add support to another NAM-based host, see [SPEC.md](SPEC.md) and
[`cpp/ParametricWaveNet.h`](cpp/ParametricWaveNet.h).

## Tests

```sh
python -m pytest tests
```

## Credits and licence

MIT. See [LICENSE](LICENSE). The training method is PANAMA's. The model and
file layout are NAM's. [NOTICE.md](NOTICE.md) has the credits.
