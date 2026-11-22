# parametric-nam

Parametric captures for [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler):
one model of an amplifier that plays every setting of its knobs, not one
snapshot per setting.

This repo has three things:

1. **[A file format](SPEC.md)**, `ParametricWaveNet`. A standard NAM WaveNet
   `.nam` whose conditioning input carries the knob values next to the audio.
   NAM core runs it unmodified. A host only has to feed the knob rows. An
   optional `knob_scale` lets the knobs scale each layer too (version 1.1).
2. **A reference loader** in [`cpp/`](cpp/), about 300 lines on top of
   [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore),
   `knob_scale` included.
3. **Capture and training tools** in [`parametric_nam/`](parametric_nam/):
   plan the knob settings, record the amp at each one, train with
   [PANAMA](https://github.com/ETH-DISCO/PANAMA), and convert the result to
   the format.

> **Status: experimental.** The format and the conversion are exact: a
> converted model nulls against PyTorch. Model quality is still being tuned.
> Our first amp, a 4-knob Marshall Origin 20 capture, reaches an ESR of about
> 0.09 on knob settings held out of training. A good single-setting NAM
> capture is around 0.01 to 0.02. See [What we learned](#what-we-learned).

## Get the player

[**Coro NAM Player**](https://www.murphyresearch.ai/nam-player) is a free
plugin (AU, VST3 and standalone; macOS 12+, Apple silicon) that plays
parametric captures: it reads the knobs in the file and puts them on an amp
head. It ships with a six-knob capture of our Origin 20 and a set of cab IRs,
so you can hear one before you make one. Plain NAM captures play too. The
download is free with a Murphy Research account.

## Capture without the command line

[**Coro Capture**](https://www.murphyresearch.ai/coro-capture) is a free Mac
app (macOS 12+, Apple silicon) that does everything in this repo for you: it
plans the knob settings, records each take through your audio interface, checks
every take as it lands, trains on your Mac (the trainer is built in: no Python
to install), and exports a `ParametricWaveNet` `.nam` that opens in Coro NAM
Player. It makes single-setting captures too. The tools below are the same
method, by hand.

## How it works

PANAMA trains a WaveNet that takes a vector of knob values alongside the audio.
You record the amp many times, each at a different knob setting, all from the
same dry input signal. The model learns how the knobs change the sound, so it
can also play settings that were never recorded. The format stores that model
so that NAM-based players can load it. [SPEC.md](SPEC.md) has the details and
the weight fold that makes PANAMA's layers match NAM's.

## Quick start

New to this? [docs/GUIDE.md](docs/GUIDE.md) walks through a whole capture step
by step. With [Claude Code](https://claude.com/claude-code), open this repo and
ask it to help you capture your amp: the skills in [`.claude/skills`](.claude/skills)
guide you through planning, levels, recording, training and export.


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
    --knobs Gain,Bass,Middle,Treble --grid 2 --random 40 \
    --look blackface --author "Your Name"

# 2. check levels, then record every setting (prompts you to dial each one)
python -m parametric_nam.capture_record devices
python -m parametric_nam.capture_record meter --device "My Interface" --in 1,2 --out 3 --play
python -m parametric_nam.capture_record run --dir sessions/myamp \
    --device "My Interface" --out 3 --in 2

# 3. train and convert
scripts/train.sh sessions/myamp 30
# -> sessions/myamp/myamp.nam
```

### Plain files for other players

Most NAM players, hardware units included, play one setting per file. To take
settings of a parametric capture to one of them, write plain `.nam` files:

```sh
python -m parametric_nam.snapshot sessions/myamp/myamp.nam out/ \
    --at 0.3 0.5 0.5 0.5 --at 0.8 0.5 0.5 0.5
# -> out/<name> - Gain 3 Bass 5 Middle 5 Treble 5.nam, ...
```

Each file sounds exactly like the parametric capture at that setting (it nulls
against it) and loads in any NAM player, but it has no knobs.

The look and author go into the .nam, so a shared capture opens in the right
amp look in the Coro NAM Player. Change them at convert time with
`convert_panama.py --look tweed --label Gain=VOLUME --author "..." --note "..."`;
without a look the converter asks (or writes `generic` when not run in a
terminal).

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

## Apps

- [Coro Capture](https://www.murphyresearch.ai/coro-capture) by Murphy
  Research (free, macOS) records and trains parametric captures with these
  tools built in.

## Hosts

- [Coro NAM Player](https://www.murphyresearch.ai/nam-player) by Murphy
  Research (free) loads `ParametricWaveNet` files and shows their knobs on an
  amp head, in the look the capture names.

To add support to another NAM-based host, see [SPEC.md](SPEC.md) and
[`cpp/ParametricWaveNet.h`](cpp/ParametricWaveNet.h).

## Tests

```sh
python -m pytest tests
```

## Contact and contributing

- Questions, captures you'd like to share, or ideas: joe@murphyresearch.ai
- Bugs and feature requests: [open an issue](https://github.com/jojopas/parametric-nam/issues).
- Pull requests are welcome. Run `python -m pytest tests` before you send one.

## Credits and licence

MIT. See [LICENSE](LICENSE). The training method is PANAMA's. The model and
file layout are NAM's. [NOTICE.md](NOTICE.md) has the credits.
