# Capturing your amp, step by step

This guide takes you from an amp on your desk to a `.nam` file that plays every
setting of its knobs. Plan on an afternoon to set up and three or four hours at
the amp. If you use [Claude Code](https://claude.com/claude-code), open this
repo in it and say "help me capture my amp": the skills in `.claude/skills`
walk you through the same steps and check each one.

## What you need

- The amp, a load box (or a speaker cab with a mic, or a DI out), and a reamp
  box to send a line-level signal into the amp's input.
- An audio interface with at least one output and two inputs.
- A Mac with Apple silicon or a computer with an NVIDIA GPU, for training.
- Python 3.10+, this repo (`pip install -r requirements.txt`), and a
  [PANAMA](https://github.com/ETH-DISCO/PANAMA) checkout installed in its own
  virtual environment (`pip install -e .`).
- NAM's v3 training signal, `input.wav` (from the
  [NAM trainer](https://github.com/sdatkinson/neural-amp-modeler)).

```sh
export NAM_INPUT=/path/to/input.wav
export PANAMA_DIR=/path/to/PANAMA
```

## 1. Choose the knobs

Pick the controls you want the model to learn, in the order they sit on the
amp: Gain, Bass, Middle, Treble, Presence, Master... Up to eight. A two-way
switch (Bright, a pull boost, half/full power) counts as one.

Start with the controls that change the sound most. You can add more later
without re-recording (step 6).

Every other control stays where it is for the whole session. Write those
positions down; you will need them again if you come back another day.

## 2. Make the plan

```sh
python -m parametric_nam.capture_plan new --dir sessions/myamp \
    --knobs Gain,Bass,Middle,Treble --grid 2 --random 40 \
    --look plexi --author "Your Name"
```

This writes `sessions/myamp/plan.json`: a list of knob settings, each one a
take. `--grid 2` records every knob at both ends, `--random 40` adds settings
spread across the whole range. Four knobs at these numbers is 56 takes, about
three minutes each.

`--look` picks how the capture's amp looks in the Coro NAM Player: `generic`,
`blackface`, `tweed`, `vox`, `mesa` or `plexi`. `--author` puts your name on
it. Both can be changed later.

## 3. Wire up and check the levels

Interface output into the reamp box, reamp box into the amp, amp into the load
box, load box into an interface input.

Quit your DAW and anything else using the interface first: it can only be open
in one program.

```sh
python -m parametric_nam.capture_record devices
python -m parametric_nam.capture_record loopback --device "My Interface" --out 3 --in 2
python -m parametric_nam.capture_record meter --device "My Interface" --in 1,2 --out 3 --play
```

`loopback` measures the round trip from output to input. `meter --play` loops
the training signal through the amp while it shows the level: set the
interface's input gain so the loudest setting you'll record peaks around
−6 dBFS and never clips. Don't touch the interface's gains after this.

## 4. Record

```sh
python -m parametric_nam.capture_record run --dir sessions/myamp \
    --device "My Interface" --out 3 --in 2
```

For each take it shows the knob positions, waits while you set them on the amp,
plays the signal and saves `takes/NNNN.wav`. Set each knob carefully: the model
learns exactly what you dial.

You can stop at any time. Running the same command again picks up at the next
unfinished take. If one went wrong (a knob was off, something buzzed),
re-record just that one:

```sh
python -m parametric_nam.capture_record take --dir sessions/myamp --id 0017 \
    --device "My Interface" --out 3 --in 2
```

**Coming back another day?** Put every control back where it was, then record
the first take's setting again as a new take and compare levels. Copy the knob
values of take 0001 from `plan.json` (its `"g"` list):

```sh
python -m parametric_nam.capture_plan add --dir sessions/myamp --g 0.5,0.5,0.5,0.5
# prints the new take's id, e.g. "added 0057"
python -m parametric_nam.capture_record take --dir sessions/myamp --id 0057 \
    --device "My Interface" --out 3 --in 2
```

The recorder prints each take's level (rms dBFS). The new take should match
take 0001's within about half a dB. If it doesn't, fix the levels before going on.

## 5. Train and export

```sh
scripts/train.sh sessions/myamp 30
```

This trains for 30 passes over your takes (about 8 minutes a pass on an
M-series Mac for a 56-take session, so about four hours), picks the pass with the lowest error on takes it
held back, and writes `sessions/myamp/myamp.nam`.

The error it prints is ESR: 0.01 to 0.02 is a great single-setting capture;
a whole knob space lands higher, around 0.05 to 0.09. Listen to the result at a
few settings you recorded and a few you didn't.

To change the look, labels or author without retraining, convert again:

```sh
python -m parametric_nam.convert_panama --ckpt <the checkpoint train.sh printed> \
    --model-config sessions/myamp/model.json --plan sessions/myamp/plan.json \
    --out sessions/myamp/myamp.nam --look tweed --label Gain=VOLUME --author "Your Name"
```

## 6. Add a knob later

```sh
python -m parametric_nam.capture_plan extend --dir sessions/myamp \
    --knobs Presence --defaults 0.5
```

The takes you already have count as Presence at 5 (`--defaults 0.5`), so only
the new settings need recording. Then record and train again.

## 7. Play and share it

Load the `.nam` in the free [Coro NAM Player](https://www.murphyresearch.ai/nam-player).
Your knobs appear on the amp in the look you chose. Share the file: anyone with
the player sees the same amp, your labels and your name.

## When something goes wrong

- **"device unavailable" or no sound:** another program has the interface. Quit it.
- **Takes sound fine but the model is poor:** check the round trip didn't change
  mid-session (`loopback`, or `parametric_nam/loop_jitter.py`). A one-sample
  shift between takes ruins a high-gain capture.
- **Quiet or nearly silent settings are worst:** add a few more takes in that
  corner with `capture_plan add --dir sessions/myamp --g 0.1,0.5,0.5,0.5`.
