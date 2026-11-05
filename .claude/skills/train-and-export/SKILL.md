---
name: train-and-export
description: Use when a capture session's takes are recorded and someone wants to train the parametric model, check how good it is, or export/re-export the .nam with a look, knob labels or author credit.
---

# Train and export

Turn a finished capture session into a `.nam` that plays every knob setting.
Explain progress in plain words; most people doing this aren't programmers.

## Before training

- `python -m parametric_nam.capture_plan status --dir sessions/<amp>`: every
  take done. If not, finish recording first (`capture-session`).
- `PANAMA_DIR` set; Apple silicon trains with the default
  `configs/learning_mps.json`, NVIDIA needs `LEARNING=` pointing at a CUDA
  learning config.
- Tell them how long it takes: about 8 minutes a pass for 56 takes on an
  M-series Mac, scaling with the number of takes; 30 passes is the default.
  Training runs in the background; the computer can't record audio meanwhile
  without slowing it.

## Train

```sh
scripts/train.sh sessions/<amp> 30
```

Run it in the background and check on it. It writes checkpoints every two
passes named with their validation error (`ESR=…`) on takes held out of
training, then converts the best one to `sessions/<amp>/<name>.nam`.

Read the numbers to them plainly:
- ESR 0.01–0.02 is a great single-setting NAM. A whole knob space lands
  higher: 0.05–0.09 is good, above 0.12 something is wrong.
- Early passes bounce; judge the trend, not one pass.
- Something wrong usually means takes don't line up (the round trip changed),
  a mis-set knob in some takes, or levels changed between days. Help them
  find which takes stand out before retraining.

## Listen

Have them load the `.nam` in the Coro NAM Player
(https://www.murphyresearch.ai/nam-player) and compare against the amp at a
few recorded settings and a few in between. Their ears decide.

## Export again (no retraining)

To change the look, labels or credit, convert the same checkpoint:

```sh
$PANAMA_DIR/.venv/bin/python -m parametric_nam.convert_panama --ckpt <checkpoint> \
    --model-config sessions/<amp>/model.json --plan sessions/<amp>/plan.json \
    --out sessions/<amp>/<name>.nam --look <look> --label <Knob>=<LABEL> \
    --author "<name>" --note "<short note: the amp, the cab IR they like>"
```

Looks: `generic`, `blackface`, `tweed`, `vox`, `mesa`, `plexi`. Labels only
change what the panel says, not the sound.

## Share

The file carries its look, labels and author. Anyone with the Coro NAM Player
sees the same amp. Suggest a note (`--note`) naming the amp and any mods.
