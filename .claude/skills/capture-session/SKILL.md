---
name: capture-session
description: Use when someone wants to capture their amp (or pedal) as a parametric NAM, plan or record a capture session, check levels, resume a session, or re-record a take. Walks them from choosing knobs to a finished set of takes, one step at a time.
---

# Capture session

Guide a person through recording their amp for a parametric capture. Most
people doing this are guitarists, not programmers: explain each step in plain
words, run the commands for them, and read the output back in plain words.
`docs/GUIDE.md` is the same procedure for people without Claude; keep to it.

## Before anything

Check, and fix with them, in this order:
1. `pip install -r requirements.txt` works, `NAM_INPUT` points at NAM's v3
   `input.wav`, `PANAMA_DIR` at a PANAMA checkout with its own venv. Use that
   venv's python for every command that records or trains
   (`$PANAMA_DIR/.venv/bin/python`); the system python often lacks `soundfile`.
2. The wiring: interface out → reamp box → amp → load box (or mic'd cab / DI) →
   interface in. Ask which output and input numbers they used.
3. The DAW and any other audio app are closed. The interface can only be open
   in one program; "device unavailable" means something else has it.

## 1. Knobs

Ask what amp it is and which controls they want to capture, left to right as
on the panel. Up to 8. Two-way switches count as one each. Recommend starting
with the 3 to 5 controls that change the sound most; more can be added later
without re-recording (`add-a-knob` skill). Ask them to write down where every
other control sits; it must stay there for the whole capture.

## 2. Plan

Suggest a size and say what it costs at the amp (about 3 minutes a take):

| knobs | command flags | takes | time at the amp |
|---|---|---|---|
| 2-3 | `--grid 3 --random 24` | 33-51 | 2-2.5 h |
| 4 | `--grid 2 --random 40` | 56 | 3 h |
| 5-6 | `--grid 2 --random 50` | 82-114 | 4-6 h, over two days |

Ask which amp look they want (`generic`, `blackface`, `tweed`, `vox`, `mesa`,
`plexi`) and the name to credit, then:

```sh
python -m parametric_nam.capture_plan new --dir sessions/<amp> --knobs <Knob1,Knob2,...> \
    --grid <n> --random <n> --look <look> --author "<name>"
```

## 3. Levels

```sh
python -m parametric_nam.capture_record devices
python -m parametric_nam.capture_record loopback --device "<name>" --out <o> --in <i>
python -m parametric_nam.capture_record meter --device "<name>" --in <i> --out <o> --play
```

Ask them to set the amp to its loudest planned setting while `meter --play`
runs and set the interface's input gain so peaks sit near −6 dBFS, never
clipping. Tell them not to touch interface gains again until the session ends.
Note the `loopback` delay; if it changes later, takes from before and after
won't line up.

## 4. Record

```sh
python -m parametric_nam.capture_record run --dir sessions/<amp> --device "<name>" --out <o> --in <i>
```

The person drives this one: it prompts for each setting and waits for them.
Tell them to dial each knob carefully, that they can stop any time and the same
command resumes at the next unfinished take, and to tell you if a take went
wrong. Re-record one with `take --dir … --id <NNNN> …`.

Watch the level each take prints. A take far quieter or louder than its
neighbours at similar settings usually means a knob was mis-set: offer to redo it.

Coming back another day: same control positions, then add a repeat of take
0001's setting (`capture_plan add --g <its g values>`, then `take --id <new id>`)
and compare its rms with take 0001. Within about 0.5 dB is fine. Further off:
find out what moved before recording more.

## Done

`capture_plan status --dir sessions/<amp>` shows every take done. Hand over to
the `train-and-export` skill.
