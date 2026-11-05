---
name: add-a-knob
description: Use when someone has a finished parametric capture and wants to add another knob or a two-way switch (Presence, Master, Bright, half/full power...) without re-recording the takes they already have.
---

# Add a knob or switch

The takes already recorded count as having the new control where it sat while
they were recorded. Only new settings need recording, then the model is
trained again.

## 1. Where was it?

Ask where the new control sat during the original session (e.g. Presence at 5
→ `0.5`; a switch: which position). Wrong here means the model learns the
wrong thing, so check their notes from the first session.

## 2. Extend the plan

A knob:

```sh
python -m parametric_nam.capture_plan extend --dir sessions/<amp> \
    --knobs Presence --defaults 0.5
```

A switch (finished takes are at position 1, the second label):

```sh
python -m parametric_nam.capture_plan extend --dir sessions/<amp> \
    --switch Power --switch-labels half,full
```

`extend` keeps every finished take and adds new settings: a small grid with
the old knobs centred plus random settings over everything. Tell them how many
takes were added and the time at the amp (about 3 minutes each).

## 3. Same amp, same levels

Every other control back where it was, interface gains untouched since the
first session. Start with the drift check from `capture-session` (repeat take
0001's setting and compare levels).

## 4. Record and train

Record with `capture_record run` (it skips finished takes), then train with
`train-and-export`. Training starts fresh by default; expect the new control
to be weakest where few takes cover it.
