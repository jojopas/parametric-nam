# ParametricWaveNet `.nam` format

Version 1. A parametric capture is one neural model of an amplifier that plays
every setting of the knobs it was captured with. This document defines how such
a model is stored so that any NAM-based host can load it.

## Design in one paragraph

A parametric capture is a standard NAM WaveNet whose conditioning input is
widened from one row (the audio) to `1 + N` rows: the audio plus `N` knob
values. The file reuses NAM's WaveNet config and weight layout unchanged, so a
host built on [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore)
parses it with NAM's own parser and only supplies the knob rows. It carries its
own architecture name so that a host which does not know about knobs refuses
the file instead of running it with uninitialised conditioning rows.

## File

UTF-8 JSON with the same top-level keys as a NAM WaveNet `.nam` file:

| key | type | meaning |
|---|---|---|
| `version` | string | NAM file version the layout follows, e.g. `"0.5.4"` |
| `architecture` | string | must be `"ParametricWaveNet"` |
| `config` | object | a NAM WaveNet config plus `knobs` (below) |
| `weights` | array of numbers | NAM WaveNet weight order for the sizes in `config` |
| `sample_rate` | number | sample rate the model was trained at |
| `metadata` | object | as in NAM (`name`, `loudness`, `input_level_dbu`, ...) |

### `config.knobs`

An array of 1 to 8 objects, in conditioning-row order:

| key | required | meaning |
|---|---|---|
| `name` | yes | label to show, e.g. `"Gain"` |
| `default` | no | value 0..1 to start at; 0.5 if absent |
| `labels` | no | for a stepped switch: one label per position, an array of at least two strings. Position `i` of `n` is the value `i / (n - 1)`, e.g. `["half", "full"]` is 0 for half and 1 for full. A knob whose `labels` is not an array of at least two strings is a plain knob |

Hosts ignore keys they do not know.

### `metadata` additions

All optional. A host that does not draw panels ignores them.

| key | type | meaning |
|---|---|---|
| `modeled_by` | string | who made the capture |
| `author_note` | string | a short note from the author (the amp, the cab IR they used, ...) |
| `panel.look` | string | the amp look a host draws the capture in: `generic`, `blackface`, `tweed`, `vox`, `mesa` or `plexi`. A host draws a look it does not know in its default look |
| `panel.labels` | object | display label by knob name, e.g. `{"Gain": "VOLUME"}`. Knobs without one show their name |
| `panel.kind` | string | `"pedal"` or `"amp"`: what kind of gear the capture is. A host with no valid `panel.face` draws its default pedal template for `"pedal"`. Any other value is ignored |
| `panel.face` | object | the faceplate as pure data (no images, URLs, HTML, CSS, script or free-form paths). Schema below; an example is in [`faceplate/example-face.json`](faceplate/example-face.json) |

Panel order is `config.knobs` order. A stepped knob (`labels` of at least two strings) is drawn as a switch.

### `panel.face` schema

Version 1. Every id comes from [`faceplate/vocabulary-v1.json`](faceplate/vocabulary-v1.json)
(the "vocabulary"); the list each id is checked against is named in the tables below. The
vocabulary's `limits` hold the numeric bounds: `limits.maxBytes` (16384), `limits.maxString`
(40), `limits.maxText` (12), `limits.maxDecorations` (8), `limits.wear` ([0, 1]) and
`limits.value` ([0, 1]). Vocabulary v1 is frozen; new ids go to `v: 2`.

**Value types.**

- A colour is a string `#rrggbb` (six hex digits, either case).
- A number is a finite JSON number inside its stated range, ends included. A numeric string is
  not a number.
- A string is a JSON string of at most `limits.maxString` (40) characters, counted in Unicode
  code points (an emoji is one, a lone surrogate is one), with no control characters (U+0000 to
  U+001F, U+007F).
- An object is a plain JSON object. Every key listed as required must be present, and any key
  not listed rejects the face.
- The whole face, serialised as compact JSON (no whitespace) and encoded as UTF-8, is at most
  `limits.maxBytes` (16384) bytes. Hosts check the size first.

**The face** (top level)

| key | required | type |
|---|---|---|
| `v` | yes | the number `1` |
| `body` | yes | object, below |
| `controls` | yes | array of control objects, at most (knob slots + switch slots) of the layout |
| `template` | no | string of 1 to 40 characters from `A-Z a-z 0-9 . _ -`: the template the face started from (informational) |
| `text` | no | array of at most `limits.maxText` (12) text objects |
| `decorations` | no | array of at most `limits.maxDecorations` (8) decoration objects |
| `pedal` | when `body.kind` is `"pedal"`; rejected otherwise | object, below |
| `amp` | when `body.kind` is `"amp"`; rejected otherwise | object, below |

**`body`**

| key | required | type |
|---|---|---|
| `kind` | yes | `kinds`: `"pedal"` or `"amp"` |
| `layout` | yes | a key of `layouts`; that layout's `kind` must equal `body.kind` |
| `color` | yes | colour. The enclosure colour of a pedal. Required on amp faces too, but has no effect on them in v1 (an amp is dressed by `amp`) |
| `shape` | no, pedals only | `shapes`. Rejected on an amp face |
| `finish` | no, pedals only | `finishes`. Rejected on an amp face |
| `wear` | no | number in `limits.wear` ([0, 1]); 0 is new |
| `accent` | no | colour |

**`controls[]`** (one object per drawn knob or switch)

| key | required | type |
|---|---|---|
| `slot` | yes | a slot of the layout's `slots.knob` or `slots.switch` (see binding rules); each slot at most once |
| `style` | yes | `knobStyles` |
| `cap` | yes | colour |
| `pointer` | yes | colour |
| `knob` | no | string: the name of one of the capture's knobs (binding rules). Not with `host` |
| `host` | no | `"gain"` or `"volume"`: the host's own input gain or output volume, drawn as a knob of the face (binding rules). Not with `knob` |
| `value` | no, fixed controls only | number in `limits.value` ([0, 1]): where a fixed knob points (0 fully left, 1 fully right). Default 0.5. Rejected on a control with `knob` or `host` |

A control with neither `knob` nor `host` is printed fixed.

**`text[]`**

| key | required | type |
|---|---|---|
| `slot` | yes | a slot of the layout's `slots.text`; each slot at most once. `label-<slot>` labels the control in `<slot>` |
| `string` | yes | string |
| `font` | yes | a key of `fonts` |
| `size` | yes | `textSizes` (a host may cap the size a slot prints at) |
| `color` | yes | colour |

**`decorations[]`**

| key | required | type |
|---|---|---|
| `shape` | yes | `decorations` |
| `slot` | yes | a slot of the layout's `slots.decoration` (a slot may hold more than one) |
| `color` | yes | colour |

**`pedal`**: `footswitch` (`footswitches`), `led` (`leds`), `jacks` (`jacks`). All required.

**`amp`**: `covering` (`ampCoverings`), `grille` (`ampGrilles`), `plate` (`ampPlates`),
`piping` (`ampPiping`), `lamp` (`ampLamps`), `switch` (`ampSwitches`). All required.

**Binding rules.** A host validates the face against the capture's knob list,
`config.knobs` (a plain NAM file has none: the list is empty).

1. A knob is *stepped* when its `labels` is an array of at least two strings; every other knob
   is *plain*. `labels: []` and `labels: ["A"]` are plain. A knob list with a duplicate name,
   or with `labels` that is not an array of strings, rejects the face.
2. **Live knobs.** The capture's knobs are live. So are the host's own controls the face
   carries with `host`: `"gain"` (input gain) and `"volume"` (output volume), both plain, each at
   most once. A capture holds its volume fixed while it is made, so the volume is not one of its
   knobs; `host: "volume"` puts the player's output volume on the face where the gear had its
   volume knob. A static capture (empty list) must carry both `host` controls.
3. **Every captured knob exactly once.** Each of the capture's knobs is named by exactly one
   control's `knob`, and no control names anything else. Each slot holds at most one control.
4. **Slot types.** A plain knob goes in a `slots.knob` slot, a stepped knob in a
   `slots.switch` slot. A `host` control and a fixed control go in a `slots.knob` slot. A knob
   slot with no control is left empty.
5. **Layout fit.** With a non-empty list, the number of plain knobs plus `host` controls must be within the
   layout's `knobs` range (`[lo, hi]`; in v1 `lo == hi`, the layout's knob-slot count, so
   there are no spare knob slots) and the number of stepped knobs at most its switch-slot count.
6. **Static pedals** use `pedal-0k`. `host: "gain"` takes its reserved `gain` slot and
   `host: "volume"` its reserved `volume` slot. The captured gear's own knobs are printed fixed in `p1` to `p4`,
   with their captured positions as `value` and their names in `label-p1` to `label-p4`.
7. **Static amps** use any amp layout. The two `host` controls go in any two knob slots; the
   gear's other knobs are printed fixed, with `value`, in the spare knob slots.
8. A host drives switch slots with the stepped values `i / (n - 1)` (`n` = number of labels).
9. A host maps a `host` control onto its own control over that control's full range: knob 0 is
   the control's minimum, 1 its maximum. A face without `host: "volume"` leaves the host to show
   its output volume elsewhere.

**Reading it.** A host reads `panel.face` strictly: an unknown key, unknown id, out-of-range
value, wrong type, broken binding rule, oversize face or unknown `v` rejects the whole face,
with no partial rendering, and a validator never throws. Fallback, first row that applies:

| the file has | the host draws |
|---|---|
| a valid `panel.face`, and the user has not picked a stock look | the face |
| an invalid `panel.face` | as if it had none (rows below) |
| a valid `panel.face`, and the user picked a stock look in the host | that stock look |
| `panel.kind: "pedal"`, no valid face, and the user picked a stock look | that stock look |
| `panel.kind: "pedal"`, no valid face, no user pick | the host's default pedal template, labelled with the knobs |
| `panel.look`, no valid face | that stock amp head |
| none of these | the host's default amp head |

A host's default pedal template prints no maker name unless the file supplies one.
Files without `panel.face` or `panel.kind` draw as before.

### Constraints

With `N = len(config.knobs)`:

- every entry of `config.layers` has `condition_size == 1 + N`
- the first entry of `config.layers` has `input_size == 1 + N`
- there is no `condition_dsp`

A loader should reject a file that breaks any of these, and should report why.

## Semantics

- Conditioning row 0 is the audio input, exactly as in a plain WaveNet.
- Rows `1..N` are the knob values, each in 0..1, in `config.knobs` order.
  A knob value `v` means the physical control at `v` of its travel: 0.5 on a
  0-10 dial is 5.
- A host holds each knob constant across a processing block. It may change
  knobs between blocks. Smoothing a knob that jumps is up to the host; the
  model responds to any value in 0..1, not only the ones that were recorded.
- Everything else, including `metadata.loudness` and `input_level_dbu`, means
  what it means for a plain NAM capture.

## Loading in a NeuralAmpModelerCore host

Subclass `nam::wavenet::WaveNet`, build it from `nam::wavenet::parse_config_json`,
and override two hooks:

- `_get_condition_dim()` returns `NumInputChannels() + N`
- `_set_condition_array()` writes the audio to row 0 and the knob values to rows `1..N`

[`cpp/ParametricWaveNet.h`](cpp/ParametricWaveNet.h) is a complete reference
loader, about 160 lines, with no dependencies beyond NAM core.

## Relation to PANAMA

[PANAMA](https://github.com/ETH-DISCO/PANAMA) (arXiv:2509.26564) trains
WaveNets that take a global knob vector `g` through a per-layer linear term:

    z = conv(x) + input_mixer(audio) + global_mixer(g)

NAM core computes `z = conv(x) + input_mixin(condition)`. With
`condition = [audio, g]`, `input_mixin = [input_mixer | global_mixer.weight]`
and `global_mixer.bias` added to `conv.bias`, the two are identical. The first
block's rechannel layer gets zero columns for the knob rows. That fold is what
[`parametric_nam/convert_panama.py`](parametric_nam/convert_panama.py) does, so
a PANAMA checkpoint becomes a file in this format with no loss.
