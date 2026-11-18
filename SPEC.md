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
| `labels` | no | for a stepped switch: one label per position. Position `i` of `n` is the value `i / (n - 1)`, e.g. `["half", "full"]` is 0 for half and 1 for full |

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
| `panel.face` | object | the faceplate as pure data (no images, URLs, HTML, CSS, script or free-form paths). Schema version `v` is 1; every id (layout, shape, finish, knob style, font, ...) comes from [`faceplate/vocabulary-v1.json`](faceplate/vocabulary-v1.json), colours are `#rrggbb`, strings are at most 40 characters, at most 12 text items and 8 decorations, and the serialised face is at most 16 KB. An example is in [`faceplate/example-face.json`](faceplate/example-face.json) |

Panel order is `config.knobs` order. A knob with `labels` is drawn as a switch.

A host reads `panel.face` strictly: an unknown key, unknown id, out-of-range value, wrong type or unknown `v` rejects the whole face, with no partial rendering, and a validator never throws. Fallback:

| the file has | the host draws |
|---|---|
| a valid `panel.face` | the face |
| an invalid `panel.face` | as if it had none (rows below) |
| `panel.kind: "pedal"`, no valid face | the host's default pedal template, labelled with the knobs |
| `panel.look`, no valid face | that stock amp head |
| none of these | the host's default amp head |

Vocabulary v1 is frozen; new ids go to `v: 2`. Files without `panel.face` or `panel.kind` draw as before.

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
