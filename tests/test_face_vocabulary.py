"""The panel.face vocabulary (faceplate/vocabulary-v1.json) and its example."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "faceplate" / "vocabulary-v1.json"
EXAMPLE_PATH = ROOT / "faceplate" / "example-face.json"

# Well-known amp/pedal maker names, lower-case. Lives only in this test.
DENYLIST = [
    "fender", "marshall", "vox", "mesa", "boogie", "orange", "peavey", "ampeg",
    "soldano", "friedman", "bogner", "dumble", "matchless", "hiwatt", "supro",
    "gibson", "epiphone", "ibanez", "boss", "mxr", "electro-harmonix", "ehx",
    "jhs", "strymon", "tc electronic", "dunlop", "proco", "klon", "keeley",
    "walrus", "earthquaker", "wampler", "fulltone", "behringer", "digitech",
    "line 6", "neural", "kemper", "fractal", "positive grid", "revv",
    "tolex", "tube screamer", "big muff", "rat", "fuzz face", "plexi", "blackface",
    "silverface", "bassman", "ac30", "top boost", "twin", "deluxe", "princeton", "jcm",
    "super lead",
]

vocab = json.loads(VOCAB_PATH.read_text())
example = json.loads(EXAMPLE_PATH.read_text())


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def test_vocabulary_loads():
    assert vocab["v"] == 1
    assert vocab["limits"] == {
        "maxBytes": 16384, "maxString": 40, "maxText": 12, "maxDecorations": 8,
        "wear": [0, 1], "value": [0, 1]}


def test_every_layout_is_consistent():
    assert vocab["layouts"]
    for lid, lay in vocab["layouts"].items():
        assert lay["kind"] in vocab["kinds"], lid
        lo, hi = lay["knobs"]
        assert lo == hi, lid
        knob_slots = lay["slots"]["knob"]
        if lid == "pedal-0k":
            # reserved GAIN / VOLUME, then the captured gear's printed (fixed) knobs
            assert knob_slots == ["gain", "volume", "p1", "p2", "p3", "p4"]
            assert lay["knobs"] == [0, 0]
            for s in knob_slots:
                assert f"label-{s}" in lay["slots"]["text"], s
        else:
            assert len(knob_slots) == hi, lid
        assert len(lay["slots"]["switch"]) == lay["switches"], lid
        assert lay["switches"] <= (2 if lay["kind"] == "pedal" else 3), lid


def test_layout_ids_cover_v1():
    ids = set(vocab["layouts"])
    assert {f"pedal-{n}k" for n in range(0, 7)} <= ids
    for pos in ("top", "front"):
        assert {f"amp-{pos}-{n}k" for n in range(2, 9)} <= ids


def test_fonts_have_family_and_woff2():
    fonts = vocab["fonts"]
    assert set(fonts) == {"display-1", "display-2", "display-3", "display-4",
                          "label-1", "label-2"}
    for fid, f in fonts.items():
        assert f["family"].strip(), fid
        assert re.fullmatch(r"[a-z0-9-]+\.woff2", f["file"]), fid


def test_amp_part_lists_are_short_and_generic():
    for key in ("ampCoverings", "ampGrilles", "ampPlates", "ampPiping"):
        assert 3 <= len(vocab[key]) <= 8, key
        assert len(set(vocab[key])) == len(vocab[key]), key


def test_example_face_uses_only_vocabulary_ids():
    body = example["body"]
    assert example["v"] == vocab["v"]
    assert body["kind"] in vocab["kinds"]
    assert body["layout"] in vocab["layouts"]
    lay = vocab["layouts"][body["layout"]]
    assert lay["kind"] == body["kind"]
    assert body["shape"] in vocab["shapes"]
    assert body["finish"] in vocab["finishes"]
    for c in example["controls"]:
        assert c["style"] in vocab["knobStyles"]
        assert c["slot"] in lay["slots"]["knob"]
    assert len(example["text"]) <= vocab["limits"]["maxText"]
    for t in example["text"]:
        assert t["font"] in vocab["fonts"]
        assert t["size"] in vocab["textSizes"]
        assert t["slot"] in lay["slots"]["text"]
        assert len(t["string"]) <= vocab["limits"]["maxString"]
    assert len(example["decorations"]) <= vocab["limits"]["maxDecorations"]
    for d in example["decorations"]:
        assert d["shape"] in vocab["decorations"]
        assert d["slot"] in lay["slots"]["decoration"]
    p = example["pedal"]
    assert p["footswitch"] in vocab["footswitches"]
    assert p["led"] in vocab["leds"]
    assert p["jacks"] in vocab["jacks"]
    assert len(EXAMPLE_PATH.read_bytes()) <= vocab["limits"]["maxBytes"]


def test_no_real_names():
    pats = [re.compile(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", re.I)
            for n in DENYLIST]
    for doc in (vocab, example):
        for s in _strings(doc):
            for p in pats:
                assert not p.search(s), (s, p.pattern)


def test_spec_documents_the_face_schema():
    """SPEC.md carries the panel.face schema: every object and the rules a host needs."""
    spec = (ROOT / "SPEC.md").read_text()
    assert "### `panel.face` schema" in spec
    for needle in ("`body`", "`controls[]`", "`text[]`", "`decorations[]`", "`pedal`", "`amp`",
                   "`value`", "`wear`", "code points", "`p1`", "exactly once", "16384",
                   "at least two strings", "a stock look"):
        assert needle in spec, needle
    for key in vocab["limits"]:
        assert f"`limits.{key}`" in spec, key
