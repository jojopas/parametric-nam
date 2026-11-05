# parametric-nam

Tools for capturing an amp as one NAM model that plays every knob setting.

- Helping someone capture an amp: use the skills in `.claude/skills`:
  `capture-session` (plan, levels, recording), `train-and-export` (train,
  judge, export the .nam), `add-a-knob` (extend a finished capture).
  `docs/GUIDE.md` is the same procedure in plain English; keep the two in step.
- The people using this are mostly guitarists. Explain in plain words, run the
  commands for them, and read results back plainly.
- Record and train with the PANAMA venv's python (`$PANAMA_DIR/.venv/bin/python`).
- The file format is `SPEC.md`. `parametric_nam/format.py` writes it;
  `cpp/ParametricWaveNet.*` reads it. A format change updates all three and
  `tests/`.
- Tests: `python -m pytest tests`.
