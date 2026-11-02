# Credits

- **PANAMA** (ETH Zürich, [github.com/ETH-DISCO/PANAMA](https://github.com/ETH-DISCO/PANAMA),
  arXiv:2509.26564, MIT) is the training method these tools drive. No PANAMA
  code is included here; `convert_panama.py` reads its checkpoints.
- **Neural Amp Modeler** by Steven Atkinson
  ([NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore), MIT)
  defines the WaveNet and file layout this format extends. `fixtures/plain.nam`
  is NAM core's example WaveNet model, redistributed under its MIT licence.
- `fixtures/panama_demo.*` are converted from PANAMA's demo checkpoint and its
  PyTorch output, for hosts to null-test a loader against (MIT).
