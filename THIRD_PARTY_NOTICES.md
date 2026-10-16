# Third-party notices

This custom node includes the Python package `minimax_h3_mlx` and the text-encoder
quantization utility from:

- Project: `Argus-AiTeam/minimax-h3-mac`
- Source: <https://github.com/Argus-AiTeam/minimax-h3-mac>
- Vendored revision: `3fcc9f8c557cf353400b1acf737b985e524a509f`
- License: Apache License 2.0 (see `LICENSE`)

The vendored source is included so ZIP installations do not need to clone or install
code from GitHub. Model weights are not included and remain subject to their own
upstream licenses.

## MiniMax Music 3 nodes

`music3_nodes.py` depends on, and does not vendor, `mlx-audio`
(<https://github.com/Blaizzy/mlx-audio>, MIT License). Its generation loop mirrors
`mlx_audio.music.models.minimax_music3.Model.generate`, which mlx-audio adapts
from `mikolaj92/minimax-music3-mlx` (Apache License 2.0). MiniMax Music 3 weights
and tokenizer are not included and remain under the MiniMax-Music3 Community License.
