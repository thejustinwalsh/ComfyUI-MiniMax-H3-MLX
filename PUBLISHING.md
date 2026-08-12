# Publishing to Comfy Registry / ComfyUI Manager

The Registry is the catalog and versioned package store used by ComfyUI Manager.
Publishing here lets end users search for **MiniMax H3 MLX for Mac** in Manager and
install a Registry-hosted ZIP plus the dependencies declared in `pyproject.toml`.
End users do not need to clone this repository.

## Reserved project choices

- Registry node ID: `minimax-h3-mlx`
- Display name: `MiniMax H3 MLX for Mac`
- Intended Publisher ID: `argus-ai-team`
- Platform: macOS / Apple Metal only
- Initial version: `0.1.0`

`minimax-h3` is already owned by another publisher and describes a CUDA/PyTorch
implementation. Do not rename this project to that ID.

## One-time account setup

1. Sign in at <https://registry.comfy.org>.
2. Create the globally unique Publisher ID `argus-ai-team`. Publisher IDs cannot
   be changed later. If the team chooses another ID, update `[tool.comfy].PublisherId`
   in `pyproject.toml` before the first publication.
3. At <https://registry.comfy.org/nodes>, open that publisher and create a publishing
   API key. Save it securely; never commit it.
4. Make the source repository in `[project.urls].Repository` publicly accessible.
   The prepared URL is:

   `https://github.com/Argus-AiTeam/ComfyUI-MiniMax-H3-MLX`

## Pre-publication gates

Before publishing an immutable Registry version:

1. Run an end-to-end 64x64, one-second generation on an Apple Silicon Mac.
2. Run one INT8 + native MLX Turbo generation and verify the MP4 contains both
   H.264 video and AAC stereo audio.
3. Confirm a clean ComfyUI environment can install `requirements.txt`.
4. Confirm `pyproject.toml` and `requirements.txt` contain identical runtime
   dependencies.
5. Confirm no model weights, tokens, `.env` files, generated media, or virtual
   environments are tracked by git.
6. Run:

   ```bash
   python -m pip install -U comfy-cli ruff
   comfy node validate
   comfy node pack
   unzip -t node.zip
   ```

## Manual publication without GitHub Actions

From the repository root:

```bash
python -m pip install -U comfy-cli ruff
comfy node publish --changelog "Initial Apple Silicon MLX preview with INT8, streamed low-memory generation, stereo audio, and MP4 preview."
```

The CLI prompts for the Registry API key, packages git-tracked files while honoring
`.comfyignore`, and uploads the archive to the Comfy Registry CDN.

## Publication through GitHub Actions

1. In the source repository, create an Actions secret named
   `REGISTRY_ACCESS_TOKEN` containing the Registry publishing API key.
2. Open **Actions -> Publish to Comfy Registry -> Run workflow**.
3. The workflow in `.github/workflows/publish.yml` publishes the version currently
   declared in `pyproject.toml`.

## Releasing updates

Registry versions are immutable. Never try to overwrite a published version. Bump
`[project].version` using semantic versioning before every publication:

- patch (`0.1.1`) for compatible fixes;
- minor (`0.2.0`) for compatible features;
- major (`1.0.0`) for a stable contract or later breaking releases.

After publication, verify:

```text
https://registry.comfy.org/nodes/minimax-h3-mlx
```

Then open ComfyUI Manager, search for `MiniMax H3 MLX for Mac`, install it, restart
ComfyUI, and verify all four node classes appear.
