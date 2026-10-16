"""ComfyUI nodes for MiniMax-H3 and MiniMax Music 3 inference with MLX on Apple Silicon."""

WEB_DIRECTORY = "./web"

if __package__:
    from .music3_nodes import NODE_CLASS_MAPPINGS as _MUSIC3_NODES
    from .music3_nodes import NODE_DISPLAY_NAME_MAPPINGS as _MUSIC3_NAMES
    from .nodes import NODE_CLASS_MAPPINGS as _H3_NODES
    from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _H3_NAMES

    NODE_CLASS_MAPPINGS = {**_H3_NODES, **_MUSIC3_NODES}
    NODE_DISPLAY_NAME_MAPPINGS = {**_H3_NAMES, **_MUSIC3_NAMES}
else:
    # Imported outside ComfyUI as a plain module (for example by pytest while
    # collecting tests/); there is no package to resolve relative imports.
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]
