from pathlib import Path


def _get_project_root():
    p = Path.cwd()
    while p.name != "zoo_vision":
        if p == p.parent:
            raise RuntimeError(
                "Could not find a path named zoo_vision in the hierarchy. Cannot determine project root."
            )
        p = p.parent
    return p


PROJECT_ROOT = _get_project_root()
