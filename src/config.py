from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    outputs_dir: Path
    preprocessed_dir: Path
    models_dir: Path
    figures_dir: Path


def build_paths(project_root: str | Path = ".") -> ProjectPaths:
    """
    Build standardized project paths.
    - project_root: repository root (default ".")
    """
    root = Path(project_root).expanduser().resolve()
    data_dir = root / "data"
    outputs_dir = root / "outputs"

    return ProjectPaths(
        root=root,
        data_dir=data_dir,
        outputs_dir=outputs_dir,
        preprocessed_dir=outputs_dir / "preprocessed",
        models_dir=outputs_dir / "models",
        figures_dir=outputs_dir / "figures",
    )


def ensure_dirs(paths: ProjectPaths) -> None:
    """Create output directories if missing."""
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    paths.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
