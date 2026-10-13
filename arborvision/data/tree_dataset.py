"""Loader for the initial tree-species benchmark dataset.

Expected directory layout (ImageFolder style):

    data/tree_species/
        oak/
            img_001.jpg
            img_002.jpg
        maple/
            ...
        pine/
            ...

Each subdirectory name is treated as a class label. The loader returns
`((image_path, label), ...)` tuples plus a sorted label list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .loader import iter_image_paths, is_image_path


@dataclass(frozen=True)
class TreeSample:
    """A single labeled sample from the tree-species benchmark."""

    path: Path
    label: str


@dataclass(frozen=True)
class TreeDataset:
    """An in-memory index of the tree-species benchmark."""

    root: Path
    samples: tuple[TreeSample, ...]
    labels: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.samples)


def load_tree_dataset(root: Path) -> TreeDataset:
    """Walk `root` and build a `TreeDataset`.

    Skips empty subdirectories and any non-image files. Returns a dataset
    with zero samples (and an empty labels tuple) if the directory does not
    exist or contains no class subdirectories.
    """
    root = Path(root)
    if not root.exists():
        return TreeDataset(root=root, samples=(), labels=())

    labels: list[str] = []
    samples: list[TreeSample] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        label = class_dir.name
        labels.append(label)
        for p in sorted(class_dir.iterdir()):
            if p.is_file() and is_image_path(p):
                samples.append(TreeSample(path=p, label=label))

    return TreeDataset(root=root, samples=tuple(samples), labels=tuple(labels))


def iter_image_paths_unlabeled(root: Path) -> list[Path]:
    """Walk `root` and return all image paths (no label requirement).

    Used when the user supplies an unstructured folder for inference-only runs.
    """
    return iter_image_paths(root)