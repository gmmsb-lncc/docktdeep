import random

import numpy as np
import torch

from src.docktdeep.transforms import Random90DegreesRotation


def test_all_24_90degree_rotations_are_distinct():
    voxel_grid = torch.load("tests/data/6rnt_grid.pt")
    transform = Random90DegreesRotation()
    grids = []
    for j in range(6):
        for i in range(4):
            params = transform.get_rotation_params(j)
            grid = transform.rotate90(voxel_grid, params["k"], params["adjusted_axes"])
            grid = transform.rotate90(grid, i, params["axes"])
            grids.append(grid)

    assert len(set(str(x) for x in grids)) == 24


def test_large_sample_includes_all_24_distinct_rotations():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    voxel_grid = torch.load("tests/data/6rnt_grid.pt")
    transform = Random90DegreesRotation()

    grids = []
    for _ in range(250):
        grid = transform(voxel_grid)
        grids.append(str(grid))

    assert len(set(grids)) == 24
