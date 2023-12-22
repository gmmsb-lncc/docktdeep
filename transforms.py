import random

import torch
from docktgrid.molecule import MolecularComplex
from docktgrid.view import View

__all__ = ["MolecularDropout", "Random90DegreesRotation"]


class Random90DegreesRotation:
    def __init__(self, **kwargs) -> None:
        pass

    def rotate90(self, cube: torch.Tensor, k: int or None, axes: tuple[int]):
        """Perform rotation by 90 degrees in the given axis."""
        if not k or not axes:
            return cube

        return torch.stack(
            [torch.rot90(cube[ch], k, axes) for ch in range(cube.shape[0])]
        )

    def get_rotation_params(self, param_idx: int = random.randint(0, 5)):
        # suppose shape is pointing in axis 0 (up)
        roulette = {
            # no pre-rotation, still points up
            0: {"k": None, "adjusted_axes": None, "axes": (1, 2)},
            # pre-rotate 180 about axis 1, now shape is pointing down in axis 0
            1: {"k": 2, "adjusted_axes": (0, 2), "axes": (1, 2)},
            # pre-rotate 90 about axis 1, now shape is pointing in axis 2
            2: {"k": 1, "adjusted_axes": (0, 2), "axes": (0, 1)},
            # pre-rotate 270 about axis 1, now shape is pointing in axis 2
            3: {"k": -1, "adjusted_axes": (0, 2), "axes": (0, 1)},
            # pre-rotate 90 about axis 2, now shape is pointing in axis 1
            4: {"k": 1, "adjusted_axes": (0, 1), "axes": (0, 2)},
            # pre-rotate 270 about axis 2, now shape is pointing in axis 1
            5: {"k": -1, "adjusted_axes": (0, 1), "axes": (0, 2)},
        }

        # adjust orientation and perform random rotation
        return roulette[param_idx]

    def rotate(self, cube: torch.Tensor) -> torch.Tensor:
        """Random grid rotations in all different orientations.

        Adapted from: https://stackoverflow.com/a/33190472
        """

        # adjust orientation and perform random rotation
        params = self.get_rotation_params(random.randint(0, 5))

        return self.rotate90(
            self.rotate90(cube, params["k"], params["adjusted_axes"]),
            random.randint(0, 3),
            params["axes"],
        )

    def __call__(self, grid: torch.Tensor) -> torch.Tensor:
        return self.rotate(grid)


class MolecularDropout(View):
    """A wrapper to apply a `molecular dropout` transformation to any view.

    Drop protein/ligand atoms with a probability `p` using samples from a uniform
    distribution. This does not change the number of channels for the view.

    It should be used as a wrapper for any view, e.g.:

    ```
    basic_view = BasicView()
    transformed_view = ViewDropout(basic_view, p=0.1, molecular_unit="protein")
    ```

    In case `molecular_unit="complex"`, the probability of dropping the protein or the
    ligand is 0.5 each.

    Args:
        view (View): The view to be transformed.
        p (float): The probability of dropping an molecule (remove it from the view).
        molecular_unit (str): The molecular unit to drop atoms from. It can be either
        "protein", "ligand", or "complex".
    """

    def __init__(
        self,
        view: View,
        p: float,
        molecular_unit: str,
        beta_probability: float = 0.5,
    ):
        self.view = view
        self.p = p
        self.molecular_unit = molecular_unit
        self.bp = beta_probability

        if molecular_unit not in ["protein", "ligand", "complex"]:
            raise ValueError(
                f"'molecular_unit' must be either 'protein', 'ligand', or 'complex', but got {molecular_unit}"
            )
        if p < 0 or p > 1:
            raise ValueError(f"'p' must be between 0 and 1, but got {p}")

    def set_random_nums(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def get_random_nums(self):
        return self.alpha, self.beta

    def get_num_channels(self):
        return self.view.get_num_channels()

    def get_channels_names(self):
        return self.view.get_channels_names()

    def get_molecular_complex_channels(self, pl_complex: MolecularComplex):
        chs = self.view.get_molecular_complex_channels(pl_complex)
        if chs is None:
            return None

        alpha, beta = self.get_random_nums()
        unit = self.molecular_unit
        if unit == "complex":
            unit = "protein" if beta <= self.bp else "ligand"

        if alpha <= self.p:
            if unit == "protein":
                chs[:, : pl_complex.n_atoms_protein] = False
            else:
                chs[:, -pl_complex.n_atoms_ligand :] = False

        return chs

    def get_protein_channels(self, pl_complex: MolecularComplex):
        chs = self.view.get_protein_channels(pl_complex)
        alpha, beta = self.get_random_nums()

        unit = self.molecular_unit
        if unit == "complex":
            unit = "protein" if beta < self.bp else "ligand"

        if alpha <= self.p and unit == "protein":
            chs[:, : pl_complex.n_atoms_protein] = False

        # there's no need to do anything else if entity is ligand,
        # since channels are already False for the protein-only channels
        return chs

    def get_ligand_channels(self, pl_complex: MolecularComplex):
        chs = self.view.get_ligand_channels(pl_complex)
        alpha, beta = self.get_random_nums()

        unit = self.molecular_unit
        if unit == "complex":
            unit = "protein" if beta < self.bp else "ligand"

        if alpha <= self.p and unit == "ligand":
            chs[:, -pl_complex.n_atoms_ligand :] = False

        # there's no need to do anything else if entity is protein,
        # since channels are already False for the ligand-only channels
        return chs

    def __call__(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Concatenate all channels in a single tensor.

        Args:
            molecular_complex: MolecularComplex object.

        Returns:
            A boolean torch.Tensor array with shape
            (num_of_channels_defined_for_this_view, n_atoms_complex)

        """
        complex = self.get_molecular_complex_channels(molecular_complex)
        protein = self.get_protein_channels(molecular_complex)
        ligand = self.get_ligand_channels(molecular_complex)
        return torch.cat(
            (
                complex if complex is not None else torch.tensor([], dtype=torch.bool),
                protein if protein is not None else torch.tensor([], dtype=torch.bool),
                ligand if ligand is not None else torch.tensor([], dtype=torch.bool),
            ),
        )
