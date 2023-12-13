import numpy as np
import torch
from docktgrid.molecule import MolecularComplex
from docktgrid.view import View


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
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.view = view
        self.p = p
        self.molecular_unit = molecular_unit
        self.bp = beta_probability
        self.rng = rng

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
        alpha, beta = self.rng.uniform(size=2)
        self.set_random_nums(alpha, beta)

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
