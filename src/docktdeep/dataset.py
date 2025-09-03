import copy
import os
import pickle
from typing import Optional

import docktgrid
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from docktgrid.transforms import RandomRotation
from torch.utils.data import Dataset

from .transforms import MolecularDropout, Random90DegreesRotation


class PDBbind(pl.LightningDataModule):
    def __init__(
        self,
        voxel_grid: docktgrid.VoxelGrid,
        batch_size: int,
        dataframe_path: str = "/home/mpds/data/pdbbind2020-refined-prepared/index.csv",
        transforms=None,
        molecular_dropout: float = 0.0,
        molecular_dropout_unit: str = "",
        root_dir: str = "",
        experiment: str = "",
        protein_path_pattern: str = "{c}_protein_prep.pdb.pkl",
        ligand_path_pattern: str = "{c}_ligand_rnum.pdb.pkl",
        split_column: str = "random_split",  # Column name in the dataframe used to select train/validation/test splits
        **kwargs,
    ):
        super().__init__()
        self.voxel_grid = voxel_grid
        self.batch_size = batch_size
        self.df_path = dataframe_path
        self.transforms = transforms
        self.molecular_dropout = molecular_dropout
        self.molecular_dropout_unit = molecular_dropout_unit
        self.root_dir = root_dir
        self.experiment = experiment
        self.protein_path_pattern = protein_path_pattern
        self.ligand_path_pattern = ligand_path_pattern
        self.split_column = split_column

    @staticmethod
    def add_specific_args(parent_parser):
        # fmt: off
        parser = parent_parser.add_argument_group("Data args")
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--vox-size", type=float, default=1.0)
        parser.add_argument("--box-dims", type=list, default=[24.0, 24.0, 24.0])
        parser.add_argument("--view", nargs="+", type=str, default=["VolumeView", "BasicView"])
        parser.add_argument("--random-rotation", action="store_true", default=False)
        parser.add_argument("--rotation-90-degrees", action="store_true", default=False)
        parser.add_argument("--molecular-dropout", type=float, default=0.0)
        parser.add_argument("--molecular-dropout-unit", type=str, default="protein", help="protein, ligand, or complex")
        parser.add_argument("--protein-path-pattern", type=str, default="{c}_protein_prep.pdb.pkl", help="Path pattern for protein files, use {c} as placeholder for PDB ID")
        parser.add_argument("--ligand-path-pattern", type=str, default="{c}_ligand_rnum.pdb.pkl", help="Path pattern for ligand files, use {c} as placeholder for PDB ID")
        parser.add_argument("--split-column", type=str, default="random_split", help="Column name in the dataframe used to select train/validation/test splits")
        # fmt: on

        return parent_parser

    def setup(self, stage: str = None) -> None:
        self.df = pd.read_csv(self.df_path)  # [:200]

        self.train_dataset = self.get_dataset("train")
        self.val_dataset = self.get_dataset("validation")
        self.test_dataset = self.get_dataset("test")

        print(f"Train dataset: {len(self.train_dataset)}")
        print(f"Validation dataset: {len(self.val_dataset)}")
        print(f"Test dataset: {len(self.test_dataset)}")

    def get_dataset(self, split: str):
        dataset = self.df[self.df[self.split_column] == split]

        protein_files = [self.protein_path_pattern.format(c=c) for c in dataset.id]
        ligand_files = [self.ligand_path_pattern.format(c=c) for c in dataset.id]

        protein_mols = [
            # pickle.load(open(os.path.join(self.root_dir, f"{f}"), "rb"))
            os.path.join(self.root_dir, f)
            for f in protein_files
        ]
        ligand_mols = [
            # pickle.load(open(os.path.join(self.root_dir, f"{f}"), "rb"))
            os.path.join(self.root_dir, f)
            for f in ligand_files
        ]

        # # exclude atoms outside the box
        # for i, ptn in enumerate(protein_mols):
        #     radius = np.ceil(np.sqrt(3) * max(self.voxel_grid.shape[1:]) / 2)
        #     inside_atoms_idx = docktgrid.molparser.extract_binding_pocket(
        #         ptn.coords, ligand_mols[i].coords.mean(dim=1), radius
        #     )

        #     # keep only the atoms inside the binding pocket, rewrite the MolecularData attributes
        #     ptn.coords = ptn.coords[:, inside_atoms_idx]
        #     ptn.element_symbols = ptn.element_symbols[inside_atoms_idx]

        # apply molecular dropout view
        voxel_grid = self.voxel_grid
        if self.molecular_dropout > 0.0 and split == "train":
            voxel_grid = copy.deepcopy(self.voxel_grid)
            views = [
                MolecularDropout(v, self.molecular_dropout, self.molecular_dropout_unit)
                for v in voxel_grid.views
            ]
            voxel_grid.views = views

        data = VoxelDataset(
            protein_files=protein_mols,
            ligand_files=ligand_mols,
            labels=dataset.delta_g.values,
            voxel=voxel_grid,
            transform=self.transforms if split == "train" else None,
            molecular_dropout=self.molecular_dropout if split == "train" else 0.0,
        )

        return data

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )


class VoxelDataset(Dataset):
    """Dataset for protein-ligand voxel data (generates voxel grids on-the-fly).

    Protein and ligand files must be in a list of strings or a list of MolecularData
    objects and must appear in the same order.
    """

    def __init__(
        self,
        protein_files: list[str] | list[docktgrid.molparser.MolecularData],
        ligand_files: list[str] | list[docktgrid.molparser.MolecularData],
        labels: list[float],
        voxel: docktgrid.VoxelGrid,
        molparser: docktgrid.molparser.MolecularParser = docktgrid.molparser.MolecularParser(),
        transform: Optional[list[docktgrid.transforms.Transform]] = None,
        molecular_dropout: float = 0.0,
        rng: np.random.Generator = np.random.default_rng(),
        root_dir: str = "",
    ):
        assert len(protein_files) == len(ligand_files), "must have the same length!"
        assert len(protein_files) == len(labels), "must have the same length!"

        self.ptn_files = protein_files
        self.lig_files = ligand_files
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.voxel = voxel
        self.molparser = molparser
        self.root_dir = root_dir
        self.transform = transform
        self.molecular_dropout = molecular_dropout
        self.rng = rng

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        molecule = docktgrid.molecule.MolecularComplex(
            self.ptn_files[idx], self.lig_files[idx], self.molparser, self.root_dir
        )
        label = self.labels[idx]

        # apply random rotation
        for transform in self.transform or []:
            if isinstance(transform, RandomRotation):
                transform(molecule.coords, molecule.ligand_center)

        # apply molecular dropout
        if self.molecular_dropout > 0.0:
            alpha, beta = self.rng.uniform(size=2)
            for v in self.voxel.views:
                v.set_random_nums(alpha, beta)

            if alpha <= self.molecular_dropout:
                label = torch.tensor(0.0, dtype=torch.float32)

        voxs = self.voxel.voxelize(molecule)  # <- voxelization happens here

        # apply random 90 degree rotation
        for transform in self.transform or []:
            if isinstance(transform, Random90DegreesRotation):
                voxs = transform(voxs)

        return voxs, label
