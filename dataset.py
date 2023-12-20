import os
import pickle
from typing import Optional

import docktgrid
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from docktgrid.transforms import RandomRotation


class PDBbind(pl.LightningDataModule):
    def __init__(
        self,
        voxel_grid: docktgrid.VoxelGrid,
        batch_size: int,
        dataframe_path: str = "/home/mpds/data/pdbbind2020-refined-prepared/index.csv",
        random_rotation: bool = False,
        root_dir: str = "",
        **kwargs,
    ):
        super().__init__()
        self.voxel_grid = voxel_grid
        self.batch_size = batch_size
        self.df_path = dataframe_path
        self.transform = random_rotation
        self.root_dir = root_dir

    @staticmethod
    def add_specific_args(parent_parser):
        # fmt: off
        parser = parent_parser.add_argument_group("Data args")
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--vox-size", type=int, default=1.0)
        parser.add_argument("--box-dims", type=list, default=[24.0, 24.0, 24.0])
        parser.add_argument("--view", nargs="+", type=str, default=["VolumeView", "BasicView"])
        parser.add_argument("--random-rotation", action="store_true", default=False)
        # fmt: on

        return parent_parser

    def setup(self, stage: str = None) -> None:
        self.df = pd.read_csv(self.df_path)  # [:200]

        self.train_dataset = self.get_dataset("train")
        self.val_dataset = self.get_dataset("validation")

    def get_dataset(self, split: str):
        dataset = self.df[self.df.split == split]

        protein_files = [f"{c}_protein.pdb.pkl" for c in dataset.id]
        ligand_files = [f"{c}_ligand_rnum.pdb.pkl" for c in dataset.id]

        protein_mols = [
            pickle.load(open(os.path.join(self.root_dir, f"{f}"), "rb"))
            for f in protein_files
        ]
        ligand_mols = [
            pickle.load(open(os.path.join(self.root_dir, f"{f}"), "rb"))
            for f in ligand_files
        ]

        # exclude atoms outside the box
        for i, ptn in enumerate(protein_mols):
            radius = np.ceil(np.sqrt(3) * max(self.voxel_grid.shape[1:]) / 2)
            inside_atoms_idx = docktgrid.molparser.extract_binding_pocket(
                ptn.coords, ligand_mols[i].coords.mean(dim=1), radius
            )

            # keep only the atoms inside the binding pocket, rewrite the MolecularData attributes
            ptn.coords = ptn.coords[:, inside_atoms_idx]
            ptn.element_symbols = ptn.element_symbols[inside_atoms_idx]

        data = docktgrid.VoxelDataset(
            protein_files=protein_mols,
            ligand_files=ligand_mols,
            labels=range(len(protein_files)),
            voxel=self.voxel_grid,
            transform=[RandomRotation] if self.transform else None,
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
