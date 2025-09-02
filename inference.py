"""
Inference module for predicting protein-ligand binding affinities with docktdeep.

Example usage:
    # basic usage with single files:
    python inference.py --proteins protein.pdb --ligands ligand.mol2

    # multiple files (using shell globbing):
    python inference.py --proteins $(ls *.pdb) --ligands $(ls *.mol2)

Requirements:
    - The number of protein files must match the number of ligand files; predictions will be made for each pair.
    - Protein files should be in PDB format (.pdb)
    - Ligand files should be in PDB or MOL2 format (.pdb, .mol2)

Output:
    A CSV file with the results. Predictions are given in kcal/mol.

Use `--help` to see all options.
"""

import argparse
import csv
import logging

import docktgrid
import numpy as np
import torch

# from docktgrid.transforms import RandomRotation
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VoxelDataset
from models import Baseline


def get_dataset(proteins, ligands):
    voxel = docktgrid.VoxelGrid(
        views=[docktgrid.view.VolumeView(), docktgrid.view.BasicView()],
        vox_size=1.0,
        box_dims=[24.0, 24.0, 24.0],
    )

    data = VoxelDataset(
        protein_files=proteins,
        ligand_files=ligands,
        labels=[0] * len(ligands),
        voxel=voxel,
        # transform=[RandomRotation()] if num_rotations > 1 else None,
        molecular_dropout=0.0,
    )

    logging.info(f"Number of samples: {len(data)}")

    return data


def get_model(model_checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = Baseline.load_from_checkpoint(model_checkpoint)
    model.to(device)
    model.eval()
    return model


def predict(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device

    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs, _ = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())

    return np.concatenate(all_predictions)


def main(args):
    assert len(args.proteins) == len(
        args.ligands
    ), "Number of proteins must match number of ligands."
    dataset = get_dataset(args.proteins, args.ligands)
    model = get_model(args.model_checkpoint)

    batch_size = min(args.max_batch_size, len(dataset))
    predictions = predict(dataset, model, batch_size)

    with open(args.output_csv, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["protein", "ligand", "delta_g"])
        for i in range(len(predictions)):
            csvwriter.writerow(
                [args.proteins[i], args.ligands[i], predictions[i].squeeze()]
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # fmt: off
    parser.add_argument("--proteins", nargs='+', required=True, help="Path(s) to the protein file(s) (.pdb).")
    parser.add_argument("--ligands", nargs='+', required=True, help="Path(s) to the ligand file(s) (.pdb, .mol2).")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Max batch size for inference.")
    parser.add_argument("--model-checkpoint", type=str, default="ckpts/v788f06d9409e4e4e87377564.ckpt", help="Path to the model checkpoint (.ckpt).")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="Path to the output CSV file.")
    args = parser.parse_args()
    main(args)
