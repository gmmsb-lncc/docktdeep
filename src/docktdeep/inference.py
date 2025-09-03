"""
Inference module for predicting protein-ligand binding affinities with docktdeep.
"""

import logging
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

import docktgrid
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import VoxelDataset
from .models import Baseline

# model configuration
MODEL_URL = "https://github.com/gmmsb-lncc/docktdeep/releases/latest/download/docktdeep-model.ckpt"
MODEL_FILENAME = "docktdeep-model.ckpt"


def get_model_cache_dir() -> Path:
    """Get the directory where model files are cached."""
    cache_dir = Path.home() / ".cache" / "docktdeep"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_model_if_needed() -> str:
    """Download the model checkpoint if it doesn't exist locally."""
    cache_dir = get_model_cache_dir()
    model_path = cache_dir / MODEL_FILENAME

    if not model_path.exists():
        logging.info(f"Downloading model checkpoint to {model_path}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            logging.info("Model downloaded successfully!")
        except Exception as e:
            local_checkpoint = (
                Path(__file__).parent.parent.parent / "ckpts" / MODEL_FILENAME
            )
            if local_checkpoint.exists():
                logging.warning(f"Failed to download model: {e}")
                logging.info(f"Using local checkpoint: {local_checkpoint}")
                return str(local_checkpoint)
            else:
                raise RuntimeError(
                    f"Failed to download model and no local checkpoint found: {e}"
                )

    return str(model_path)


def get_dataset(proteins: List[str], ligands: List[str]) -> VoxelDataset:
    """Create a VoxelDataset from protein and ligand files."""
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
        molecular_dropout=0.0,
    )

    logging.info(f"Number of samples: {len(data)}")
    return data


def get_model(model_checkpoint: Optional[str] = None) -> Baseline:
    """Load the model from checkpoint."""
    if model_checkpoint is None:
        model_checkpoint = download_model_if_needed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = Baseline.load_from_checkpoint(model_checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_binding_affinity(
    proteins: List[str],
    ligands: List[str],
    model_checkpoint: Optional[str] = None,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Predict binding affinities for protein-ligand pairs.

    Args:
        proteins: List of paths to protein files (.pdb)
        ligands: List of paths to ligand files (.pdb, .mol2)
        model_checkpoint: Optional path to model checkpoint
        batch_size: Batch size for inference

    Returns:
        Array of predicted binding affinities in kcal/mol
    """
    assert len(proteins) == len(
        ligands
    ), "Number of proteins must match number of ligands."

    dataset = get_dataset(proteins, ligands)
    model = get_model(model_checkpoint)

    dataloader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=False
    )
    device = next(model.parameters()).device

    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs, _ = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_predictions.append(outputs.cpu().numpy())

    return np.concatenate(all_predictions)
