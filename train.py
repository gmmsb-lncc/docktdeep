import argparse
import glob
import os
import subprocess
import sys

import aim
import docktgrid
import dotenv
import lightning.pytorch as pl
import torch
from aim.pytorch_lightning import AimLogger
from docktgrid.view import BasicView, VolumeView
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dataset import PDBbind
from models import *
from transforms import MolecularDropout, Random90DegreesRotation


def run(args):
    pl.seed_everything(args.seed)

    callbacks = configure_callbacks()
    logger = configure_logger(args)
    track_files(logger)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        detect_anomaly=args.detect_anomaly,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        callbacks=callbacks,
        logger=logger,
    )

    transforms = []
    if args.random_rotation:
        transforms.append(docktgrid.transforms.RandomRotation())
    if args.random_90degree_rotation:
        transforms.append(Random90DegreesRotation())

    voxel_grid = configure_voxel_grid(args)
    model = eval(args.model)(input_size=voxel_grid.shape, **vars(args))
    data_module = PDBbind(
        voxel_grid=voxel_grid,
        dataframe_path="data/index.csv",
        root_dir="data/processed",
        transforms=transforms,
        **vars(args)
    )

    trainer.fit(model, datamodule=data_module)

    return trainer


def configure_voxel_grid(args):
    views = [eval(v)() for v in args.view]

    return docktgrid.VoxelGrid(
        vox_size=args.vox_size,
        box_dims=args.box_dims,
        views=views,
    )


def configure_logger(args):
    logger = AimLogger(
        repo=os.environ.get("AIM_REPO") if args.remote else None,
        experiment=args.experiment,
        log_system_params=False,
    )
    return logger


def configure_callbacks():
    monitor, mode = "val_pearsonr", "max"
    patience = 1000
    callbacks = [
        EarlyStopping(monitor=monitor, mode=mode, patience=patience),
        ModelCheckpoint(monitor=monitor, mode=mode, save_top_k=1),
    ]
    return callbacks


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def track_files(logger) -> None:
    files = [os.path.abspath(__file__), os.path.abspath("dataset.py")]
    files.extend([os.path.abspath(f) for f in glob.glob("models/*.py")])
    files.extend([os.path.abspath(f) for f in glob.glob("transforms/*.py")])
    for idx, file in enumerate(files):
        with open(file, "r") as f:
            file = aim.Text(f.read())
        logger.experiment.track(file, name=os.path.basename(files[idx]))


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    dotenv.load_dotenv()

    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # script args
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--experiment", type=str, help="id of the experiment")
    parser.add_argument("--git-hash", type=str, default=get_git_revision_hash())
    parser.add_argument("--cmd", type=str, default=" ".join(sys.argv))

    # trainer args
    trainer_parser = parser.add_argument_group("Trainer args")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu")
    trainer_parser.add_argument("--devices", default=1)
    trainer_parser.add_argument("--max-epochs", type=int, default=1000)
    trainer_parser.add_argument("--detect-anomaly", action="store_true", default=False)
    trainer_parser.add_argument("--gradient-clip-val", type=float, default=5.0)
    trainer_parser.add_argument("--gradient-clip-algorithm", type=str, default="norm")

    # data args
    parser = PDBbind.add_specific_args(parser)

    # model args
    tmp_args, _ = parser.parse_known_args()
    eval(tmp_args.model).add_specific_args(parser)

    parser.add_argument("--help", "-h", action="help", default=argparse.SUPPRESS)
    args = parser.parse_args()

    run(args)
