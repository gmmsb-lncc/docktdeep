"""
Command-line interface for docktdeep.
"""

import argparse
import csv
import logging
import sys
from typing import List

from .inference import predict_binding_affinity


def predict_command(args):
    """Execute the predict command."""
    try:
        predictions = predict_binding_affinity(
            proteins=args.proteins,
            ligands=args.ligands,
            model_checkpoint=args.model_checkpoint,
            batch_size=args.max_batch_size,
        )

        # Write results to CSV
        with open(args.output_csv, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["protein", "ligand", "delta_g"])
            for i in range(len(predictions)):
                csvwriter.writerow(
                    [args.proteins[i], args.ligands[i], predictions[i].squeeze()]
                )

        print(f"Predictions saved to {args.output_csv}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="docktdeep",
        description="Deep learning model for protein-ligand binding affinity prediction",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict protein-ligand binding affinities",
        description="""
Predict protein-ligand binding affinities using docktdeep.

Example usage:
    # basic usage with single files:
    docktdeep predict --proteins protein.pdb --ligands ligand.mol2

    # multiple files:
    docktdeep predict --proteins protein1.pdb protein2.pdb --ligands ligand1.mol2 ligand2.mol2

Requirements:
    - The number of protein files must match the number of ligand files
    - Protein files should be in PDB format (.pdb)
    - Ligand files should be in PDB or MOL2 format (.pdb, .mol2)

Output:
    A CSV file with the results. Predictions are given in kcal/mol.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    predict_parser.add_argument(
        "--proteins",
        nargs="+",
        required=True,
        help="Path(s) to the protein file(s) (.pdb).",
    )
    predict_parser.add_argument(
        "--ligands",
        nargs="+",
        required=True,
        help="Path(s) to the ligand file(s) (.pdb, .mol2).",
    )
    predict_parser.add_argument(
        "--max-batch-size", type=int, default=32, help="Max batch size for inference."
    )
    predict_parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to a custom model checkpoint (.ckpt). If not provided, uses the default model.",
    )
    predict_parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="Path to the output CSV file.",
    )

    predict_parser.set_defaults(func=predict_command)

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
