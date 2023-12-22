import random

from docktgrid.molecule import MolecularComplex
from docktgrid.view import VolumeView

from transforms import MolecularDropout


def test_drops_protein_from_when_dropping_protein():
    mol = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        path="tests/data/",
    )

    view = MolecularDropout(view=VolumeView(), p=1.0, molecular_unit="protein")
    alpha, beta = random.random(), random.random()
    view.set_random_nums(alpha, beta)

    channels = view(mol)

    assert channels.shape == (3, mol.n_atoms)
    assert not channels[:, : mol.n_atoms_protein].any()


def test_does_not_drop_ligand_when_dropping_protein():
    mol = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        path="tests/data/",
    )

    view = MolecularDropout(view=VolumeView(), p=1.0, molecular_unit="protein")
    alpha, beta = random.random(), random.random()
    view.set_random_nums(alpha, beta)

    channels = view(mol)

    assert channels.shape == (3, mol.n_atoms)
    assert channels[:, -mol.n_atoms_ligand :].any()


def test_drops_ligand_when_dropping_ligand():
    mol = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        path="tests/data/",
    )

    view = MolecularDropout(view=VolumeView(), p=1.0, molecular_unit="ligand")
    alpha, beta = random.random(), random.random()
    view.set_random_nums(alpha, beta)

    channels = view(mol)

    assert channels.shape == (3, mol.n_atoms)
    assert not channels[:, -mol.n_atoms_ligand :].any()


def test_does_not_drop_protein_when_dropping_ligand():
    mol = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        path="tests/data/",
    )

    view = MolecularDropout(view=VolumeView(), p=1.0, molecular_unit="ligand")
    alpha, beta = random.random(), random.random()
    view.set_random_nums(alpha, beta)

    channels = view(mol)

    assert channels.shape == (3, mol.n_atoms)
    assert channels[:, : mol.n_atoms_protein].any()


def test_drops_protein_when_dropping_complex():
    mol = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        path="tests/data/",
    )

    view = MolecularDropout(
        view=VolumeView(),
        p=1.0,
        molecular_unit="complex",
        beta_probability=1.0,  # ensures that beta <= bp, so protein is dropped
    )

    alpha, beta = random.random(), random.random()
    view.set_random_nums(alpha, beta)

    channels = view(mol)

    assert channels.shape == (3, mol.n_atoms)
    assert not channels[:, : mol.n_atoms_protein].any()


def test_drops_ligand_when_dropping_complex():
    mol = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        path="tests/data/",
    )

    view = MolecularDropout(
        view=VolumeView(),
        p=1.0,
        molecular_unit="complex",
        beta_probability=0.0,  # ensures that beta > bp, so ligand is dropped
    )

    alpha, beta = random.random(), random.random()
    view.set_random_nums(alpha, beta)

    channels = view(mol)

    assert channels.shape == (3, mol.n_atoms)
    assert not channels[:, -mol.n_atoms_ligand :].any()


def test_does_not_drop_protein_when_dropping_complex():
    mol = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        path="tests/data/",
    )

    view = MolecularDropout(
        view=VolumeView(),
        p=1.0,
        molecular_unit="complex",
        beta_probability=0.0,  # ensures that beta > bp, so ligand is dropped
    )

    alpha, beta = random.random(), random.random()
    view.set_random_nums(alpha, beta)

    channels = view(mol)

    assert channels.shape == (3, mol.n_atoms)
    assert channels[:, : mol.n_atoms_protein].any()


def test_does_not_drop_ligand_when_dropping_complex():
    mol = MolecularComplex(
        "6rnt_protein.pdb",
        "6rnt_ligand.pdb",
        path="tests/data/",
    )

    view = MolecularDropout(
        view=VolumeView(),
        p=1.0,
        molecular_unit="complex",
        beta_probability=1.0,  # ensures that beta <= bp, so protein is dropped
    )

    alpha, beta = random.random(), random.random()
    view.set_random_nums(alpha, beta)

    channels = view(mol)

    assert channels.shape == (3, mol.n_atoms)
    assert channels[:, -mol.n_atoms_ligand :].any()
