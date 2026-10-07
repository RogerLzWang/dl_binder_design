"""Post-processing scores for two-chain AlphaFold predictions.

The implementations here operate directly on AlphaFold's expected PAE matrix,
pLDDT values, and atom37 coordinates.  They intentionally have no JAX,
PyRosetta, or AlphaFold imports so they can also be used to rescore saved output.
"""

import math

import numpy as np


# Indices in AlphaFold's atom37 representation.
_CA_INDEX = 1
_CB_INDEX = 3


def _validate_common_inputs(pae, plddt, binder_length):
    pae = np.asarray(pae, dtype=np.float64)
    plddt = np.asarray(plddt, dtype=np.float64)

    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError("pae must be a square [num_res, num_res] matrix")
    if plddt.shape != (pae.shape[0],):
        raise ValueError("plddt must contain one value per residue")
    if binder_length <= 0 or binder_length >= pae.shape[0]:
        raise ValueError("binder_length must split the model into two non-empty chains")

    return pae, plddt


def _ipsae_direction(pae_block, pae_cutoff):
    """Calculate the asymmetric ipSAE score for one ordered chain pair."""
    valid = pae_block < pae_cutoff
    n0res = np.sum(valid, axis=1)

    # DunbrackLab/IPSAE uses 1.0 as the minimum d0 for proteins.  Clipping the
    # residue count to 26 is equivalent to that minimum in the vector formula.
    clipped_n0res = np.maximum(n0res, 26).astype(np.float64)
    d0res = np.maximum(
        1.0,
        1.24 * np.cbrt(clipped_n0res - 15.0) - 1.8,
    )

    transformed = 1.0 / (1.0 + np.square(pae_block / d0res[:, None]))
    row_sums = np.sum(np.where(valid, transformed, 0.0), axis=1)
    row_scores = np.divide(
        row_sums,
        n0res,
        out=np.zeros_like(row_sums),
        where=n0res > 0,
    )
    return float(np.max(row_scores)) if row_scores.size else 0.0


def calculate_ipsae(pae, binder_length, pae_cutoff=10.0):
    """Return the maximum of the two directional ipSAE scores."""
    pae = np.asarray(pae, dtype=np.float64)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError("pae must be a square [num_res, num_res] matrix")
    if binder_length <= 0 or binder_length >= pae.shape[0]:
        raise ValueError("binder_length must split the model into two non-empty chains")
    if pae_cutoff <= 0:
        raise ValueError("pae_cutoff must be positive")

    binder_to_target = _ipsae_direction(
        pae[:binder_length, binder_length:], pae_cutoff
    )
    target_to_binder = _ipsae_direction(
        pae[binder_length:, :binder_length], pae_cutoff
    )
    return max(binder_to_target, target_to_binder)


def _interface_contacts(atom_positions, atom_mask, binder_length, cutoff):
    """Return the inter-chain C-beta contact mask and interface residues."""
    atom_positions = np.asarray(atom_positions)
    atom_mask = np.asarray(atom_mask)
    num_res = atom_positions.shape[0]

    if atom_positions.ndim != 3 or atom_positions.shape[1:] != (37, 3):
        raise ValueError("atom_positions must have shape [num_res, 37, 3]")
    if atom_mask.shape != (num_res, 37):
        raise ValueError("atom_mask must have shape [num_res, 37]")
    if cutoff <= 0:
        raise ValueError("contact cutoff must be positive")

    # Glycine has no C-beta. Falling back to C-alpha also makes the scoring
    # robust to any other residue whose predicted C-beta is unexpectedly absent.
    has_cb = atom_mask[:, _CB_INDEX].astype(bool)
    representative_positions = np.where(
        has_cb[:, None],
        atom_positions[:, _CB_INDEX, :],
        atom_positions[:, _CA_INDEX, :],
    )

    binder_positions = representative_positions[:binder_length]
    target_positions = representative_positions[binder_length:]
    distances = np.linalg.norm(
        binder_positions[:, None, :] - target_positions[None, :, :], axis=-1
    )
    contacts = distances <= cutoff

    interface_residues = np.zeros(num_res, dtype=bool)
    interface_residues[:binder_length] = np.any(contacts, axis=1)
    interface_residues[binder_length:] = np.any(contacts, axis=0)
    return contacts, interface_residues


def calculate_pdockq(plddt, contacts, interface_residues):
    """Calculate pDockQ from interface pLDDT and the number of contacts."""
    plddt = np.asarray(plddt, dtype=np.float64)
    contacts = np.asarray(contacts, dtype=bool)
    interface_residues = np.asarray(interface_residues, dtype=bool)
    num_contacts = int(np.sum(contacts))

    if num_contacts == 0:
        return 0.0

    mean_plddt = float(np.mean(plddt[interface_residues]))
    x = mean_plddt * math.log10(num_contacts)
    return 0.724 / (1.0 + math.exp(-0.052 * (x - 152.611))) + 0.018


def _pdockq2_direction(pae_block, contacts, mean_interface_plddt):
    if not np.any(contacts):
        return 0.0

    pae_tm = 1.0 / (1.0 + np.square(pae_block[contacts] / 10.0))
    x = mean_interface_plddt * float(np.mean(pae_tm))
    return 1.31 / (1.0 + math.exp(-0.075 * (x - 84.733))) + 0.005


def calculate_pdockq2(pae, plddt, binder_length, contacts, interface_residues):
    """Calculate pDockQ2, reporting the better of the two PAE directions."""
    if not np.any(contacts):
        return 0.0

    mean_plddt = float(np.mean(np.asarray(plddt)[interface_residues]))
    binder_to_target = _pdockq2_direction(
        pae[:binder_length, binder_length:], contacts, mean_plddt
    )
    target_to_binder = _pdockq2_direction(
        pae[binder_length:, :binder_length], contacts.T, mean_plddt
    )
    return max(binder_to_target, target_to_binder)


def calculate_scores(
        pae,
        plddt,
        atom_positions,
        atom_mask,
        binder_length,
        ipsae_pae_cutoff=10.0,
        contact_cutoff=8.0):
    """Calculate ipSAE, pDockQ, and pDockQ2 for a two-chain prediction."""
    pae, plddt = _validate_common_inputs(pae, plddt, binder_length)
    if np.asarray(atom_positions).shape[0] != pae.shape[0]:
        raise ValueError("atom_positions and pae must contain the same residues")
    contacts, interface_residues = _interface_contacts(
        atom_positions, atom_mask, binder_length, contact_cutoff
    )

    return {
        "ipsae": calculate_ipsae(pae, binder_length, ipsae_pae_cutoff),
        "pdockq": calculate_pdockq(plddt, contacts, interface_residues),
        "pdockq2": calculate_pdockq2(
            pae, plddt, binder_length, contacts, interface_residues
        ),
    }
