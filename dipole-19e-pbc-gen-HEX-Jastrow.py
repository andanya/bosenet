"""19-particle gen-HEX with bare 1/r^3 dipole-dipole and Astrakharchik Jastrow.

Differs from dipole-19e-pbc-gen-HEX.py in two ways:
  * Interaction is bare 1/r^3 (no short-range r0 regularization). Set via
    cfg.interaction_small_length_cutoff = 0.0.
  * Adds the two-piece 2D-dipolar bosonic Jastrow factor of Astrakharchik
    et al., Phys. Rev. Lett. 98, 060405 (2007):
        f_2(r) = K_0(2*sqrt(r_0/r))                          r <= R_match
        f_2(r) = C_2 * [exp(-C_3/r) + exp(-C_3/(L-r))]        r >  R_match
    with r_0 = interaction_strength, L = shortest lattice vector length,
    and C_3 a trainable scalar. R_match defaults to L/4.
"""
from pyscf import gto

from ferminet import base_2DEG_config
from ferminet import train2DEG
from ferminet.utils import system

from ferminet.pbc import envelopes

import numpy as np
import os


N_PARTICLES = 19
# Hex primitive cell area = L^2 * sqrt(3)/2; match density n = 1/pi:
#   L^2 * sqrt(3)/2 = N*pi  ->  L = sqrt(2*N*pi/sqrt(3))
TORUS_LENGTH = np.sqrt(2 * N_PARTICLES * np.pi / np.sqrt(3))

mol = gto.Mole()
mol.build(
    atom = 'H  0 0 0',
    charge = 1 - N_PARTICLES,
    spin = N_PARTICLES,
    basis = 'sto-3g', unit='bohr')

cfg = base_2DEG_config.default()
cfg.system.pyscf_mol = mol


# Pretraining is not currently implemented for systems in PBC
cfg.pretrain.method = None

cfg.lattice_type = 'hexagonal'
lattice = base_2DEG_config.make_lattice(
    cfg.lattice_type, TORUS_LENGTH, ndim=2)

cfg.network.make_feature_layer_fn = (
    "ferminet.pbc.feature_layer.make_pbc_feature_layer")
cfg.network.make_feature_layer_kwargs = {
    "lattice": lattice,
    "include_r_ae": False
}

cfg.network.make_envelope_fn = (
    "ferminet.envelopes.make_null_envelope")
cfg.network.make_envelope_kwargs = {}
cfg.network.full_det = True
cfg.network.predict_logits = True

# === Bosonic dipole Jastrow (Astrakharchik et al. PRL 98, 060405) ===
cfg.network.jastrow = 'dipole_2d'
cfg.network.jastrow_kwargs = {
    'lattice': lattice,
    'rmatch_frac': 0.25,
}


# Set training parameters
cfg.optim.iterations = 50000
cfg.network.network_type = 'psiformer'
cfg.log.save_path = './' + os.path.basename(__file__).removesuffix('.py')

# Multi-lambda generative training
cfg.interaction_strength_training_set = [0.1, 0.5, 5.0, 20.0, 50.0, 80.0]
cfg.interaction_strength = 1.0  # default for inference
# Bare 1/r^3 interaction (no soft regularizer).
cfg.interaction_small_length_cutoff = 0.0

# === Conservative optimizer (matches WORKING) ===
cfg.optim.lr.rate = 0.02
cfg.optim.lr.delay = 10000.0
cfg.optim.lr.decay = 1.5
cfg.optim.clip_local_energy = 3.0
cfg.optim.kfac.damping = 0.01
cfg.optim.kfac.norm_constraint = 0.003

train2DEG.train(cfg)
