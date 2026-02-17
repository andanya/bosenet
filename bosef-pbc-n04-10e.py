from pyscf import gto

from ferminet import base_2DEG_config
from ferminet import train2DEG
from ferminet.utils import system

from ferminet.pbc import envelopes

import numpy as np
import os


# Torus dimension: L x L periodic box
TORUS_LENGTH = 5.0
N_PARTICLES = 10

mol = gto.Mole()
mol.build(
    atom = 'H  0 0 0',
    charge = 1 - N_PARTICLES,
    spin = N_PARTICLES,
    basis = 'sto-3g', unit='bohr')

cfg = base_2DEG_config.default()
cfg.system.pyscf_mol = mol


# cfg.system.electrons = (7, 7)
# A ghost atom at the origin defines one-electron coordinate system.
# Element 'X' is a dummy nucleus with zero charge
# cfg.system.molecule = [system.Atom("X", (0., 0., 0.))]
# Pretraining is not currently implemented for systems in PBC
cfg.pretrain.method = None

lattice = TORUS_LENGTH * np.eye(2)
# kpoints = envelopes.make_kpoints(lattice, cfg.system.electrons)

# cfg.system.make_local_energy_fn = "ferminet.pbc.hamiltonian.local_energy"
# cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True}
cfg.network.make_feature_layer_fn = (
    "ferminet.pbc.feature_layer.make_pbc_feature_layer")
cfg.network.make_feature_layer_kwargs = {
    "lattice": lattice,
    "include_r_ae": False
}
# cfg.network.make_envelope_fn = (
#     "ferminet.pbc.envelopes.make_multiwave_envelope")
# cfg.network.make_envelope_kwargs = {"kpoints": kpoints}

cfg.network.make_envelope_fn = (
    "ferminet.envelopes.make_null_envelope")
cfg.network.make_envelope_kwargs = {}
cfg.network.full_det = True
# return cfg



# Set training parameters
# cfg.batch_size = 256
# cfg.pretrain.iterations = 100
cfg.optim.iterations = 10000
cfg.network.network_type = 'psiformer'
# cfg.network.complex = True
cfg.log.save_path = './' + os.path.basename(__file__).removesuffix('.py')

# cfg.update(system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))
# for atom in cfg.system.molecule:
#     atom.coords = atom.coords[:2]

cfg.interaction_strength = 0.1

train2DEG.train(cfg)
