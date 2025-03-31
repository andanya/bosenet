from pyscf import gto

from ferminet import base_2DEG_config
from ferminet import train2DEG
from ferminet.utils import system

mol = gto.Mole()
mol.build(
    atom = 'H  0 0 0',
    charge = -1,
    spin = 2,
    basis = 'sto-3g', unit='bohr')

cfg = base_2DEG_config.default()
cfg.system.pyscf_mol = mol

# Set training parameters
# cfg.batch_size = 256
# cfg.pretrain.iterations = 100
cfg.optim.iterations = 3000
cfg.network.network_type = 'psiformer'
# cfg.network.complex = True
cfg.log.save_path = './bosef-2e-nonint'

# cfg.update(system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))
# for atom in cfg.system.molecule:
#     atom.coords = atom.coords[:2]

train2DEG.train(cfg)
