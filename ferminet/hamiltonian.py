# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluating the Hamiltonian on a wavefunction."""

import itertools

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import chex
from ferminet import networks
from ferminet import pseudopotential as pp
from ferminet.pbc.feature_layer import periodic_norm
from ferminet.utils import utils
import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol


Array = Union[jnp.ndarray, np.ndarray]


@chex.dataclass
class LocalEnergyAux:
  """Auxiliary outputs from a local-energy evaluation.

  Attributes:
    kinetic: per-configuration local kinetic energy (scalar).
    potential: per-configuration local potential energy (scalar).
    energy_mat: for excited states, the local energy matrix; otherwise None.
  """
  kinetic: jax.Array
  potential: jax.Array
  energy_mat: Optional[jax.Array] = None


class LocalEnergy(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, LocalEnergyAux]:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

  def __call__(
      self,
      f: networks.FermiNetLike,
      charges: jnp.ndarray,
      nspins: Sequence[int],
      use_scan: bool = False,
      complex_output: bool = False,
      **kwargs: Any
  ) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      charges: nuclear charges.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      complex_output: If true, the output of f is complex-valued.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


KineticEnergy = Callable[
    [networks.ParamTree, networks.FermiNetData], jnp.ndarray
]


def local_kinetic_energy(
    f: networks.FermiNetLike,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
) -> KineticEnergy:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the wavefunction as a
      (sign or phase, log magnitude) tuple.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  phase_f = utils.select_output(f, 0)
  logabs_f = utils.select_output(f, 1)

  if laplacian_method == 'default':

    def _lapl_over_f(params, data):
      n = data.positions.shape[0]
      eye = jnp.eye(n)
      grad_f = jax.grad(logabs_f, argnums=1)
      def grad_f_closure(x):
        return grad_f(params, x, data.spins, data.atoms, data.charges,
                      data.interaction_strength)

      primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

      if complex_output:
        grad_phase = jax.grad(phase_f, argnums=1)
        def grad_phase_closure(x):
          return grad_phase(params, x, data.spins, data.atoms, data.charges,
                            data.interaction_strength)
        phase_primal, dgrad_phase = jax.linearize(
            grad_phase_closure, data.positions)
        hessian_diagonal = (
            lambda i: dgrad_f(eye[i])[i] + 1.j * dgrad_phase(eye[i])[i]
        )
      else:
        hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

      if use_scan:
        _, diagonal = lax.scan(
            lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
        result = -0.5 * jnp.sum(diagonal)
      else:
        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
      result -= 0.5 * jnp.sum(primal ** 2)
      if complex_output:
        result += 0.5 * jnp.sum(phase_primal ** 2)
        result -= 1.j * jnp.sum(primal * phase_primal)
      return result

  elif laplacian_method == 'folx':
    def _lapl_over_f(params, data):
      f_closure = lambda x: f(params, x, data.spins, data.atoms, data.charges,
                               data.interaction_strength)
      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
      output = f_wrapped(data.positions)
      result = - (output[1].laplacian +
                  jnp.sum(output[1].jacobian.dense_array ** 2)) / 2
      if complex_output:
        result -= 0.5j * output[0].laplacian
        result += 0.5 * jnp.sum(output[0].jacobian.dense_array ** 2)
        result -= 1.j * jnp.sum(output[0].jacobian.dense_array *
                                output[1].jacobian.dense_array)
      return result
  else:
    raise NotImplementedError(f'Laplacian method {laplacian_method} '
                              'not implemented.')

  return _lapl_over_f


def excited_kinetic_energy_matrix(
    f: networks.FermiNetLike,
    states: int,
    complex_output: bool = False,
    laplacian_method: str = 'default') -> KineticEnergy:
  """Creates a f'n which evaluates the matrix of local kinetic energies.

  Args:
    f: A network which returns a tuple of sign(psi) and log(|psi|) arrays, where
      each array contains one element per excited state.
    states: the number of excited states
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian

  Returns:
    A function which computes the matrices (psi) and (K psi), which are the
      value of the wavefunction and the kinetic energy applied to the
      wavefunction for all combinations of electron sets and excited states.
  """

  def _lapl_all_states(params, pos, spins, atoms, charges,
                       interaction_strength=0.0):
    """Return K psi/psi for each excited state."""
    n = pos.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.jacrev(utils.select_output(f, 1), argnums=1)
    grad_f_closure = lambda x: grad_f(params, x, spins, atoms, charges,
                                       interaction_strength)
    primal, dgrad_f = jax.linearize(grad_f_closure, pos)

    if complex_output:
      grad_phase = jax.jacrev(utils.select_output(f, 0), argnums=1)
      def grad_phase_closure(x):
        return grad_phase(params, x, spins, atoms, charges,
                          interaction_strength)
      phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, pos)
      hessian_diagonal = (
          lambda i: dgrad_f(eye[i])[:, i] + 1.j * dgrad_phase(eye[i])[:, i]
      )
    else:
      phase_primal = 1.0
      hessian_diagonal = lambda i: dgrad_f(eye[i])[:, i]

    if complex_output:
      if pos.dtype == jnp.float32:
        dtype = jnp.complex64
      elif pos.dtype == jnp.float64:
        dtype = jnp.complex128
      else:
        raise ValueError(f'Unsupported dtype for input: {pos.dtype}')
    else:
      dtype = pos.dtype

    result = -0.5 * lax.fori_loop(
        0, n, lambda i, val: val + hessian_diagonal(i),
        jnp.zeros(states, dtype=dtype))
    result -= 0.5 * jnp.sum(primal ** 2, axis=-1)
    if complex_output:
      result += 0.5 * jnp.sum(phase_primal ** 2, axis=-1)
      result -= 1.j * jnp.sum(primal * phase_primal, axis=-1)

    return result

  def _lapl_over_f(params, data):
    """Return the kinetic energy (divided by psi) summed over excited states."""
    pos_ = jnp.reshape(data.positions, [states, -1])
    spins_ = jnp.reshape(data.spins, [states, -1])

    if laplacian_method == 'default':
      vmap_f = jax.vmap(f, (None, 0, 0, None, None, None))
      sign_mat, log_mat = vmap_f(params, pos_, spins_, data.atoms, data.charges,
                                 data.interaction_strength)
      vmap_lapl = jax.vmap(_lapl_all_states, (None, 0, 0, None, None, None))
      lapl = vmap_lapl(params, pos_, spins_, data.atoms,
                       data.charges,
                       data.interaction_strength)  # K psi_i(r_j) / psi_i(r_j)
    elif laplacian_method == 'folx':
      # CAUTION!! Only the first array of spins is being passed!
      f_closure = lambda x: f(params, x, spins_[0], data.atoms, data.charges,
                               data.interaction_strength)
      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
      sign_out, log_out = folx.batched_vmap(f_wrapped, 1)(pos_)
      log_mat = log_out.x
      lapl = -(log_out.laplacian +
               jnp.sum(log_out.jacobian.dense_array ** 2, axis=-2)) / 2
      if complex_output:
        sign_mat = sign_out.x
        lapl -= 0.5j * sign_out.laplacian
        lapl += 0.5 * jnp.sum(sign_out.jacobian.dense_array ** 2, axis=-2)
        lapl -= 1.j * jnp.sum(sign_out.jacobian.dense_array *
                              log_out.jacobian.dense_array, axis=-2)
      else:
        sign_mat = sign_out
    else:
      raise NotImplementedError(f'Laplacian method {laplacian_method} '
                                'not implemented with excited states.')

    # psi_i(r_j)
    # subtract off largest value to avoid under/overflow
    if complex_output:
      psi_mat = jnp.exp(log_mat + 1.j * sign_mat - jnp.max(log_mat))
    else:
      psi_mat = sign_mat * jnp.exp(log_mat - jnp.max(log_mat))
    kpsi_mat = lapl * psi_mat  # K psi_i(r_j)
    return psi_mat, kpsi_mat

  return _lapl_over_f


def compute_periodic_r_ee(ee: Array, lattice: Array) -> jnp.ndarray:
  """Computes periodic electron-electron distances using minimum image convention.

  Args:
    ee: Shape (nelectrons, nelectrons, ndim). Electron-electron displacement vectors.
    lattice: Shape (ndim, ndim). Matrix of lattice vectors (columns are vectors).

  Returns:
    r_ee: Shape (nelectrons, nelectrons, 1). Periodic distances.
  """
  reciprocal_vecs = jnp.linalg.inv(lattice)
  lattice_metric = lattice.T @ lattice
  # Convert to fractional coordinates
  scaled_ee = jnp.einsum('il,jkl->jki', reciprocal_vecs, ee)
  # Compute periodic norm
  n = ee.shape[0]
  # Add identity to diagonal to avoid zero-norm issues, then mask out
  scaled_ee_safe = scaled_ee + jnp.eye(n)[..., None]
  r_ee = periodic_norm(lattice_metric, scaled_ee_safe) * (1.0 - jnp.eye(n))
  return r_ee[..., None]


def potential_electron_electron(
    ee: Array,
    r_ee: Array,
    interaction_strength: float,
    interaction_small_length_cutoff: float = 0.1,
    lattice: Optional[Array] = None,
    interaction_truncation_limit: int = 5,
    interaction_cutoff_radius: Optional[float] = None,
) -> jnp.ndarray:
  """Returns the electron-electron repulsion potential.

  Args:
    ee: Shape (nelectrons, nelectrons, ndim). Electron-electron displacements.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j under minimum-image convention.
    interaction_strength: Prefactor ``k`` in the k / (r + r0)^3 pair potential.
    interaction_small_length_cutoff: Softening length r0 that regularises the
      1/r^3 singularity at short range.
    lattice: Shape (ndim, ndim). Lattice vectors for periodic cell.
    interaction_truncation_limit: Number of unit-cell images included in each
      positive/negative lattice-vector direction for the real-space summation.
    interaction_cutoff_radius: Optional spherical cutoff R_c. When set, pair
      (and image) contributions at distance > R_c are dropped from the explicit
      sum; the neglected long-range part is meant to be supplied separately as
      an isotropic tail correction (see ``electron_electron_tail_coefficient``).
      When None (default) the explicit sum is unchanged.
  """
  r0 = interaction_small_length_cutoff
  n = ee.shape[0]
  if lattice is None:
    r_pairs = r_ee[jnp.triu_indices_from(r_ee[..., 0], 1)]
    pair_energy = 1.0 / (r_pairs + r0) ** 3
    if interaction_cutoff_radius is not None:
      pair_energy = jnp.where(r_pairs <= interaction_cutoff_radius,
                              pair_energy, 0.0)
    return interaction_strength * jnp.sum(pair_energy)

  dim = ee.shape[-1]
  ordinals = jnp.arange(-interaction_truncation_limit,
                        interaction_truncation_limit + 1)
  image_indices = jnp.array(list(itertools.product(ordinals, repeat=dim)))
  # R = A n with lattice vectors as columns in A.
  image_shifts = jnp.einsum('ij,nj->ni', lattice, image_indices)

  idx_i, idx_j = jnp.triu_indices(n, k=1)
  pair_displacements = ee[idx_i, idx_j]
  # Wrap pair displacements to the minimum-image cell so the lattice sum is
  # correct even when MCMC walkers have drifted many unit cells away from [0,L).
  reciprocal = jnp.linalg.inv(lattice)
  frac = jnp.einsum('ij,pj->pi', reciprocal, pair_displacements)
  pair_displacements = jnp.einsum('ij,pj->pi', lattice, frac - jnp.round(frac))
  all_displacements = pair_displacements[:, None, :] + image_shifts[None, :, :]
  distances = jnp.linalg.norm(all_displacements, axis=-1)
  pair_energy = 1.0 / (distances + r0) ** 3
  if interaction_cutoff_radius is not None:
    pair_energy = jnp.where(distances <= interaction_cutoff_radius,
                            pair_energy, 0.0)
  return interaction_strength * jnp.sum(pair_energy)


def electron_electron_tail_coefficient(
    lattice: Array,
    n_particles: int,
    interaction_small_length_cutoff: float,
    interaction_cutoff_radius: float,
    g2_r: Optional[Array] = None,
    g2_values: Optional[Array] = None,
) -> float:
  """Isotropic large-r tail of the 1/(r+r0)^3 pair energy, per unit strength.

  Returns a scalar ``tail_coeff`` such that the long-range correction to the
  electron-electron potential energy of one configuration is

      E_tail = interaction_strength * tail_coeff,

  with

      E_tail / k = 0.5 * N * n * \\int_{R_c}^{\\infty}
                        [1 / (r + r0)^3] g2(r) (2 pi r) dr ,

  where ``N`` is the particle number, ``n = N / A`` the 2D number density,
  ``R_c`` the spherical cutoff applied to the explicit pair sum, and ``g2(r)``
  the radial pair-correlation function. ``g2(r)`` defaults to 1 (uniform fluid,
  the standard mean-field tail). A tabulated ``g2`` may be supplied via
  ``(g2_r, g2_values)``; it is integrated numerically over its range (which
  should start at ``R_c``) with ``g2 = 1`` assumed beyond the last tabulated
  point.

  This is computed once at construction time in NumPy (it does not depend on
  the instantaneous walker positions) and added as a constant inside the jitted
  local-energy evaluation.

  Args:
    lattice: Shape (ndim, ndim). Real-space lattice vectors (as columns).
    n_particles: Number of particles N.
    interaction_small_length_cutoff: Softening length r0.
    interaction_cutoff_radius: Spherical cutoff R_c (lower limit of the tail).
    g2_r: Optional radii at which g2 is tabulated.
    g2_values: Optional g2 values matching ``g2_r``.

  Returns:
    Python float ``tail_coeff``.
  """
  import numpy as np  # local import; runs at construction, not under jit.
  lattice = np.asarray(lattice, dtype=np.float64)
  area = np.abs(np.linalg.det(lattice))
  n_density = n_particles / area
  r0 = float(interaction_small_length_cutoff)
  r_cut = float(interaction_cutoff_radius)

  def analytic_uniform(a):
    # \int_a^\infty (2 pi r) / (r + r0)^3 dr
    #   = 2 pi [ 1 / (a + r0) - r0 / (2 (a + r0)^2) ]
    return 2.0 * np.pi * (1.0 / (a + r0) - r0 / (2.0 * (a + r0) ** 2))

  if g2_r is None or g2_values is None:
    integral = analytic_uniform(r_cut)
  else:
    g2_r = np.asarray(g2_r, dtype=np.float64)
    g2_values = np.asarray(g2_values, dtype=np.float64)
    mask = g2_r >= r_cut
    rr = g2_r[mask]
    gg = g2_values[mask]
    integrand = (1.0 / (rr + r0) ** 3) * gg * (2.0 * np.pi * rr)
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz  # NumPy 2.x rename.
    integral = float(trapezoid(integrand, rr))
    # Add the analytic remainder beyond the tabulated range assuming g2 -> 1.
    integral += analytic_uniform(rr[-1])

  return float(0.5 * n_particles * n_density * integral)


def make_ewald_1r3_potential(
    lattice: Array,
    n_particles: int,
    interaction_small_length_cutoff: float = 0.1,
    eta: Optional[float] = None,
    eta_scale: float = 1.0,
    real_shells: int = 3,
    recip_shells: int = 8,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Builds a jittable 2D Ewald evaluator for the 1/(r + a)^3 pair potential.

  Returns the *exact* periodic potential energy (per unit interaction strength)
  of one configuration under an arbitrary 2D Bravais lattice, with no cutoff or
  tail-correction error. This is the Ewald alternative to the minimum-image sum
  (`potential_electron_electron`) and the analytic tail correction; unlike those
  it also includes the particle self-image (Madelung) interaction, which is the
  physically correct energy of the infinite periodic crystal.

  The softened dipole potential is split as

      1/(r + a)^3 = [1/(r + a)^3 - 1/r^3] + 1/r^3 ,

  where the bracket is short-ranged (~ -3a/r^4) and summed directly in real
  space over lattice images, and the bare 1/r^3 part is evaluated by the
  generalized 2D dipolar Ewald sum (cf. Astrakharchik-reproducing/ewald.py,
  extended here to a non-orthogonal lattice and to a > 0):

      phi_short(r) = [erfc(eta r) + (2/sqrt(pi)) eta r exp(-(eta r)^2)] / r^3
      I(G)         = 2 eta exp(-G^2/4eta^2) - sqrt(pi) G erfc(G/(2 eta))
      U = 0.5 sum_{i,j,R != self} [phi_short + (1/(r+a)^3 - 1/r^3)]
          + (sqrt(pi)/A) sum_{G != 0} I(G) |rho(G)|^2
          + 2 sqrt(pi) eta N^2 / A - 2 N eta^3 / (3 sqrt(pi)) .

  The result is independent of the Ewald parameter eta (verified numerically);
  a = 0 recovers the triangular-lattice Madelung constant 4.446 n^{3/2}.

  Args:
    lattice: Shape (2, 2). Real-space primitive vectors as *columns*.
    n_particles: Number of particles N.
    interaction_small_length_cutoff: Softening length a in 1/(r + a)^3.
    eta: Ewald splitting parameter. If None, uses eta_scale * sqrt(pi)/sqrt(A).
    eta_scale: Multiplier on the default eta (energy is eta-independent; this
      only trades real- vs reciprocal-space convergence).
    real_shells: Half-width (in lattice vectors) of the real-space image block.
    recip_shells: Half-width of the reciprocal-space G block.

  Returns:
    Callable f(positions) -> scalar, where positions has shape (N, 2), giving
    the periodic 1/(r + a)^3 energy per unit interaction strength.
  """
  A = np.asarray(lattice, dtype=np.float64)
  if A.shape != (2, 2):
    raise ValueError('Ewald summation is implemented for 2D lattices only '
                     f'(got lattice shape {A.shape}).')
  a = float(interaction_small_length_cutoff)
  area = float(np.abs(np.linalg.det(A)))
  if eta is None:
    eta = eta_scale * np.sqrt(np.pi) / np.sqrt(area)
  eta = float(eta)
  A_inv = np.linalg.inv(A)
  B = 2.0 * np.pi * A_inv.T  # reciprocal primitive vectors as columns.

  # Real-space image shifts R = A n (includes the origin).
  rr = np.arange(-real_shells, real_shells + 1)
  ns = np.array(list(itertools.product(rr, repeat=2)))
  Rsh = (A @ ns.T).T
  Rzero = np.linalg.norm(Rsh, axis=1) < 1e-9

  # Reciprocal vectors G = B m (excludes the origin).
  rg = np.arange(-recip_shells, recip_shells + 1)
  ms = np.array([m for m in itertools.product(rg, repeat=2) if m != (0, 0)])
  G = (B @ ms.T).T
  Gn = np.linalg.norm(G, axis=1)
  from scipy.special import erfc as _erfc_np  # host-side kernel precompute.
  IG = (2.0 * eta * np.exp(-(Gn ** 2) / (4.0 * eta ** 2))
        - np.sqrt(np.pi) * Gn * _erfc_np(Gn / (2.0 * eta)))
  self_const = (2.0 * np.sqrt(np.pi) * eta * n_particles ** 2 / area
                - 2.0 * n_particles * eta ** 3 / (3.0 * np.sqrt(np.pi)))

  # Move precomputed constants onto the device.
  A_j = jnp.asarray(A)
  A_inv_j = jnp.asarray(A_inv)
  Rsh_j = jnp.asarray(Rsh)
  self_pair = jnp.asarray(Rzero)  # (nR,) True where R == 0
  G_j = jnp.asarray(G)
  IG_j = jnp.asarray(IG)
  inv_sqrt_pi = float(1.0 / np.sqrt(np.pi))
  sqrt_pi_over_area = float(np.sqrt(np.pi) / area)
  n_elec = int(n_particles)

  def ewald_energy(positions: jnp.ndarray) -> jnp.ndarray:
    """positions: (N, 2) -> scalar periodic 1/(r+a)^3 energy per unit strength."""
    pos = positions.reshape(n_elec, 2)
    # Minimum-image pair displacements under the (possibly non-orthogonal) cell.
    d = pos[:, None, :] - pos[None, :, :]           # (N, N, 2)
    frac = jnp.einsum('ij,nmj->nmi', A_inv_j, d)
    d = jnp.einsum('ij,nmj->nmi', A_j, frac - jnp.round(frac))

    dd = d[:, :, None, :] + Rsh_j[None, None, :, :]  # (N, N, nR, 2)
    eye = jnp.eye(n_elec, dtype=bool)
    self_mask = eye[:, :, None] & self_pair[None, None, :]
    # Guard the (i=j, R=0) self entries before taking the norm: at those points
    # dd = 0, and both norm(0) and 1/r have NaN gradients. Replacing the squared
    # distance by 1 there keeps the value and gradient finite (the term is masked
    # to 0 below regardless).
    r2 = jnp.sum(dd ** 2, axis=-1)                   # (N, N, nR)
    r_safe = jnp.sqrt(jnp.where(self_mask, 1.0, r2))
    z = eta * r_safe
    # Real-space near-field of the 1/(r+a)^3 energy. Naively this is
    #   phi_short(r) + [1/(r+a)^3 - 1/r^3]
    # with phi_short = [erfc(z) + (2/sqrt(pi)) z e^{-z^2}] / r^3, z = eta r. For
    # close particles (small r) the two 1/r^3 pieces are individually huge and
    # cancel, destroying float32 precision (bosons *can* approach closely, so this
    # blows training up). Regroup exactly as
    #   near = 1/(r+a)^3 + [phi_short - 1/r^3] = 1/(r+a)^3 + eta^3 * s(eta r),
    #   s(z) = ((2/sqrt(pi)) z e^{-z^2} - erf(z)) / z^3   (smooth, -> -4/(3 sqrt(pi))),
    # where 1/(r+a)^3 is bounded by 1/a^3 and s(z) is evaluated by a small-z
    # Taylor series (its leading O(z) terms cancel analytically), so no large
    # cancellation ever occurs.
    z2 = z * z
    c = 2.0 * inv_sqrt_pi
    # Closed form of s(z), accurate for z >= 0.5 (guard small z to keep the
    # unused branch's gradient finite).
    z_closed = jnp.where(z < 0.5, 1.0, z)
    s_closed = (c * z_closed * jnp.exp(-(z_closed ** 2))
                - jax.scipy.special.erf(z_closed)) / (z_closed ** 3)
    # Taylor series of s(z) for small z: s = c*(-2/3 + 2/5 z^2 - 1/7 z^4 + 1/27 z^6 - ...)
    s_series = c * (-2.0 / 3.0 + (2.0 / 5.0) * z2
                    - (1.0 / 7.0) * z2 ** 2 + (1.0 / 27.0) * z2 ** 3)
    s = jnp.where(z < 0.5, s_series, s_closed)
    near = 1.0 / (r_safe + a) ** 3 + (eta ** 3) * s
    u_real = 0.5 * jnp.sum(jnp.where(self_mask, 0.0, near))

    phase = pos @ G_j.T                              # (N, nG)
    rho = jnp.sum(jnp.exp(1j * phase), axis=0)       # (nG,)
    u_rec = sqrt_pi_over_area * jnp.sum(
        IG_j * (rho.real ** 2 + rho.imag ** 2))

    return u_real + u_rec + self_const

  return ewald_energy


def potential_electron_nuclear(charges: Array, r_ae: Array, barrier_sharpness=1.) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
  """
  DISK_RADIUS = 10.
  RIM_WIDTH = 0.1 / barrier_sharpness #  0.1 / 2
  BARRIER_HEIGHT = 10 * barrier_sharpness # 2 * 10
  # return 0. * (-jnp.sum(charges / r_ae[..., 0]))             # turning off e-a for now
  # returning a disk potential instead
  potential_energy = jnp.sum(BARRIER_HEIGHT * (1. + jnp.tanh((r_ae[..., 0, 0] - DISK_RADIUS) / RIM_WIDTH)) / 2.)
  # if ~jnp.all(jnp.isfinite(potential_energy)):
  #   raise ValueError('Potential energy is infinite.')
  return 0 * potential_energy # no potential energy with pbc

def potential_nuclear_nuclear(charges: Array, atoms: Array) -> jnp.ndarray:
  """Returns the electron-nuclearpotential.

  Args:
    charges: Shape (natoms). Nuclear charges of the atoms.
    atoms: Shape (natoms, ndim). Positions of the atoms.
  """
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  return 0. * jnp.sum(
      jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))    # turning off a-a for now


def potential_energy(
    r_ae: Array,
    ee: Array,
    r_ee: Array,
    atoms: Array,
    charges: Array,
    interaction_strength: float,
    interaction_small_length_cutoff: float = 0.1,
    barrier_sharpness: float = 1.,
    lattice: Optional[Array] = None,
    interaction_truncation_limit: int = 5,
    interaction_cutoff_radius: Optional[float] = None,
) -> jnp.ndarray:
  """Returns the potential energy for this electron configuration.

  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """
  return (potential_electron_electron(
              ee,
              r_ee,
              interaction_strength=interaction_strength,
              interaction_small_length_cutoff=interaction_small_length_cutoff,
              lattice=lattice,
              interaction_truncation_limit=interaction_truncation_limit,
              interaction_cutoff_radius=interaction_cutoff_radius) +
          potential_electron_nuclear(charges, r_ae, barrier_sharpness=barrier_sharpness) +
          potential_nuclear_nuclear(charges, atoms))


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    interaction_strength: float = 0.0,
    interaction_small_length_cutoff: float = 0.1,
    interaction_truncation_limit: int = 5,
    barrier_sharpness=1.,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    states: int = 0,
    state_specific: bool = False,
    pp_type: str = 'ccecp',
    pp_symbols: Sequence[str] | None = None,
    lattice: Optional[jnp.ndarray] = None,
    interaction_cutoff_radius: Optional[float] = None,
    interaction_tail_correction: bool = False,
    interaction_tail_g2_r: Optional[jnp.ndarray] = None,
    interaction_tail_g2_values: Optional[jnp.ndarray] = None,
    interaction_ewald: bool = False,
    interaction_ewald_eta: Optional[float] = None,
    interaction_ewald_eta_scale: float = 1.0,
    interaction_ewald_real_shells: int = 3,
    interaction_ewald_recip_shells: int = 8,
) -> LocalEnergy:
  """Creates the function to evaluate the local energy.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian
    states: Number of excited states to compute. If 0, compute ground state with
      default machinery. If 1, compute ground state with excited state machinery
    state_specific: Only used for excited states (states > 0). If true, then
      the local energy is computed separately for each output from the network,
      instead of the local energy matrix being computed.
    pp_type: type of pseudopotential to use. Only used if ecp_symbols is
      provided.
    pp_symbols: sequence of element symbols for which the pseudopotential is
      used.
    lattice: Shape (ndim, ndim). Lattice vectors for periodic boundary conditions.
      If provided, electron-electron distances are computed using minimum-image
      convention. If None, standard Euclidean distances are used.
    interaction_cutoff_radius: Optional spherical cutoff R_c applied to the
      explicit electron-electron pair sum. When None (default) the pair sum is
      unchanged (standard behaviour). When set, only pairs within R_c are summed
      explicitly and the long-range remainder is supplied by the tail
      correction below.
    interaction_tail_correction: When True, add the isotropic large-r tail of
      the 1/(r+r0)^3 pair energy beyond ``interaction_cutoff_radius`` to the
      potential energy. Requires ``lattice`` and ``interaction_cutoff_radius``.
      Defaults to False, leaving the standard config untouched.
    interaction_tail_g2_r, interaction_tail_g2_values: Optional tabulated radial
      pair-correlation function g2(r) used to weight the tail integral. When not
      supplied, g2(r) = 1 (uniform fluid) is assumed.
    interaction_ewald: When True, evaluate the electron-electron 1/(r+a)^3 energy
      by exact 2D Ewald summation (`make_ewald_1r3_potential`) instead of the
      minimum-image pair sum. Requires `lattice`; mutually exclusive with the
      tail correction. Defaults to False, leaving the standard config untouched.
    interaction_ewald_eta, interaction_ewald_eta_scale,
    interaction_ewald_real_shells, interaction_ewald_recip_shells: Ewald
      convergence controls (see `make_ewald_1r3_potential`). The energy is
      independent of eta; the defaults are balanced for the HEX cells here.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  n_particles = sum(nspins)
  del nspins

  # Long-range tail correction is a configuration-independent constant per unit
  # interaction strength; precompute it here (NumPy, outside jit).
  tail_coeff = 0.0
  if interaction_tail_correction:
    if lattice is None or interaction_cutoff_radius is None:
      raise ValueError(
          'interaction_tail_correction requires both `lattice` and '
          '`interaction_cutoff_radius` to be set.')
    tail_coeff = electron_electron_tail_coefficient(
        lattice=lattice,
        n_particles=n_particles,
        interaction_small_length_cutoff=interaction_small_length_cutoff,
        interaction_cutoff_radius=interaction_cutoff_radius,
        g2_r=interaction_tail_g2_r,
        g2_values=interaction_tail_g2_values,
    )

  # Optional exact Ewald evaluator for the electron-electron 1/(r+a)^3 energy.
  ewald_energy_fn = None
  if interaction_ewald:
    if lattice is None:
      raise ValueError('interaction_ewald requires `lattice` to be set.')
    if interaction_tail_correction:
      raise ValueError('interaction_ewald and interaction_tail_correction are '
                       'mutually exclusive; enable only one.')
    if states:
      raise NotImplementedError('Ewald summation is not implemented for '
                                'excited states.')
    ewald_energy_fn = make_ewald_1r3_potential(
        lattice=lattice,
        n_particles=n_particles,
        interaction_small_length_cutoff=interaction_small_length_cutoff,
        eta=interaction_ewald_eta,
        eta_scale=interaction_ewald_eta_scale,
        real_shells=interaction_ewald_real_shells,
        recip_shells=interaction_ewald_recip_shells,
    )

  if not pp_symbols:
    effective_charges = charges
    use_pp = False
  else:
    effective_charges, pp_local, pp_nonlocal = pp.make_pp_potential(
        charges=charges,
        symbols=pp_symbols,
        quad_degree=4,
        ecp=pp_type,
        complex_output=complex_output
    )
    use_pp = not jnp.all(effective_charges == charges)

  if not use_pp:
    pp_local = lambda *args, **kwargs: 0.0
    pp_nonlocal = lambda *args, **kwargs: 0.0

  def _e_l(
      params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
  ) -> Tuple[jnp.ndarray, LocalEnergyAux]:
    """Returns the total energy and per-configuration kinetic/potential split.
    """
    if states:
      # Compute features
      vmap_features = jax.vmap(networks.construct_input_features, (0, None))
      positions = jnp.reshape(data.positions, [states, -1])
      ae, ee, r_ae, r_ee = vmap_features(positions, data.atoms)

      # Compute potential energy (use per-walker interaction_strength from data)
      vmap_pot = jax.vmap(
          potential_energy,
          (0, 0, 0, None, None, None, None, None, None, None, None))
      pot_spectrum = vmap_pot(
          r_ae,
          ee,
          r_ee,
          data.atoms,
          effective_charges,
          data.interaction_strength,
          interaction_small_length_cutoff,
          barrier_sharpness,
          lattice,
          interaction_truncation_limit,
          interaction_cutoff_radius)[:, None]
      if interaction_tail_correction:
        # Scalar per walker (states index excited states of one walker).
        pot_spectrum += data.interaction_strength * tail_coeff

      if use_pp:
        data_vmap_dims = networks.FermiNetData(
            positions=0, spins=0, atoms=None, charges=None)
        data_ = networks.FermiNetData(
            positions=positions,
            spins=jnp.reshape(data.spins, [states, -1]),
            atoms=data.atoms,
            charges=data.charges,
        )
        pot_spectrum += jax.vmap(pp_local, (0,))(r_ae)[:, None]    # pseudopotential part, can do without it
        vmap_pp_nonloc = jax.vmap(
            pp_nonlocal, (None, None, None, data_vmap_dims, 0, 0))
        pot_spectrum += vmap_pp_nonloc(key, f, params, data_, ae, r_ae)

      # Combine terms
      if state_specific:
        # For simplicity, we will only implement a folx version of the kinetic    # folx for kinetic energy (laplacian)!
        # energy calculation here.
        # TODO(pfau): factor out code repeated here and in _lapl_over_f
        pos_ = jnp.reshape(data.positions, [states, -1])
        spins_ = jnp.reshape(data.spins, [states, -1])
        f_closure = lambda x: f(params, x, spins_[0], data.atoms, data.charges,
                                 data.interaction_strength)
        f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
        sign_out, log_out = folx.batched_vmap(f_wrapped, 1)(pos_)
        kin = -(log_out.laplacian +
                jnp.sum(log_out.jacobian.dense_array ** 2, axis=-2)) / 2
        if complex_output:
          kin -= 0.5j * sign_out.laplacian
          kin += 0.5 * jnp.sum(sign_out.jacobian.dense_array ** 2, axis=-2)
          kin -= 1.j * jnp.sum(sign_out.jacobian.dense_array *
                               log_out.jacobian.dense_array, axis=-2)
        total_energy = jnp.diag(kin) + pot_spectrum[:, 0]
        energy_mat = None
        kinetic_scalar = jnp.sum(jnp.diag(kin))
        potential_scalar = jnp.sum(pot_spectrum[:, 0])
      else:
        # Compute kinetic energy and matrix of states
        ke = excited_kinetic_energy_matrix(
            f, states, complex_output, laplacian_method)
        psi_mat, kin_mat = ke(params, data)
        hpsi_mat = kin_mat + psi_mat * pot_spectrum
        energy_mat = jnp.linalg.solve(psi_mat, hpsi_mat)
        total_energy = jnp.trace(energy_mat)
        kin_per_state = jnp.linalg.solve(psi_mat, kin_mat)
        kinetic_scalar = jnp.trace(kin_per_state)
        potential_scalar = total_energy - kinetic_scalar
    else:
      ke = local_kinetic_energy(f,
                                use_scan=use_scan,
                                complex_output=complex_output,
                                laplacian_method=laplacian_method)
      ae, ee, r_ae, r_ee = networks.construct_input_features(
          data.positions, data.atoms
      )
      # Use periodic distances if lattice is provided
      if lattice is not None:
        r_ee = compute_periodic_r_ee(ee, lattice)
      # Use per-walker interaction_strength from data (enables multi-λ training)
      walker_interaction_strength = data.interaction_strength
      if ewald_energy_fn is not None:
        # Exact Ewald electron-electron energy replaces the pair sum. The
        # (zeroed) electron-nuclear / nuclear-nuclear terms are kept for parity.
        potential = (
            walker_interaction_strength * ewald_energy_fn(data.positions) +
            potential_electron_nuclear(
                effective_charges, r_ae, barrier_sharpness=barrier_sharpness) +
            potential_nuclear_nuclear(effective_charges, data.atoms) +
            pp_local(r_ae) +
            pp_nonlocal(key, f, params, data, ae, r_ae))
      else:
        potential = (potential_energy(
                        r_ae,
                        ee,
                        r_ee,
                        data.atoms,
                        effective_charges,
                        interaction_strength=walker_interaction_strength,
                        interaction_small_length_cutoff=interaction_small_length_cutoff,
                        barrier_sharpness=barrier_sharpness,
                        lattice=lattice,
                        interaction_truncation_limit=interaction_truncation_limit,
                        interaction_cutoff_radius=interaction_cutoff_radius) +
                     pp_local(r_ae) +
                     pp_nonlocal(key, f, params, data, ae, r_ae))
        if interaction_tail_correction:
          # Configuration-independent long-range tail (scales with walker's k).
          potential = potential + walker_interaction_strength * tail_coeff
      kinetic = ke(params, data)
      total_energy = potential + kinetic
      energy_mat = None  # Not necessary for ground state
      kinetic_scalar = kinetic
      potential_scalar = potential
    return total_energy, LocalEnergyAux(
        kinetic=kinetic_scalar,
        potential=potential_scalar,
        energy_mat=energy_mat,
    )

  return _e_l
