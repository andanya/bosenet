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

"""Multiplicative Jastrow factors."""

import enum
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


_EULER_GAMMA = 0.5772156649015329


def _k0_small_series_coeffs(n_terms: int = 10) -> tuple:
  """Returns (H_k / (k!)^2) for k = 1..n_terms, used in the K_0 small-x series:
    K_0(z) = -(log(z/2) + gamma) * I_0(z) + Sum_{k=1}^inf H_k (z/2)^{2k} / (k!)^2.
  10 terms give accuracy << float32 epsilon for 0 < z <= 2.
  """
  coeffs = []
  harmonic = 0.0
  factorial = 1.0
  for k in range(1, n_terms + 1):
    harmonic += 1.0 / k
    factorial *= k
    coeffs.append(harmonic / (factorial * factorial))
  return tuple(coeffs)


_K0_SERIES_COEFFS = _k0_small_series_coeffs(10)

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()
  # 2D bosonic dipole-dipole Jastrow following Astrakharchik et al.,
  # Phys. Rev. Lett. 98, 060405 (2007). Two-piece form:
  #   f_2(r) = K_0(2*sqrt(r_0/r))                          r <= R_match
  #   f_2(r) = C_2 * [exp(-C_3/r) + exp(-C_3/(L-r))]        r >  R_match
  # with r_0 = interaction_strength, L = shortest lattice vector length,
  # value continuity at R_match fixing C_2, and C_3 trainable.
  DIPOLE_2D = enum.auto()


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps."""
  r_ees = [
      jnp.split(r, nspins[0:1], axis=1)
      for r in jnp.split(r_ee, nspins[0:1], axis=0)
  ]
  r_ees_parallel = jnp.concatenate([
      r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
      r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
  ])

  if r_ees_parallel.shape[0] > 0:
    jastrow_ee_par = jnp.sum(
        jastrow_fun(r_ees_parallel, 0.25, params['ee_par'])
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  if r_ees[0][1].shape[0] > 0:
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees[0][1], 0.5, params['ee_anti']))
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow() -> ...:
  """Creates a Jastrow factor for electron-electron cusps."""

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    params['ee_par'] = jnp.ones(
        shape=1,
    )
    params['ee_anti'] = jnp.ones(
        shape=1,
    )
    return params

  def apply(
      r_ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
      ee: Optional[jnp.ndarray] = None,
      interaction_strength: Union[float, jnp.ndarray] = 0.0,
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    del ee, interaction_strength
    return _jastrow_ee(r_ee, params, nspins, jastrow_fun=simple_ee_cusp_fun)

  return init, apply


def _log_k0(x: jnp.ndarray) -> jnp.ndarray:
  """Log of the modified Bessel function K_0(x) for x > 0.

  For x <= 2, uses the exact series identity
    K_0(x) = -(log(x/2) + gamma) I_0(x) + Sum_{k>=1} H_k (x/2)^{2k}/(k!)^2,
  truncated at 10 terms (accuracy << float32 epsilon on this branch), with
  I_0 supplied by ``jax.scipy.special.i0``. For x > 2 uses the A&S 9.8.6
  asymptotic polynomial for sqrt(x)*exp(x)*K_0(x) in 2/x.

  Both branches are evaluated on inputs clipped to their valid domain so
  that gradient flow through ``jnp.where`` stays finite even when the
  unselected branch would otherwise be evaluated outside its domain.
  """
  safe_x = jnp.maximum(x, 1e-30)
  use_small = safe_x <= 2.0

  # Feed each branch only its own domain to keep gradients finite.
  x_small = jnp.where(use_small, safe_x, jnp.asarray(2.0, dtype=safe_x.dtype))
  x_large = jnp.where(use_small, jnp.asarray(2.0, dtype=safe_x.dtype), safe_x)

  # x <= 2 branch: series expansion using jax-supplied I_0.
  half = x_small * 0.5
  u = half * half
  series = jnp.asarray(0.0, dtype=safe_x.dtype)
  for c in reversed(_K0_SERIES_COEFFS):
    series = series * u + c
  series = series * u  # series = Sum_{k>=1} H_k * u^k / (k!)^2
  i0 = jax.scipy.special.i0(x_small)
  k0_small = -(jnp.log(half) + _EULER_GAMMA) * i0 + series
  log_k0_small = jnp.log(jnp.maximum(k0_small, 1e-30))

  # x > 2 branch (A&S 9.8.6): sqrt(x) * e^x * K_0(x) = poly(2/x)
  y = 2.0 / x_large
  sxe = (1.25331414
      + y * (-0.07832358
      + y * (0.02189568
      + y * (-0.01062446
      + y * (0.00587872
      + y * (-0.00251540
      + y * 0.00053208))))))
  log_k0_large = jnp.log(jnp.maximum(sxe, 1e-30)) - 0.5 * jnp.log(x_large) - x_large

  return jnp.where(use_small, log_k0_small, log_k0_large)


def make_dipole_2d_jastrow(
    lattice: Optional[jnp.ndarray] = None,
    rmatch_frac: float = 0.25,
) -> Tuple[Callable, Callable]:
  """Two-body Jastrow for 2D bosonic dipoles.

  Following Astrakharchik et al., Phys. Rev. Lett. 98, 060405 (2007):
    f_2(r) = K_0(2*sqrt(r_0/r))                          r <= R_match
    f_2(r) = C_2 * [exp(-C_3/r) + exp(-C_3/(L-r))]        r >  R_match
  where r_0 = interaction_strength (the characteristic dipolar length in
  units where m = hbar = 1 and the interaction is r_0/r^3), L is the
  shortest lattice-vector length, and C_2 is fixed by value continuity
  at r = R_match. C_3 is a trainable scalar (one per network).

  The minimum-image distance under the (periodic) lattice is used for
  each pair so that the Jastrow is periodic in the simulation cell.
  """
  if lattice is None:
    raise ValueError('DIPOLE_2D Jastrow requires `lattice`.')

  lattice_arr = jnp.asarray(lattice, dtype=jnp.float32)
  reciprocal = jnp.linalg.inv(lattice_arr)
  # Use the shortest lattice vector length as the cell "diameter" L.
  L_val = float(np.min(np.linalg.norm(np.asarray(lattice), axis=0)))

  R_match = rmatch_frac * L_val
  L_const = jnp.asarray(L_val, dtype=jnp.float32)
  R_const = jnp.asarray(R_match, dtype=jnp.float32)

  def init() -> Mapping[str, jnp.ndarray]:
    # log_c3 trainable; init so C_3 = R_match.
    return {'log_c3': jnp.asarray(np.log(R_match), dtype=jnp.float32)}

  def apply(
      r_ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
      ee: Optional[jnp.ndarray] = None,
      interaction_strength: Union[float, jnp.ndarray] = 0.0,
  ) -> jnp.ndarray:
    del nspins  # bosonic system: all pairs treated identically
    n = r_ee.shape[0]
    iu, ju = jnp.triu_indices(n, k=1)
    if ee is not None:
      pair_disp = ee[iu, ju]
      frac = jnp.einsum('ij,pj->pi', reciprocal, pair_disp)
      pair_disp = jnp.einsum('ij,pj->pi', lattice_arr, frac - jnp.round(frac))
      r_pairs = jnp.linalg.norm(pair_disp, axis=-1)
    else:
      r_pairs = r_ee[iu, ju, 0]

    # Safe r in (eps, L - eps) to keep the symmetrized form finite.
    eps = jnp.asarray(1e-6, dtype=r_pairs.dtype)
    r_safe = jnp.clip(r_pairs, eps, L_const - eps)

    r0 = jnp.asarray(interaction_strength, dtype=r_pairs.dtype)
    r0_safe = jnp.maximum(r0, 1e-12)

    # Short-range: log K_0(2 sqrt(r0/r))
    u = 2.0 * jnp.sqrt(r0_safe / r_safe)
    log_short = _log_k0(u)

    # Long-range: log[exp(-C3/r) + exp(-C3/(L-r))] + log C2
    c3 = jnp.exp(params['log_c3'])
    log_long_raw = jnp.logaddexp(-c3 / r_safe, -c3 / (L_const - r_safe))
    # Value continuity at R_match
    u_R = 2.0 * jnp.sqrt(r0_safe / R_const)
    log_short_at_R = _log_k0(u_R)
    log_long_at_R = jnp.logaddexp(-c3 / R_const, -c3 / (L_const - R_const))
    log_c2 = log_short_at_R - log_long_at_R
    log_long = log_long_raw + log_c2

    log_f = jnp.where(r_safe <= R_const, log_short, log_long)
    # Zero contribution when interaction is effectively off.
    log_f = jnp.where(r0 > 1e-12, log_f, jnp.zeros_like(log_f))
    return jnp.sum(log_f)

  return init, apply


def get_jastrow(jastrow: JastrowType, **kwargs) -> ...:
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    jastrow_init, jastrow_apply = make_simple_ee_jastrow()
  elif jastrow == JastrowType.DIPOLE_2D:
    jastrow_init, jastrow_apply = make_dipole_2d_jastrow(**kwargs)
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

  return jastrow_init, jastrow_apply
