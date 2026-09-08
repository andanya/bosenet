# Copyright 2026 DeepMind Technologies Limited.
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

"""Smooth, periodic, lambda-conditioned two-body Jastrow prefactor (FIX A).

Adds a fixed (non-trainable) multiplicative pair factor to the bosonic
wavefunction that imposes the exact short-range two-body physics of the 2D
dipole:

    log psi  +=  sum_{i<j} log f_2( rho( r~_ij ) ),
    f_2(r) = K_0( 2 sqrt(lambda / r) ),

where lambda = interaction_strength (the dipolar length r_0 in units
hbar = m = 1), r~ is the *smooth periodic distance* (`pbc.feature_layer.
periodic_norm`, C-infinity on the torus, ~ r at short range), and rho is a
smooth saturating map that gently caps the large-r growth without any matching
point.  f_2 is the zero-energy two-body scattering solution for the 1/r^3
potential (Astrakharchik et al., PRL 98, 060405 (2007)): its cusp cancels the
r -> 0 singularity of the local energy, so E_L stays finite at contact and the
network no longer has to represent the correlation hole itself.

Because r~ is smooth everywhere (no minimum-image kink) and rho introduces no
piecewise seam, log psi is C-infinity: the Laplacian estimator picks up the
full kinetic energy with no missing surface term, so the variational bound is
respected (unlike the broken min-image DIPOLE_2D Jastrow this replaces).

The factor is strictly positive (K_0 > 0), so it only shifts log|psi| and never
the sign/phase.  It is applied per-walker with that walker's own lambda, so it
is compatible with multi-lambda ("gen") training.
"""

from typing import Callable, Optional

from ferminet.pbc.feature_layer import periodic_norm
import jax
import jax.numpy as jnp
import numpy as np


_EULER_GAMMA = 0.5772156649015329


def _k0_small_series_coeffs(n_terms: int = 10) -> tuple:
  """Returns (H_k / (k!)^2) for k = 1..n_terms for the K_0 small-x series:

    K_0(z) = -(log(z/2) + gamma) I_0(z) + Sum_{k>=1} H_k (z/2)^{2k} / (k!)^2.

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


def _log_k0(x: jnp.ndarray) -> jnp.ndarray:
  """Log of the modified Bessel function K_0(x) for x > 0.

  For x <= 2, uses the exact series identity
    K_0(x) = -(log(x/2) + gamma) I_0(x) + Sum_{k>=1} H_k (x/2)^{2k}/(k!)^2,
  truncated at 10 terms, with I_0 from ``jax.scipy.special.i0``. For x > 2 uses
  the A&S 9.8.6 asymptotic polynomial for sqrt(x) exp(x) K_0(x) in 2/x. Both
  branches are evaluated on inputs clipped to their valid domain so gradient
  flow through ``jnp.where`` stays finite even when the unselected branch would
  be evaluated outside its domain.
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
  log_k0_large = (jnp.log(jnp.maximum(sxe, 1e-30))
                  - 0.5 * jnp.log(x_large) - x_large)

  return jnp.where(use_small, log_k0_small, log_k0_large)


def make_smooth_periodic_log_jastrow(
    lattice: np.ndarray,
    ndim: int = 2,
    rmatch_frac: float = 1.0,
    eps: float = 1e-12,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
  """Builds the fixed smooth-periodic log-Jastrow evaluator for one walker.

  Args:
    lattice: (ndim, ndim) real-space primitive vectors as columns.
    ndim: spatial dimension (2 here).
    rmatch_frac: sets the saturation length R0 = rmatch_frac * L_min, where
      L_min is the shortest primitive-vector length. rho(r) = R0 * tanh(r/R0)
      equals r for r << R0 (exact short-range asymptotics preserved) and
      saturates smoothly for r >~ R0, so f_2 stops varying near the cell edge
      with no matching point. rmatch_frac >= 1 is a gentle cap; smaller values
      shorten the Jastrow range.
    eps: floor added inside the sqrt to keep gradients finite at contact.

  Returns:
    logJ(pos, interaction_strength) -> scalar, where pos is the flat
    (n_particles * ndim,) position vector of one walker and
    interaction_strength is that walker's scalar lambda.
  """
  lattice = np.asarray(lattice, dtype=np.float64)
  if lattice.shape != (ndim, ndim):
    raise ValueError(
        f'smooth-periodic Jastrow expects a ({ndim},{ndim}) lattice, got '
        f'{lattice.shape}.')
  reciprocal = jnp.asarray(np.linalg.inv(lattice))  # 2pi factor omitted.
  metric = jnp.asarray(lattice.T @ lattice)
  l_min = float(np.min(np.linalg.norm(lattice, axis=0)))
  r0 = float(rmatch_frac) * l_min

  def logJ(pos: jnp.ndarray, interaction_strength: jnp.ndarray) -> jnp.ndarray:
    x = jnp.reshape(pos, (-1, ndim))
    n = x.shape[0]
    iu, ju = jnp.triu_indices(n, k=1)
    disp = x[iu] - x[ju]                             # (P, ndim)
    scaled = jnp.einsum('il,pl->pi', reciprocal, disp)
    r_p = periodic_norm(metric, scaled)             # (P,) smooth periodic dist
    rho = r0 * jnp.tanh(r_p / r0)                    # smooth saturating map
    lam = jnp.maximum(jnp.asarray(interaction_strength), 0.0)
    arg = 2.0 * jnp.sqrt(lam / (rho + eps))
    return jnp.sum(_log_k0(arg))

  return logJ
