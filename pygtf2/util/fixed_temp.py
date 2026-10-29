"""
Routines for thermal_boundary == FIXED_TEMPERATURE
"""

import numpy as np 
from numba import njit, float64, types, void
import math

@njit(float64(float64, float64, float64, float64, float64, float64,), cache=True, fastmath=True,)
def _linearized_interface_luminosity(coef, pref, uL, uR, duL, duR,):
    """
    Conductive luminosity linearized about the pre-solve energies.
    This is used for the FIXED_TEMPERATURE boundary condition

    coef includes interface density, the BH limiter, r**2,
    and the inverse distance between cell centers.
    """
    sqrt2 = 1.41421356237309

    su = uL + uR
    diff = uR - uL

    inv_sqrt_su = 1.0 / math.sqrt(su)
    inv_su32 = inv_sqrt_su / su

    common_p = inv_sqrt_su + 0.5 * diff * inv_su32
    common_m = inv_sqrt_su - 0.5 * diff * inv_su32

    return sqrt2 * pref * coef * (
        -diff * inv_sqrt_su
        + common_p * duL
        - common_m * duR
    )

@njit(
    void(
        float64[:],     # a
        float64[:],     # b
        float64[:],     # c
        float64[:],     # d
        float64[:],     # rk
        float64[:],     # mk
        float64[:],     # rhok_int
        float64[:],     # uk
        float64[:, :],  # v2_bdy
        float64[:],     # tau_bdy
        float64,        # pref
        float64,        # dt
        types.int64,    # k
        types.int64,    # N
    ),
    cache=True,
    fastmath=True,
)
def adjust_tridiag_system_fixed_temp(
    a, b, c, d,
    rk, mk, rhok_int, uk,
    v2_bdy, tau_boundary,
    pref, dt, k, N,
):
    """
    Add finite-time coupling to fixed-temperature reservoirs.

    The first and last stellar cells exchange heat with reservoirs
    whose target v2 values are fixed. The stellar-cell temperatures
    themselves are not fixed.
    """
    for side in range(2):
        cell = 0 if side == 0 else N - 1
        i = 1 if side == 0 else N - 1

        # Conductive coefficient at the neighboring interior interface.
        drc = 0.5 * (rk[i + 1] - rk[i - 1])
        coef = rhok_int[i - 1] * rk[i] * rk[i] / drc

        # Restore the endpoint row to shell-energy-balance units.
        scale = -math.sqrt(2.0) * pref * dt * coef

        a[cell] *= scale
        b[cell] *= scale
        c[cell] *= scale
        d[cell] *= scale

        # Add the implicit reservoir contribution.
        dm = mk[cell + 1] - mk[cell]
        term = dm * dt / tau_boundary[side]

        b[cell] += term
        d[cell] += term * (
            1.5 * v2_bdy[k, side] - uk[cell]
        )

@njit(void(float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:, :], float64, float64, types.int64, types.int64), cache=True, fastmath=True)
def accumulate_reservoir_heat(dq_res_trial, rk, mk, rhok_int, uk, du_trial, pref, dt, k, N):
    """
    Accumulate the heat exchanged with the reservoirs at the boundaries
    """
    for side in range(2):
        if side == 0:
            j = 0
            cell = 0
        else:
            j = N - 2
            cell = N - 1

        i = j + 1

        drc = 0.5 * (rk[i + 1] - rk[i - 1])
        coef = rhok_int[j] * rk[i] * rk[i] / drc

        lum_lin = _linearized_interface_luminosity(
            coef, pref,
            uk[j], uk[j + 1],
            du_trial[k, j], du_trial[k, j + 1],
        )

        dm = mk[cell + 1] - mk[cell]
        dq = dm * du_trial[k, cell]

        if side == 0:
            dq += dt * lum_lin
        else:
            dq -= dt * lum_lin

        dq_res_trial[k, side] += dq

@njit(void(float64[:, :], float64[:, :], float64[:, :], float64[:, :],), cache=True, fastmath=True,)
def restore_boundary_v2(v2, m, v2_bdy, dq_res_trial,):
    """
    Heat exchange and binary heating can change the boundary temperatures independently of conduction.
    For a FIXED_TEMPERATURE thermal boundary, we want to restore this.
    """
    s, N = v2.shape

    for k in range(s):
        for side in range(2):
            cell = 0 if side == 0 else N - 1

            dm = m[k, cell + 1] - m[k, cell]
            delta = v2_bdy[k, side] - v2[k, cell]
            dq = 1.5 * dm * delta

            dq_res_trial[k, side] += dq

            v2[k, cell] = v2_bdy[k, side]

@njit(void(float64[:, :], float64[:, :], float64[:, :], float64[:], float64[:, :], types.int64, types.int64), cache=True, fastmath=True,)
def add_pdv_to_reservoir(r, m, v2_bdy, vol_old, dq_res_trial, k, N):
    """
    To accumulate reservoir heat after revirialization
    """
    for side in range(2):
        cell = 0 if side == 0 else N - 1

        V_old = vol_old[cell]
        V_new = (
            r[k, cell + 1]**3
            - r[k, cell]**3
        )

        dm = m[k, cell + 1] - m[k, cell]

        dq = (
            dm * v2_bdy[k, side]
            * math.log(V_new / V_old)
        )

        dq_res_trial[k, side] += dq