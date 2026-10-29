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

@njit(void(float64[:], float64[:], float64[:], float64[:], float64[:,:], float64[:], types.int64, types.int64), cache=True, fastmath=True)
def adjust_tridiag_system_fixed_temp(a, b, c, d, v2_bdy, uk, k, N):
    """
    Adjust the tridiagonal system for fixed-temperature boundary conditions.
    """
    a[0] = 0.0
    b[0] = 1.0
    c[0] = 0.0
    d[0] = 1.5 * v2_bdy[k, 0] - uk[0]

    a[N - 1] = 0.0
    b[N - 1] = 1.0
    c[N - 1] = 0.0
    d[N - 1] = 1.5 * v2_bdy[k, 1] - uk[N - 1]

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