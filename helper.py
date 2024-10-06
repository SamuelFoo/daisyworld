import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

from constants import *


def kelvin(celsius: ArrayLike):
    return celsius + 273.15


def celsius(kelvin: ArrayLike):
    return kelvin - 273.15


def surface_area(radius: ArrayLike):
    return 4 * np.pi * radius**2


def average_temperature(area_fraction: np.ndarray, temperatures: np.ndarray) -> float:
    return (area_fraction * temperatures).sum()


def energy_flux_gain(
    albedo: ArrayLike,
    solar_flux_along_normal: ArrayLike,
    solar_multiplier: float = 1.0,
):
    return solar_flux_along_normal * (1 - albedo) * solar_multiplier


def energy_flux_loss(temperature: ArrayLike, average_temperature: float):
    return A + B * celsius(temperature) + G * (temperature - average_temperature)


def albedo(temperature: np.ndarray):
    albedo = np.zeros_like(temperature)
    threshold = kelvin(-10)
    albedo[temperature >= threshold] = 0.3
    albedo[temperature < threshold] = 0.6
    return albedo


def beta(D_f: float, T_opt: float, T: ArrayLike):
    """Daisy's growth

    Args:
        D_f (float): Daisy Death Factor.
        T_opt (float): Optimal temperature.
        T (float): Daisy's local temperature.

    Returns:
        float: daisy's growth
    """
    return 1 - D_f * (T_opt - T) ** 2


def _A_g(A_w: float, A_b: float):
    return 1 - A_w - A_b


def _T_p(A_w, A_b, L):
    alpha_p = A_w * alpha_w + A_b * alpha_b + _A_g(A_w=A_w, A_b=A_b) * alpha_g
    return (L * S_0 / sigma * (1 - alpha_p)) ** (1 / 4)


def __solve(L, A_w, A_b, D_f, T_opt, dense_output=False):
    local_T = lambda alpha_p, alpha_i, T_p: (
        R * L * S_0 / sigma * (alpha_p - alpha_i) + T_p**4
    ) ** (1 / 4)

    def f(t, y):
        A_w, A_b = y
        A_g = x = 1 - A_w - A_b
        alpha_p = A_w * alpha_w + A_b * alpha_b + A_g * alpha_g
        T_p = (L * S_0 / sigma * (1 - alpha_p)) ** (1 / 4)

        T_w = local_T(alpha_p, alpha_i=alpha_w, T_p=T_p)
        T_b = local_T(alpha_p, alpha_i=alpha_b, T_p=T_p)

        beta_w = beta(D_f=D_f, T_opt=T_opt, T=T_w)
        beta_b = beta(D_f=D_f, T_opt=T_opt, T=T_b)

        return [A_w * (x * beta_w - gamma), A_b * (x * beta_b - gamma)]

    t_f = 50
    sol = solve_ivp(f, [0, t_f], y0=[A_w, A_b], dense_output=dense_output)
    return sol


def solve_dense(L, A_w, A_b, D_f, T_opt):
    sol = __solve(L=L, A_w=A_w, A_b=A_b, D_f=D_f, T_opt=T_opt, dense_output=True)
    sol_fn = sol.sol
    t_f = 50
    ts = np.linspace(0, t_f, 1000)
    A_w, A_b = sol_fn(ts)

    alpha_p = A_w * alpha_w + A_b * alpha_b + _A_g(A_w=A_w, A_b=A_b) * alpha_g
    assert np.all(alpha_p < 1)
    T_p = (L * S_0 / sigma * (1 - alpha_p)) ** (1 / 4)
    return ts, T_p, A_w, A_b


def solve(L, A_w, A_b, D_f, T_opt):
    sol = __solve(L=L, A_w=A_w, A_b=A_b, D_f=D_f, T_opt=T_opt)
    A_w, A_b = sol.y
    alpha_p = A_w * alpha_w + A_b * alpha_b + _A_g(A_w=A_w, A_b=A_b) * alpha_g
    T_p = (L * S_0 / sigma * (1 - alpha_p)) ** (1 / 4)
    return sol.t, T_p, A_w, A_b


def solve_varying_L_quasi_static(Ls, A_w0, A_b0, D_f, T_opt):
    T_ps = []
    A_ws, A_bs = [], []
    A_w = A_w0
    A_b = A_b0

    for L in Ls:
        _, T_p_against_time, A_w_against_time, A_b_against_time = solve(
            L, A_w=A_w, A_b=A_b, D_f=D_f, T_opt=T_opt
        )
        A_w = max(A_w_against_time[-1], 0.01)
        A_b = max(A_b_against_time[-1], 0.01)
        A_ws.append(A_w)
        A_bs.append(A_b)
        T_ps.append(T_p_against_time[-1])

    return T_ps, A_ws, A_bs


def years_to_seconds(num_years: float):
    return num_years * 365 * 24 * 60 * 60


def seconds_to_years(seconds: float):
    return seconds / (365 * 24 * 60 * 60)
