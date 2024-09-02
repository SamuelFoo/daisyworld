import numpy as np
from constants import *
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp


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


def beta(T: ArrayLike):
    """Daisy's growth

    Args:
        T (float): Daisy's local temperature

    Returns:
        float: daisy's growth
    """
    return 1 - 0.003265 * (295.5 - T) ** 2


def solve(L, A_w, A_b):
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

        beta_w = beta(T_w)
        beta_b = beta(T_b)

        return [A_w * (x * beta_w - gamma), A_b * (x * beta_b - gamma)]

    t_f = 50
    sol = solve_ivp(f, [0, t_f], y0=[A_w, A_b])

    A_w, A_b = sol.y
    A_g = x = 1 - A_w - A_b
    alpha_p = A_w * alpha_w + A_b * alpha_b + A_g * alpha_g
    T_p = (L * S_0 / sigma * (1 - alpha_p)) ** (1 / 4)
    return T_p[-1], max(A_w[-1], 0.01), max(A_b[-1], 0.01)


def years_to_seconds(num_years: float):
    return num_years * 365 * 24 * 60 * 60


def seconds_to_years(seconds: float):
    return seconds / (365 * 24 * 60 * 60)
