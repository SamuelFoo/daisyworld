import numpy as np

sigma = 5.67037442e-8

# Sun
R_S = 696340e3
T_S = 5772 # Surface temperature
F_S = sigma * T_S**4

# Earth
R_E = 6371e3
d_E = 151.31e9

# Mars
R_M = 3389.5e3
d_M = 218.58e9

# Venus
R_V = 6051.8e3
d_V = 107.95e9


def surface_area(radius):
    return 4 * np.pi * radius**2


# Constants
G, A, B = 3.8, 204, 2.17

# Array containing area of each band and its local solar flux
area_fraction = np.array([0.156434465, 0.152582529, 0.144973505, 0.133794753, 0.119321529, 0.101910213, 0.08198953, 0.060049992, 0.036631824, 0.012311659])
local_solar_flux_along_normal = np.array([416.6480383, 406.3887808, 386.1228828, 356.349358, 317.8013293, 271.4279772, 218.3711674, 160.4091221, 112.7543514, 73.14499959])

def average_temperature(area_fraction: np.ndarray, temperatures: np.ndarray):
    return (area_fraction * temperatures).sum()

def energy_flux_gain(albedo, solar_flux_along_normal, solar_multiplier=1.0):
    return solar_flux_along_normal * (1 - albedo) * solar_multiplier

def energy_flux_loss(temperature, average_temperature):
    return A + B * celsius(temperature) + G * (temperature - average_temperature)

def kelvin(celsius):
    return celsius + 273.15

def celsius(kelvin):
    return kelvin - 273.15

def albedo(temperature):
    albedo = np.zeros_like(temperature)
    threshold = kelvin(-10)
    albedo[temperature >= threshold] = 0.3
    albedo[temperature < threshold] = 0.6
    return albedo
