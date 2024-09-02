import numpy as np

# Boltzmann constant
sigma = 5.67037442e-8

# Sun
# Radius
R_S = 696340e3
# Surface temperature
T_S = 5772
# Flux
F_S = sigma * T_S**4

# Radius of planets
# Earth
R_E = 6371e3
# Mars
R_M = 3389.5e3
# Venus
R_V = 6051.8e3

# Constants
G, A, B = 3.8, 204, 2.17

S_0 = 917
L = 1.2
R = 0.2
gamma = 0.3
sigma = 5.67037442e-8

alpha_b = 0.25
alpha_w = 0.75
alpha_g = 0.5

# Area of each band of Earth
area_fraction = np.array(
    [
        0.156434465,
        0.152582529,
        0.144973505,
        0.133794753,
        0.119321529,
        0.101910213,
        0.08198953,
        0.060049992,
        0.036631824,
        0.012311659,
    ]
)

# Local solar flux
local_solar_flux_along_normal = np.array(
    [
        416.6480383,
        406.3887808,
        386.1228828,
        356.349358,
        317.8013293,
        271.4279772,
        218.3711674,
        160.4091221,
        112.7543514,
        73.14499959,
    ]
)
