import numpy as np
from scipy.misc import derivative
from scipy.optimize import minimize

from constants import S_0, R, alpha_b, alpha_g, alpha_w, gamma, sigma
from helper import kelvin


def vb(Ab, Aw, L, D_f=0.003265, T_opt=kelvin(22.5)):
    """
    dA/dt of Black Daisy
    """
    alpha_p = Aw * alpha_w + Ab * alpha_b + (1 - Aw - Ab) * alpha_g
    T_p = (L * (S_0 / sigma) * (1 - alpha_p)) ** 0.25
    T_b = (R * L * (S_0 / sigma) * (alpha_p - alpha_b) + (T_p**4)) ** 0.25
    beta_b = 1 - (D_f * (T_opt - T_b) ** 2)
    return Ab * ((1 - Ab - Aw) * beta_b - gamma)


def vw(Ab, Aw, L, D_f=0.003265, T_opt=kelvin(22.5)):
    """
    dA/dt of White Daisy
    """
    ap = Aw * alpha_w + Ab * alpha_b + (1 - Aw - Ab) * alpha_g
    Te = (L * (S_0 / sigma) * (1 - ap)) ** 0.25
    Tw = (R * L * (S_0 / sigma) * (ap - alpha_w) + (Te**4)) ** 0.25
    bw = 1 - (D_f * (T_opt - Tw) ** 2)
    return Aw * ((1 - Aw - Ab) * bw - gamma)


def equilibrium(L):
    """
    Taken from https://colab.research.google.com/drive/1tmF-7pDIfFTrBC9GTOT_xmd1vc_vh6AU
    """
    def cost(A):
        """
        Auxillary Function for Cost (to be minimized)
        """
        Ab, Aw = A
        return (vb(Ab, Aw, L=L)) ** 2 + (vw(Ab, Aw, L=L)) ** 2

    # Root Finder
    solutions = []
    testvec = np.linspace(0, 1, num=21)
    for Ab0 in testvec:
        for Aw0 in testvec:
            if (Ab0 + Aw0) <= 1:
                A0 = np.array([Ab0, Aw0])
                res = minimize(cost, A0, method="nelder-mead", options={"xatol": 1e-8})
                sb = round(res.x[0], 5)
                sw = round(res.x[1], 5)
                if (
                    ((sb + sw) <= 1)
                    and (sb >= 0)
                    and (sw >= 0)
                    and (cost([sb, sw]) < 1e-7)
                ):
                    if [sb, sw] not in solutions:
                        solutions.append([sb, sw])

    def partial_derivative(func, var, point):
        """
        Auxillary Function for Partial Derivatives
        """
        args = point[:]

        def wraps(x):
            args[var] = x
            return func(*args, L=L)

        return derivative(wraps, point[var], dx=1e-5)

    # Stability Finder
    nature = []
    for s in solutions:
        eb, ew = s
        j00 = partial_derivative(vb, var=0, point=[eb, ew])
        j01 = partial_derivative(vb, var=1, point=[eb, ew])
        j10 = partial_derivative(vw, var=0, point=[eb, ew])
        j11 = partial_derivative(vw, var=1, point=[eb, ew])
        jacobian = np.array([[j00, j01], [j10, j11]])
        eigval, _ = np.linalg.eig(jacobian)
        if eigval[0] < 0 and eigval[1] < 0:
            nature.append("Stable")
        else:
            nature.append("Unstable")

    solutions = np.array(solutions)
    nature = np.array(nature)
    stable_points = solutions[nature == "Stable"]
    unstable_points = solutions[nature == "Unstable"]
    return stable_points, unstable_points
