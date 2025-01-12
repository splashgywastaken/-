"""
Крутько А.С. ЧИСЛЕННОЕ РЕШЕНИЕ СТАЦИОНАРНОГО УРАВНЕНИЯ ШРЁДИНГЕРА: ТЕОРИЯ ВОЗМУЩЕНИЙ.
Программа написана на языке Python 3.12, в среде разработки PyCharm Community Edition 2024.3.1,
операционная система Windows 10.
"""
from typing import TextIO

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn


def u0(x):
    if abs(x) < L:
        return V0 * jn(2, x)
    else:
        return 4.0


def q(e, x, potential_func):
    return 2.0 * (e - potential_func(x))


def find_derivative(Y, h, m):
    return (Y[m - 2] - Y[m + 2] + 8.0 * (Y[m + 1] - Y[m - 1])) / (12.0 * h)


def normalize_wave_function(Y):
    norm = np.sqrt(np.trapz(Y**2, X))
    return Y / norm


def mean_momentum(Psi, X):
    hbar = 1.0
    dPsi_dx = np.gradient(Psi, X)
    integrand = Psi.conj() * dPsi_dx
    mean_Px = -1j * hbar * np.trapz(integrand, X)
    return mean_Px.real


def mean_square_momentum(Psi, X):
    hbar = 1.0
    d2Psi_dx2 = np.gradient(np.gradient(Psi, X), X)
    integrand = Psi.conj() * d2Psi_dx2
    mean_Px2 = -hbar**2 * np.trapz(integrand, X)
    return mean_Px2.real


def f_fun(e, n, potential_func):
    F = np.array([c * q(e, X[i], potential_func) for i in np.arange(n)])
    Psi.fill(0.0)
    Psi[0] = 0.0
    Fi[n - 1] = 0.0
    Psi[1] = d1
    Fi[n - 2] = d2

    for i in np.arange(1, n - 1, 1):
        p1 = 2.0 * (1.0 - 5.0 * F[i]) * Psi[i]
        p2 = (1.0 + F[i - 1]) * Psi[i - 1]
        Psi[i + 1] = (p1 - p2) / (1.0 + F[i + 1])

    for i in np.arange(n - 2, 0, -1):
        f1 = 2.0 * (1.0 - 5.0 * F[i]) * Fi[i]
        f2 = (1.0 + F[i + 1]) * Fi[i + 1]
        Fi[i - 1] = (f1 - f2) / (1.0 + F[i - 1])

    p1 = np.abs(Psi).max()
    p2 = np.abs(Psi).min()
    big = p1 if p1 > p2 else p2

    Psi[:] = Psi[:] / big
    coef = Psi[r] / Fi[r]
    Fi[:] = coef * Fi[:]

    return find_derivative(Psi, h, r) - find_derivative(Fi, h, r)


def energy_scan(E_min, E_max, step, potential_func):
    energies = []
    values = []
    E = E_min
    while E <= E_max:
        f_value = f_fun(E, n, potential_func)
        energies.append(E)
        values.append(f_value)
        E += step
    return energies, values


def find_exact_energies(E_min, E_max, step, tol, potential_func):
    energies, values = energy_scan(E_min, E_max, step, potential_func)
    exact_energies = []
    for i in range(1, len(values)):
        Log1 = values[i] * values[i - 1] < 0.0
        Log2 = np.abs(values[i] - values[i - 1]) < limit
        if Log1 and Log2:
            E1, E2 = energies[i - 1], energies[i]
            exact_energy = bisection_method(E1, E2, tol, potential_func)
            f_fun(exact_energy, n, potential_func)
            exact_energies.append(exact_energy)
    return exact_energies


def bisection_method(E1, E2, tol, potential_func):
    while abs(E2 - E1) > tol:
        Emid = (E1 + E2) / 2.0
        f1, f2, fmid = f_fun(E1, n, potential_func), f_fun(E2, n, potential_func), f_fun(Emid, n, potential_func)
        if f1 * fmid < 0.0:
            E2 = Emid
        else:
            E1 = Emid
        if f2 * fmid < 0.0:
            E1 = Emid
        else:
            E2 = Emid
    return (E1 + E2) / 2.0


def count_zeros(psi, x):
    crossings = 0
    for i in range(1, len(psi) - 1):
        if psi[i - 1] * psi[i] < 0:
            crossings += 1
    return crossings

c_energy = 27.212
c_length = 0.5292
V0 = 25.0 / c_energy
L = 3.0 / c_length
A, B = -L, L
n = 1001
h = (B - A) / (n - 1)
c, W = h ** 2 / 12.0, 3.0
Psi, Fi, X = np.zeros(n), np.zeros(n), np.linspace(A, B, n)
r = (n-1)//2 - 15
limit = 4.0

d1, d2 = 1.e-09, 1.e-09
tol = 1e-6

E_min, E_max, step = -0.1, 9.0, 0.01
exact_energies = find_exact_energies(E_min + 0.001, E_max, step, tol, u0)

if len(exact_energies) == 0:
    print("Error.")
else:
    print("Energies:")
    for i, E in enumerate(exact_energies):
        f_fun(E, n, u0)
        crossings = count_zeros(Psi, X)
        print(f"Energy level {i}: {E:.6f}, cross: {crossings}")

results_file = open(f"./python_results/result.txt", "a")


def U(x):
    if abs(x) < L:
        if 2.5 <= x <= 3.0:
            return 1.5
        else:
            return V0 * jn(2, x)
    else:
        return 4.0


def V(x):
    return U(x) - u0(x)


def e0(k):
    if k < len(exact_energies):
        return exact_energies[k]
    else:
        raise ValueError(f"There is no level with the {k} number")


def psi_interp(k, x):
    f_fun(exact_energies[k], n, u0)
    Psi_copy = normalize_wave_function(Psi.copy())
    return np.interp(x, X, Psi_copy)


def funct(x, k1, k2):
    psi_k1 = psi_interp(k1, x)
    psi_k2 = psi_interp(k2, x)
    return psi_k1 * V(x) * psi_k2


def matel(k1, k2):
    integrand_values = np.array([funct(x, k1, k2) for x in X])
    res = np.trapz(integrand_values, X)
    return res if abs(res) > 1e-14 else 0.0


def e_corr_2(kmax, root):
    s = 0.0
    for k in range(0, root):
        s += matel(root, k)**2 / (e0(root) - e0(k))
        energy_diff = abs(e0(root) - e0(k))
        if abs(matel(root, k)) >= energy_diff:
            print(f"Возможное расхождение: ")
        print("k=", k, "  s=", s)
        print("k=", k, "  s=", s, file=results_file)
    for k in range(root + 1, kmax + 1):
        s += matel(root, k)**2 / (e0(root) - e0(k))
        print("k=", k, "  s=", s)
        print("k=", k, "  s=", s, file=results_file)
    return s


def c_psi_corr_1(kmax, root):
    c = np.zeros(kmax)
    for k in range(0, root):
        c[k] = matel(root, k) / (e0(root) - e0(k))
    for k in range(root + 1, kmax + 1):
        c[k - 1] = matel(root, k) / (e0(root) - e0(k))
    return c


def psi_corr_1(x, c, n, root):
    kmax = len(c)
    s = 0.0
    for k in range(0, root):
        s += c[k] * psi_interp(k, x)
    for k in range(root + 1, kmax + 1):
        s += c[k - 1] * psi_interp(k, x)
    return s


def psi(x, c, n, root):
    return psi_interp(root, x) + psi_corr_1(x, c, n, root)


def result(root):
    kmax = int(len(exact_energies) - 1)
    print("=======================================")
    print("=======================================", file=results_file)
    print(f"State {root}")
    print(f"State {root}", file=results_file)
    print("kmax = ", kmax)
    print("kmax = ", kmax, file=results_file)
    e1 = e0(root) + matel(root, root)
    print("e0(1)=", e0(root))
    print("e0(1)=", e0(root), file=results_file)
    print("e=", e1, "   (1-approximation)")
    print("e=", e1, "   (1-approximation)", file=results_file)
    e2 = e0(root) + matel(root, root) + e_corr_2(kmax, root)
    print("e=", e2, "   (2-approximation)")
    print("e=", e2, "   (2-approximation)", file=results_file)

    c1 = c_psi_corr_1(kmax, root)
    for i in range(len(c1)):
        print("c[{:1d}] = {:15.8e}".format(i, c1[i]))
        print("c[{:1d}] = {:15.8e}".format(i, c1[i]), file=results_file)
    psi_arr = np.array([psi(x, c1, n, root) for x in X])
    E_min2, E_max2, step2 = 0.0, 3.0, 0.01
    exact_energies2 = find_exact_energies(E_min2 + 0.001, E_max2, step2, tol, U)
    f_fun(exact_energies2[root], n, U)
    Psi_copy2 = normalize_wave_function(Psi.copy())
    psi_arr2 = normalize_wave_function(psi_arr.copy())
    psi_density = psi_arr2 ** 2

    mean_Px = mean_momentum(psi_arr2, X)
    mean_Px2 = mean_square_momentum(psi_arr2, X)
    print(f"E_perturbation = {e2:.6f}, <p_x> = {mean_Px:.6e}, <p_x^2> = {mean_Px2:.6e}")
    print(f"E_perturbation = {e2:.6f}, <p_x> = {mean_Px:.6e}, <p_x^2> = {mean_Px2:.6e}", file=results_file)
    mean_Px = mean_momentum(Psi_copy2, X)
    mean_Px2 = mean_square_momentum(Psi_copy2, X)
    print(f"E_target = {exact_energies2[root]:.6f}, <p_x> = {mean_Px:.6e}, <p_x^2> = {mean_Px2:.6e}")
    print(f"E_target = {exact_energies2[root]:.6f}, <p_x> = {mean_Px:.6e}, <p_x^2> = {mean_Px2:.6e}", file=results_file)
    print("=======================================")
    print("=======================================", file=results_file)


    Zero = np.zeros(n, dtype=float)
    Upot = np.array([U(X[i]) for i in np.arange(n)])

    plt.xlabel("x", fontsize=18, color="k")
    plt.ylabel("Psi(x)", fontsize=18, color="k")
    plt.plot(X, Zero, 'k--', linewidth=1.0)
    plt.plot(X, Upot, 'g-', linewidth=3.0, label="U(x)")
    line1, = plt.plot(X, psi_arr2, 'b-', linewidth=3.0, label="")
    line2, = plt.plot(X, psi_density, 'r-', linewidth=3.0, label="")

    plt.subplots_adjust(bottom=0.3)

    plt.figtext(0.5, 0.02,
                f"Psi{root} perturbation method (E = {e2:.4f})\n"
                f"Psi{root} probability density)",
                ha="center", fontsize=10)

    plt.legend([line1, line2],
               [f"Psi{root} perturbation method ", f"Psi{root} probability density"],
               fontsize=7, loc='upper right')

    plt.grid(True)
    plt.twinx()
    plt.yticks([])
    plt.savefig(f"./python_results/State{root} probability density.jpg", dpi=300)
    plt.show()
    plt.close('all')


    plt.xlabel("x", fontsize=18, color="k")
    plt.ylabel("Psi(x)", fontsize=18, color="k")
    plt.plot(X, Zero, 'k--', linewidth=1.0)
    plt.plot(X, Upot, 'g-', linewidth=3.0, label="U(x)")
    line1, = plt.plot(X, psi_arr2, 'b-', linewidth=3.0, label="")
    line2, = plt.plot(X, Psi_copy2, 'r-', linewidth=3.0, label="")

    plt.subplots_adjust(bottom=0.3)

    plt.figtext(0.5, 0.02,
                f"Psi{root} perturbation method (E = {e2:.4f})\n"
                f"Psi{root} target method (E = {exact_energies2[root]:.4f})",
                ha="center", fontsize=10)

    plt.legend([line1, line2],
               [f"Psi{root} perturbation method ", f"Psi{root} target method"],
               fontsize=10, loc='upper right')

    plt.grid(True)
    plt.twinx()
    plt.yticks([])
    plt.savefig(f"./python_results/State{root}.jpg", dpi=300)
    plt.show()
    plt.close('all')

result(0)
result(2)

plt.plot(X, [u0(x) for x in X], 'g-', linewidth=3.0, label="U(x)")
plt.xlabel("X")
plt.ylabel("U0(X)")
plt.grid(True)
plt.legend()
plt.savefig("./python_results/U0(X).jpg", dpi=300)
plt.show()

plt.plot(X, [U(x) for x in X], 'g-', linewidth=3.0, label="U(x)")
plt.xlabel("X")
plt.ylabel("U(X)")
plt.grid(True)
plt.legend()
plt.savefig("./python_results/U(X).jpg", dpi=300)
plt.show()