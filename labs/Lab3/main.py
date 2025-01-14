"""
Крутько А.С. РАСЧЕТ ОСНОВНОГО КВАНТОВОГО СОСТОЯНИЯ ЧАСТИЦЫ В ОДНОМЕРНОЙ ПОТЕНЦИАЛЬНОЙ ЯМЕ
С БЕСКОНЕЧНЫМИ СТЕНКАМИ С ИСПОЛЬЗОВАНИЕМ РАЗЛОЖЕНИЯ ИСКОМОЙ ВОЛНОВОЙ ФУНКЦИИ ПО БАЗИСУ.
Программа написана на языке Python 3.0, в среде разработки PyCharm Community Edition 2024.2.1,
операционная система Windows 10.
"""
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.special import eval_laguerre


def u_func(x):
    return V0 * eval_laguerre(5, x) if abs(x) < L else W


def basis_function(k):
    result = np.zeros(N)
    h = (2 * L) / (N - 1)
    for i in range(N):
        arg = (np.pi * (k + 1) * (-L + i * h)) / (2 * L)
        result[i] = np.sin(arg) / np.sqrt(L) if (k + 1) % 2 == 0 else np.cos(arg) / np.sqrt(L)
    return result


def find_second_derivative(y, h):
    derivative = np.zeros_like(y)
    for i in range(len(y)):
        if i == 0:
            derivative[i] = (2 * y[i] - 5 * y[i + 1] + 4 * y[i + 2] - y[i + 3]) / (h * h)
        elif i == len(y) - 1:
            derivative[i] = (-y[i - 3] + 4 * y[i - 2] - 5 * y[i - 1] + 2 * y[i]) / (h * h)
        else:
            derivative[i] = (y[i - 1] - 2 * y[i] + y[i + 1]) / (h * h)
    return derivative


def h_psi(k):
    psi_k = basis_function(k)
    result = np.zeros(N)
    h = (2 * L) / (N - 1)
    derivative_psi = find_second_derivative(psi_k, h)
    for i in range(N):
        result[i] = derivative_psi[i] / (-2) + u_func(-L + i * h) * psi_k[i]
    return result


def hamiltonian_element(m, k):
    fi_m = basis_function(m)
    h_fi_k = h_psi(k)
    return np.trapz(fi_m * h_fi_k, dx=(2 * L) / (N - 1))


def hamiltonian_matrix():
    inner_h_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            inner_h_matrix[i, j] = hamiltonian_element(i, j)
    return inner_h_matrix


def eigen_solve(inner_h_matrix):
    eigenvalues, eigenvalues_vectors = eigh(inner_h_matrix)
    return eigenvalues, eigenvalues_vectors


def compute_wave_function(coefficient):
    result = np.zeros(N)
    for k in range(M):
        result += coefficient[k] * basis_function(k)
    return result


def mean_momentum(arg_psi, arg_x):
    h_bar = 1.0
    dPsi_dx = np.gradient(arg_psi, arg_x)
    integrand = arg_psi.conj() * dPsi_dx
    mean_Px = -1j * h_bar * np.trapz(integrand, arg_x)
    return mean_Px.real


def mean_square_momentum(arg_psi, arg_x):
    h_bar = 1.0
    d2Psi_dx2 = np.gradient(np.gradient(arg_psi, arg_x), arg_x)
    integrand = arg_psi.conj() * d2Psi_dx2
    mean_Px2 = -h_bar**2 * np.trapz(integrand, arg_x)
    return mean_Px2.real


def plot_wave_functions(arg_energies, arg_wave_functions):
    x_vals = np.linspace(-L, L, N)
    potential = np.array([u_func(x) for x in x_vals])
    for i, psi in enumerate(arg_wave_functions):
        f_fun(exact_energies[i], n)
        Psi_norm = normalize_wavefunction(Psi.copy())
        mean_Px = mean_momentum(Psi_norm, X)
        mean_Px2 = mean_square_momentum(Psi_norm, X)
        mean_P = mean_momentum(psi.copy(), x_vals)
        mean_P2 = mean_square_momentum(psi.copy(), x_vals)
        density_psi = psi ** 2

        plt.plot(x_vals, psi, 'r', label=f"State {i} (E = {arg_energies[i]:.5f})")
        plt.plot(x_vals, density_psi, 'b', label=f"Probability Density {i}")
        plt.plot(x_vals, potential, 'g', label="U(x)")
        plt.xlabel("x")
        plt.ylabel("Psi(x), U(x)")
        plt.legend()
        plt.grid()
        plt.savefig(f"./python_results/State {i} probability density.jpg", dpi=300)
        plt.show()

        print("-----------------------------------------------------")
        print("-----------------------------------------------------", file=output_file)
        print(f"State {i}: E = {arg_energies[i]:.6f}, <p_x> = {mean_Px:.6e}, <p_x^2> = {mean_Px2:.6e}")
        print(f"State {i}: E = {arg_energies[i]:.6f}, <p_x> = {mean_Px:.6e}, <p_x^2> = {mean_Px2:.6e}", file=output_file)
        print(f"State {i}: E_target = {exact_energies[i]:.6f}, <p_x> = {mean_P:.6e}, <p_x^2> = {mean_P2:.6e}")
        print(f"State {i}: E_target = {exact_energies[i]:.6f}, <p_x> = {mean_P:.6e}, <p_x^2> = {mean_P2:.6e}", file=output_file)
        print("-----------------------------------------------------")
        print("-----------------------------------------------------", file=output_file)

        plt.plot(x_vals, psi, 'r', label=f"State {i} (E = {arg_energies[i]:.5f})")
        plt.plot(X, Psi_norm, 'b--', label=f"State {i} (E_target = {exact_energies[i]:.5f})")
        plt.plot(x_vals, potential, 'g', label="U(x)")
        plt.xlabel("x")
        plt.ylabel("Psi(x), U(x)")
        plt.legend()
        plt.grid()
        plt.savefig(f"./python_results/State {i}.jpg",dpi=300)
        plt.show()


def normalize_wave_function(psi):
    dx = (2 * L) / (N - 1)  # Шаг по x
    norm = np.sqrt(np.trapz(psi**2, dx=dx))  # Интеграл по x
    return psi / norm


"""
The Target method
"""

def normalize_wavefunction(Y):
    norm = np.sqrt(np.trapz(Y**2, X))
    return Y / norm

def q(e, x):
    return 2.0 * (e - u_func(x))

def deriv(Y, h, m):
    return (Y[m - 2] - Y[m + 2] + 8.0 * (Y[m + 1] - Y[m - 1])) / (12.0 * h)

def f_fun(e, n):
    F = np.array([c * q(e, X[i]) for i in np.arange(n)])
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

    coefficient = Psi[r] / Fi[r]
    Fi[:] = coefficient * Fi[:]

    return deriv(Psi, h, r) - deriv(Fi, h, r)

def energy_scan(arg_e_min, arg_e_max, arg_step):
    inner_energies = []
    values = []
    E = arg_e_min
    while E <= arg_e_max:
        f_value = f_fun(E, n)
        inner_energies.append(E)
        values.append(f_value)
        E += arg_step
    return inner_energies, values


def find_exact_energies(arg_e_min, arg_e_max, arg_step, arg_tol):
    inner_energies, values = energy_scan(arg_e_min, arg_e_max, arg_step)
    inner_exact_energies = []
    for i in range(1, len(values)):
        Log1 = values[i] * values[i - 1] < 0.0
        Log2 = np.abs(values[i] - values[i - 1]) < limit
        if Log1 and Log2:
            E1, E2 = inner_energies[i - 1], inner_energies[i]
            exact_energy = bisection_method(E1, E2, arg_tol)
            f_fun(exact_energy, n)
            inner_exact_energies.append(exact_energy)
    return inner_exact_energies

def bisection_method(arg_e1, arg_e2, arg_tol):
    while abs(arg_e2 - arg_e1) > arg_tol:
        Emid = (arg_e1 + arg_e2) / 2.0
        f1, f2, f_mid = f_fun(arg_e1, n), f_fun(arg_e2, n), f_fun(Emid, n)
        if f1 * f_mid < 0.0:
            arg_e2 = Emid
        else:
            arg_e1 = Emid
        if f2 * f_mid < 0.0:
            arg_e1 = Emid
        else:
            arg_e2 = Emid
    return (arg_e1 + arg_e2) / 2.0


c_length = 0.5292
c_energy = 27.212

L = 3.0 / c_length
V0 = 25.0 / c_energy
W = 3.0

N = 1001
M = 21

A, B = -L, L
n = 1000
h = (B - A) / (n - 1)
c = h ** 2 / 12.0
Psi, Fi, X = np.zeros(n), np.zeros(n), np.linspace(A, B, n)
r = (n-1)//2 - 80
limit = 7.0

d1, d2 = 1.e-09, 1.e-09
tol = 1e-6
U_min = 0.0
E_min, E_max, step = U_min + 0.01, 3.0, 0.001
exact_energies = find_exact_energies(E_min, E_max, step, tol)


h_matrix = hamiltonian_matrix()
energies, eigenvectors = eigen_solve(h_matrix)

wave_functions = [compute_wave_function(eigenvectors[:, i]) for i in range(3)]
normalized_wave_functions = [normalize_wave_function(psi) for psi in wave_functions]

output_file = open("./python_results/result.txt", "a")
for i, energy in enumerate(energies[:5]):
    print(f"State {i}: E = {energy:.6f}")
    print(f"State {i}: E = {energy:.6f}", file=output_file)

plot_wave_functions(energies, normalized_wave_functions)










