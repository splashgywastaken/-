"""
Крутько А.С. ЧИСЛЕННОЕ РЕШЕНИЕ СТАЦИОНАРНОГО УРАВНЕНИЯ ШРЁДИНГЕРА: МЕТОД ПРИСТРЕЛКИ
Программа написана на языке Python 3.0, в среде разработки PyCharm Community Edition 2024.2.1,
операционная система Windows 10.
"""
import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_laguerre

def draw_potential_graph():
    n = 500
    c_energy = 27.212
    c_length = 0.5292
    v0 = 25.0 / c_energy
    l = 3.0 / c_length
    a, b = -l, l
    x = np.linspace(a - 0.01, b + 0.01, n)

    def u_func():
        u_val = np.zeros(n)
        for i in range(n):
            if np.abs(x[i]) <= l:
                u_val[i] = v0 * eval_laguerre(5, np.abs(x[i]))
            else:
                u_val[i] = l

        return u_val

    y = u_func()

    plt.plot(x, y, 'g-', linewidth=6.0, label="U(x)")
    plt.title(f"График потенциальной функции")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()

    plt.savefig('Potential_func_graph.jpg')
    plt.show()


class Solver:
    # Params
    def __init__(self):
        self.U_min = -0.149124
        self.c_energy = 27.212
        self.c_length = 0.5292
        self.V0 = 25.0 / self.c_energy
        self.L = 3.0 / self.c_length
        self.A, self.B = -self.L - 0.01, self.L + 0.01
        # self.A, self.B = -self.L + 3, self.L - 3
        self.n = 1000
        self.h = (self.B - self.A) / (self.n - 1)
        self.c, self.W = self.h ** 2 / 12.0, 3.0
        self.Psi, self.Fi, self.X = np.zeros(self.n), np.zeros(self.n), np.linspace(self.A, self.B, self.n)
        self.r = (self.n - 1) // 2 - 100
        self.limit_value = 4.0

        self.d1, self.d2 = 1.e-09, 1.e-09
        self.tol = 1.e-9

        self.E_min, self.E_max, self.step = self.U_min + 0.01, 0.9, 0.001


    def u_func(self, x):
        # Проверяем, скаляр ли x
        if np.isscalar(x):
            # x - число
            return self.V0 * eval_laguerre(5, abs(x)) if abs(x) <= self.L else self.W
        u_val = np.zeros(self.n)
        for i in range(self.n):
            if np.abs(x[i]) <= self.L:
                u_val[i] = self.V0 * eval_laguerre(5, np.abs(x[i]))
            else:
                u_val[i] = self.W
        return u_val

    def q(self, e, x):
        return 2.0 * (e - self.u_func(x))


    @staticmethod
    def derivative_func(y, h, m):
        return (y[m - 2] - y[m + 2] + 8.0 * (y[m + 1] - y[m - 1])) / (12.0 * h)


    def normalize_wave_function(self, y):
        norm = np.sqrt(np.trapz(y ** 2, self.X))
        return y / norm


    @staticmethod
    def mean_momentum(psi, x):
        h_bar = 1.0
        d_psi_dx = np.gradient(psi, x)
        integrand = psi.conj() * d_psi_dx
        mean_px = -1j * h_bar * np.trapz(integrand, x)
        return mean_px.real


    @staticmethod
    def mean_square_momentum(psi, x):
        h_bar = 1.0
        d2_psi_dx2 = np.gradient(np.gradient(psi, x), x)
        integrand = psi.conj() * d2_psi_dx2
        mean_px2 = -h_bar**2 * np.trapz(integrand, x)
        return mean_px2.real

    def f_fun(self, e, n):
        f = np.array([self.c * self.q(e, self.X[i]) for i in np.arange(n)])
        self.Psi[0] = 0.0
        self.Fi[n - 1] = 0.0
        self.Psi[1] = self.d1
        self.Fi[n - 2] = self.d2

        for i in np.arange(1, n - 1, 1):
            p1 = 2.0 * (1.0 - 5.0 * f[i]) * self.Psi[i]
            p2 = (1.0 + f[i - 1]) * self.Psi[i - 1]
            self.Psi[i + 1] = (p1 - p2) / (1.0 + f[i + 1])

        for i in np.arange(n - 2, 0, -1):
            f1 = 2.0 * (1.0 - 5.0 * f[i]) * self.Fi[i]
            f2 = (1.0 + f[i + 1]) * self.Fi[i + 1]
            self.Fi[i - 1] = (f1 - f2) / (1.0 + f[i - 1])

        p1 = np.abs(self.Psi).max()
        p2 = np.abs(self.Psi).min()
        big = p1 if p1 > p2 else p2

        self.Psi[:] = self.Psi[:] / big

        coefficient = self.Psi[self.r] / self.Fi[self.r]
        self.Fi[:] = coefficient * self.Fi[:]

        return Solver.derivative_func(self.Psi, self.h, self.r) - Solver.derivative_func(self.Fi, self.h, self.r)

    def energy_scan(self, e_min, e_max, step):
        energies = []
        values = []
        e = e_min
        while e <= e_max:
            f_value = self.f_fun(e, self.n)
            energies.append(e)
            values.append(f_value)
            e += step
        return energies, values

    def find_exact_energies(self, e_min, e_max, step, tol):
        energies, values = self.energy_scan(e_min, e_max, step)
        exact_energies = []
        for i in range(1, len(values)):
            log1 = values[i] * values[i - 1] < 0.0
            log2 = np.abs(values[i] - values[i - 1]) < self.limit_value
            if log1 and log2:
                e1, e2 = energies[i - 1], energies[i]
                exact_energy = self.bisection_method(e1, e2, tol)
                self.f_fun(exact_energy, self.n)
                exact_energies.append(exact_energy)
        return exact_energies

    def bisection_method(self, e1, e2, tol):
        while abs(e2 - e1) > tol:
            e_mid = (e1 + e2) / 2.0
            f1, f2, f_mid = self.f_fun(e1, self.n), self.f_fun(e2, self.n), self.f_fun(e_mid, self.n)
            if f1 * f_mid < 0.0:
                e2 = e_mid
            else:
                e1 = e_mid
            if f2 * f_mid < 0.0:
                e1 = e_mid
            else:
                e2 = e_mid
        return (e1 + e2) / 2.0

    def plot_wave_functions(self, energies):
        for i, E in enumerate(energies):
            self.f_fun(E, self.n)
            psi_norm = self.normalize_wave_function(self.Psi.copy())
            fi_norm = self.normalize_wave_function(self.Fi.copy())
            mean_px = Solver.mean_momentum(fi_norm, self.X)
            mean_px2 = Solver.mean_square_momentum(fi_norm, self.X)
            file = open("result.txt", "w")
            file.close()
            file1 = open("result.txt", "a")
            print(f"Состояние {i}: E = {E:.6f}, <p_x> = {mean_px:.6e}, <p_x^2> = {mean_px2:.6e}")
            print(f"Состояние {i}: E = {E:.6f}, <p_x> = {mean_px:.6e}, <p_x^2> = {mean_px2:.6e}", file = file1)

            plt.scatter(self.X[self.r], psi_norm[self.r], color='red', s=50, zorder=5)  # Точка на Psi
            plt.scatter(self.X[self.r], fi_norm[self.r], color='blue', s=50, zorder=5)  # Точка на Fi
            plt.plot(self.X, [self.u_func(x) for x in self.X], 'g-', linewidth=6.0, label="U(x)")
            plt.plot(self.X, psi_norm, label=f"Нормализованное состояние Пси {i}")
            plt.plot(self.X, fi_norm, '--', label=f"Нормализованное состояние Фи {i}")
            plt.title(f"Состояние {i} (Нормализованное) при E = {E:.4f}")
            plt.xlabel("X")
            plt.ylabel("Нормализованные волновые функции")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"Condition_{i}_(normalized).jpg", dpi=300)
            plt.show()


            prob_density_psi = psi_norm**2
            prob_density_fi = fi_norm**2
            plt.plot(self.X, [self.u_func(x) for x in self.X], 'g-', linewidth=6.0, label="U(x)")
            plt.plot(self.X, prob_density_psi, label=f"Вероятностная плотность Пси состояния {i+1}")
            plt.plot(self.X, prob_density_fi, '--', label=f"Probability Density Фи состояния {i+1}")
            plt.title(f"Состояние {i} - Вероятностная плотность при E = {E:.4f}")
            plt.xlabel("X")
            plt.ylabel("Вероятностная плотность")
            plt.grid(True)
            plt.legend()
            plt.savefig(f"Condition_{i}_(Probability_density).jpg", dpi=300)
            plt.show()


    def solve(self):
        e_min, e_max, step = self.U_min + 0.01, 3.0, 0.01
        exact_energies = self.find_exact_energies(e_min, e_max, step, self.tol)

        if len(exact_energies) == 0:
            print("Ошибка: энергии не были найдены.")
        else:
            print("Энергии:")
            for i, E in enumerate(exact_energies):
                print(f"Состояние {i}: Энергия = {E:.6f}")

            self.plot_wave_functions(exact_energies)
