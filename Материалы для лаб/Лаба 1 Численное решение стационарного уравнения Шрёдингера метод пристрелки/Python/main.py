import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre


# Потенциальная функция на основе полинома Лагерра
def u(x):
    if abs(x) < L:
        return V0 * eval_genlaguerre(5, 0, abs(x))  # Полином Лагерра L_5(x)
    else:
        return W  # Бесконечный потенциал за пределами ямы


# Функция q(E, x) для преобразования потенциала
def q(e, x):
    return 2.0 * (e - u(x))


# Численное вычисление производной (формула 19)
def deriv(Y, h, m):
    return (Y[m - 2] - Y[m + 2] + 8.0 * (Y[m + 1] - Y[m - 1])) / (12.0 * h)


# Вычисление разности производных в узле сшивки (формула 18)
def f_fun(e, r, n):
    F = np.array([c * q(e, X[i]) for i in range(n)])
    Psi[0] = 0.0
    Fi[n - 1] = 0.0
    Psi[1] = d1
    Fi[n - 2] = d2

    # Решение задачи Коши "вперед" методом Нумерова
    for i in range(1, n - 1):
        p1 = 2.0 * (1.0 - 5.0 * F[i]) * Psi[i]
        p2 = (1.0 + F[i - 1]) * Psi[i - 1]
        Psi[i + 1] = (p1 - p2) / (1.0 + F[i + 1])

    # Решение задачи Коши "назад" методом Нумерова
    for i in range(n - 2, 0, -1):
        f1 = 2.0 * (1.0 - 5.0 * F[i]) * Fi[i]
        f2 = (1.0 + F[i + 1]) * Fi[i + 1]
        Fi[i - 1] = (f1 - f2) / (1.0 + F[i - 1])

    # Масштабирование волновых функций
    big = max(np.abs(Psi).max(), np.abs(Psi).min())
    Psi[:] = Psi[:] / big
    coef = Psi[r] / Fi[r]
    Fi[:] = coef * Fi[:]

    # Разность производных
    f = deriv(Psi, h, r) - deriv(Fi, h, r)
    return f


# Основные параметры задачи
L = 3.0  # Полуширина потенциальной ямы (Å)
A: float = -L  # Левая граница
B = L  # Правая граница
n = 501  # Количество узлов сетки
h = (B - A) / (n - 1)  # Шаг сетки
c = h ** 2 / 12.0  # Константа для метода Нумерова
V0 = 25.0  # Амплитуда потенциала (eV)
W = 1e9  # Бесконечный потенциал за пределами ямы

# Граничные условия
d1 = 1e-9  # Малая величина для Psi[1]
d2 = d1  # Малая величина для Fi[n-2]

# Массивы для расчета
Psi = np.zeros(n)
Fi = np.zeros(n)
X = np.linspace(A, B, n)
r = (n - 1) // 2 + 15  # Узел сшивки

# Ввод энергии и расчет f(E)
E = float(input("Energy E = "))
print("E =", E)
f = f_fun(E, r, n)
print("f(E) =", f)

# Построение графика
Upot = np.array([u(X[i]) for i in range(n)])
plt.axis([A - 0.1, B + 0.1, -5.0, V0 + 5.0])
Zero = np.zeros(n, dtype=float)
plt.plot(X, Zero, 'k-', linewidth=1.0)
plt.plot(X, Upot, 'g-', linewidth=6.0, label="U(x)")
plt.plot(X[1:n - 1], Psi[1:n - 1], 'r-', linewidth=2.0, label="Psi(x)")
plt.plot(X[1:n - 1], Fi[1:n - 1], 'b-', linewidth=2.0, label="Phi(x)")
plt.xlabel("X", fontsize=20, color="k")
plt.ylabel("Psi(x), Phi(x), U(x)", fontsize=20, color="k")
plt.grid(True)
plt.legend(fontsize=16, shadow=True, fancybox=True)
plt.plot([X[r]], [Psi[r]], color='red', marker='o', markersize=10)
plt.text(-L, V0 - 5, f"E = {E:.7f}", fontsize=16, color="black")
plt.text(-L, V0 - 10, f"f(E) = {f:.7e}", fontsize=16, color="black")
plt.savefig("PotentialLaguerre.pdf", dpi=300)
plt.show()
