import numpy as np
from scipy.integrate import solve_ivp
from scipy.misc import derivative
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import brentq

LST = open("shrodinger-2b.txt", "wt")

# потенциальная функция
def U(x, L, U0, W):
    return np.where(np.abs(x) < L, U0, W)

# функция ф-ла (13)
def q(e_arg, x, L, U0, W):
    return 2.0 * (e_arg - U(x, L, U0, W))

# вычисление правых частей системы ОДУ 1-го порядка (интегрирование "вперёд")
def system1(cond, x, eee, L, U0, W):
    Y0, Y1 = cond
    dY0dX = Y1
    dY1dX = -q(eee, x, L, U0, W)*Y0
    return [dY0dX, dY1dX]

# вычисление правых частей системы ОДУ 1-го порядка (интегрирование "назад")
def system2(cond, x, eee, L, U0, W):
    Z0, Z1 = cond
    dZ0dX = Z1
    dZ1dX = -q(eee, x, L, U0, W)*Z0
    return [dZ0dX, dZ1dX]

# вычисление разности производных в узле сшивки (формула (18))
def f_fun(e_arg, X, XX, r, rr, L, U0, W):
    eee = e_arg
    # Решение задачи Коши ("вперед")
    cond1 = np.asarray([0.0, 1.0], dtype="float64")
    sol1 = solve_ivp(lambda x, y: system1(y, x, eee, L, U0, W),
                     [X[0], X[-1]], cond1, t_eval=X, method='RK45')
    Psi, Psi1 = sol1.y[0], sol1.y[1]

    # Решение задачи Коши ("назад")
    cond2 = np.asarray([0.0, 1.0], dtype="float64")
    sol2 = solve_ivp(lambda x, y: system2(y, x, eee, L, U0, W),
                     [XX[0], XX[-1]], cond2, t_eval=XX, method='RK45')
    Fi, Fi1 = sol2.y[0], sol2.y[1]

    # поиск максимального по величине элемента Psi
    p1 = np.abs(Psi).max()
    p2 = np.abs(Psi).min()
    big = p1 if p1 > p2 else p2
    # Масштабирование Psi
    Psi[:] = Psi[:] / big
    # Математическая нормировка Fi для достижения равенства F[rr] = Psi[r]
    coef = Psi[r] / Fi[rr]
    Fi[:] = coef * Fi[:]

    # вычисление f(E) для узла сшивки, формула (18)
    curve1 = interp1d(X, Psi, kind='cubic')
    curve2 = interp1d(X, Fi, kind='cubic')
    der1 = derivative(curve1, X[r], dx=1.e-6)
    der2 = derivative(curve2, XX[rr], dx=1.e-6)
    f = der1 - der2
    return f, Psi, Fi

# Функция для нахождения корня при помощи brentq (пункт 7)
def find_root(x1, x2, X, XX, r, rr, L, U0, W):
    root = brentq(lambda e: f_fun(e, X, XX, r, rr, L, U0, W)[0], x1, x2)
    return root

# функция для вывода графика f(E)
def plotting_f(U0, e2, fmin, fmax, eee, af, energy, nroots):
    plt.axis((U0, e2, fmin, fmax))
    ZeroE = np.zeros(len(eee), dtype=float)
    plt.plot(eee, ZeroE, 'k-', linewidth=1.0)
    plt.plot(eee, af, 'bo', markersize=2)
    for i in range(nroots):
        plt.plot([energy[i]], [0.0], color='red', marker='*', markersize=10)
    plt.xlabel("E", fontsize=18, color="k")
    plt.ylabel("f(E)", fontsize=18, color="k")
    plt.grid(True)
    # Сохранение в файл
    plt.savefig('schrodinger-2b-f.pdf', dpi=300)
    plt.show()

# Функция для вывода графика волновых функций и потенциала
def plotting_wf(e_arg, X, XX, r, rr, L, U0, W, ngr):
    ff, Psi, Fi = f_fun(e_arg, X, XX, r, rr, L, U0, W)
    plt.axis((X[0], X[-1], U0, W))
    Upot = np.array([U(X[i], L, U0, W) for i in np.arange(len(X))])
    plt.plot(X, Upot, 'g-', linewidth=6.0, label="U(x)")
    Zero = np.zeros(len(X), dtype=float)
    plt.plot(X, Zero, 'k-', linewidth=1.0)
    plt.plot(X, Psi, 'r-', linewidth=2.0, label="Psi(x)")
    plt.plot(XX, Fi, 'b-', linewidth=2.0, label="Fi(x)")
    plt.xlabel("X", fontsize=18, color="k")
    plt.ylabel("Psi(x), Fi(x), U(x)", fontsize=18, color="k")
    plt.grid(True)
    plt.legend(fontsize=16, shadow=True, fancybox=True, loc='upper right')
    plt.plot([X[r]], [Psi[r]], color='red', marker='o', markersize=7)
    string1 = "E    = " + format(e_arg, "10.7f")
    string2 = "f(e) = " + format(ff, "10.3e")
    plt.text(-1.5, 2.7, string1, fontsize=14, color='black')
    plt.text(-1.5, 2.3, string2, fontsize=14, color='black')
    # Сохранение в файл
    name = "shrodinger-2b-" + str(ngr) + ".pdf"
    plt.savefig(name, dpi=300)
    plt.show()

# Основная часть
L = 2.0
A = -L
B = +L
n = 1001
print(f"n={n}")
print(f"n={n}", file=LST)
U0 = -1.0
W = 4.0
X = np.asarray(np.linspace(A, B, n), dtype="float64")  # Для интегрирования "вперед"
XX = np.asarray(np.linspace(A, B, n), dtype="float64") # Для интегрирования "назад"
r = (n - 1) * 3 // 4    # для Psi
rr = n - r - 1          # для Fi
print(f"r = {r}")
print(f"r = {r}", file=LST)
print(f"r = {rr}")
print(f"r = {rr}", file=LST)
print(f"X[r] = {X[r]}")
print(f"X[r] = {X[r]}", file=LST)
print(f"XX[r] = {XX[r]}")
print(f"XX[r] = {XX[r]}", file=LST)
e1 = U0 + 0.05
e2 = 3.0
print(f"e1 = {e1}")
print(f"e1 = {e1}", file=LST)
print(f"e1 = {e2}")
print(f"e1 = {e2}", file=LST)
ne = 101
print(f"ne = {ne}")
print(f"ne = {ne}", file=LST)
eee = np.linspace(e1, e2, ne)
af = np.zeros(ne, dtype=float)
porog = 5.0
tol = 1.0e-7
energy = []
ngr = 0

# Цикл поиска простых корней f(e) на отрезке [e1, e2]
for i in np.arange(ne):
    e = eee[i]
    f_val, Psi, Fi = f_fun(e, X, XX, r, rr, L, U0, W)
    af[i] = f_val
    stroka = "i = {:3d}   e = {:8.5f}  f[e] = {:12.5e}"
    print(stroka.format(i, e, af[i]))
    print(stroka.format(i, e, af[i]), file=LST)
    if i > 0:
        Log1 = af[i] * af[i - 1] < 0.0
        Log2 = np.abs(af[i] - af[i - 1]) < porog
        if Log1 and Log2:
            energy1 = eee[i - 1]
            energy2 = eee[i]
            # Используем brentq для нахождения корня
            bis_eval = find_root(energy1, energy2, X, XX, r, rr, L, U0, W)
            print("eval = {:12.5e}".format(bis_eval))
            # Вызов plotting_wf сохранен
            _ = plotting_wf(bis_eval, X, XX, r, rr, L, U0, W, ngr)
            energy.append(bis_eval)
            ngr += 1

# Вывод значения корней уравнения f(e) = 0
nroots = len(energy)
print(f"nroots ={nroots}")
print(f"nroots ={nroots}", file=LST)
for i in np.arange(nroots):
    stroka = "i = {:1d}    energy[i] = {:12.5e}"
    print(stroka.format(i, energy[i]))
    print(stroka.format(i, energy[i]), file=LST)

fmax = +10.0
fmin = -10.0
# Вызов plotting_f сохранен
_ = plotting_f(U0, e2, fmin, fmax, eee, af, energy, nroots)

LST.close()
