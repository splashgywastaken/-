import numpy as np
from scipy.integrate import odeint
from scipy.misc import derivative
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
global r, n, Psi, Fi, X, XX, eee

LST = open("shrodinger-2b.txt", "wt")


# потенциальная функция, рис.3
# (на рис. 3 "а" соответствует "L")
def U(x):
    return np.where(np.abs(x) < L, U0, W)
    # return float(U0 if any(np.abs(x) < L) else W)


def q(e_arg, x):
    return 2.0 * (e_arg - U(x))


# вычисление правых частей системы ОДУ 1-го порядка
# (интегрирование "вперёд")
def system1(cond1, X_arg):
    Y0, Y1 = cond1[0], cond1[1]
    dY0dX = Y1
    dY1dX = - q(ee, X_arg) * Y0
    return [dY0dX, dY1dX]


# вычисление правых частей системы ОДУ 1-го порядка
# (интегрирование "назад")
def system2(cond2, XX_arg):
    Z0, Z1 = cond2[0], cond2[1]
    dZ0dX = Z1
    dZ1dX = - q(ee, XX_arg) * Z0
    return [dZ0dX, dZ1dX]


# вычисление разности производных в узле сшивки
# формула (18)
def f_fun(e_arg):
    eee = e_arg
    """
    Решение задачи Коши ("вперед")
    dPsi1(x)/dx = - q(e, x)*Psi(x);
    dPsi(x)/dx = Psi1(x);
    Psi(A) = 0.0
    Psi1(A) = 1.0
    """
    cond1 = [0.0, 1.0]
    sol1 = odeint(system1, cond1, X)
    Psi, Psi1 = sol1[:, 0], sol1[:, 1]
    """
    Решение задачи Коши ("назад")
    dPsi1(x)/dx = - q(e, x)*Psi(x);
    dPsi(x)/dx = Psi1(x);
    Psi(B) = 0.0
    Psi1(B) = 1.0
    """
    cond2 = [0.0, 1.0]
    sol2 = odeint(system2, cond2, XX)
    Fi, Fi1 = sol2[:, 0], sol2[:, 1]
    # поиск максимального по величине элемента Psi
    p1 = np.abs(Psi).max()
    p2 = np.abs(Psi).min()
    big = p1 if p1 > p2 else p2
    # Масштабирование Psi
    Psi[:] = Psi[:] / big
    # Математическая нормировка Fi для
    # достижения равенства F[rr] = Psi[r]
    coef = Psi[r] / Fi[rr]
    Fi[:] = coef * Fi[:]
    # вычисление f(E) для узла сшивки, формула (18)
    curve1 = interp1d(X, Psi, kind='cubic')
    curve2 = interp1d(X, Fi, kind='cubic')
    der1 = derivative(curve1, X[r], dx=1.e-6)
    der2 = derivative(curve2, XX[rr], dx=1.e-6)
    f = der1 - der2
    return f


# Функция для решения уравнения f(E) = 0 методом бисекций
def m_bis(x1, x2, tol_arg):
    if f_fun(e_arg=x2)*f_fun(e_arg=x1) > 0.0:
        print("ERROR no root!!!")
        print(f"x1 = {x1}")
        print(f"x2 = {x2}")
        print(f"f_fun(e=x1,r=r,n=n)={f_fun(e_arg=x1)}")
        print(f"f_fun(e=x2,r=r,n=n)={f_fun(e_arg=x2)}")
        exit()
    while abs(x2-x1) > tol_arg:
        xr = (x1 + x2) / 2.0
        if f_fun(e_arg=x2)*f_fun(e_arg=xr) < 0.0:
            x1 = xr
        else:
            x2 = xr
        if f_fun(e_arg=x1)*f_fun(e_arg=xr) < 0.0:
            x2 = xr
        else:
            x1 = xr
    return (x1 + x2) / 2.0


# функция для вывода графика f(E)
def plotting_f():
    plt.axis([U0, e2, fmin, fmax])
    ZeroE = np.zeros(ne, dtype=float)
    plt.plot(ee, ZeroE, 'k-', linewidth=1.0)
    plt.plot(ee, af, 'bo', markersize=2)
    for i in range(nroots):
        plt.plot([energy[i]], [0.0], color='red', marker='*', markersize=10)
        plt.xlabel("E", fontsize=18, color="k")
        plt.ylabel("f(E)", fontsize=18, color="k")
        plt.grid(True)
        # Сохранение в файл
        plt.savefig('schrodinger-2b-f.pdf', dpi=300)
        plt.show()


# Функция для вывода графика волновых функций и потенциала
def plotting_wf(e_arg):
    ff = f_fun(e_arg)
    plt.axis([A - 0.01, B + 0.01, U0, W])
    Upot = np.array([U(X[i]) for i in np.arrange(n)])
    plt.plot(X, Upot, 'g-', linewidth=6.0, label="U(x)")
    Zero = np.zeros(n, dtype=float)
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


# Задание отрезка [A, B] (каря ямы)
L = 2.0
A = -L
B = +L
# кол-во узлов сетки на отрезке
n = 1001
print(f"n={n}")
print(f"n={n}", file=LST)
# минимальное значение потенциальной функции
U0 = -1.0
# максимальное значение потенциальной функции
W = 4.0
# х-координаты узлов сетки
X = np.linspace(A, B, n)  # Для интегрирования "вперед"
XX = np.linspace(A, B, n)  # Для интегрирования "назад"
# номер узла сшивки
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
# график f(e)
e1 = U0 + 0.05
e2 = 3.0
print(f"e1 = {e1}")
print(f"e1 = {e1}", file=LST)
print(f"e1 = {e2}")
print(f"e1 = {e2}", file=LST)
ne = 101
print(f"ne = {ne}")
print(f"ne = {ne}", file=LST)
ee = np.linspace(e1, e2, ne)
af = np.zeros(ne, dtype=float)
porog = 5.0
tol = 1.0e-7
energy = []
ngr = 0
# Цикл поиска простых корней f(e) на отрезке [e1, e2]
for i in np.arange(ne):
    e = ee[i]
    af[i] = f_fun(e)
    stroka = "i = {:3d}   e = {:8.5f}  f[e] = {:12.5e}"
    print(stroka.format(i, e, af[i]))
    print(stroka.format(i, e, af[i]), file=LST)
    if i > 0:
        Log1 = af[i] * af[i - 1] < 0.0
        Log2 = np.abs(af[i] - af[i - 1]) < porog
        if Log1 and Log2:
            energy1 = ee[i - 1]
            energy2 = ee[i]
            bis_eval = m_bis(energy1, energy2, tol)
            print("eval = {:12.5e}".format(bis_eval))
            _ = plotting_wf(bis_eval)
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
_ = plotting_f()
