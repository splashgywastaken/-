"""
Вычисление собственных значений и собственных функций
оператора Гамильтона методом пристрелки.
Одномерная потенциальная яма с бесконечными стенками.
Атомные единицы Хартри.
Версия 1.
"""
import numpy as np
import matplotlib.pyplot as plt

# потенциальная функция, рис.3
# (на рис.3 "a" соответствует "L")
def U(x):
    return float(U0 if abs(x) < L else W)

# функция, ф-ла (13)
def q(e, x):
    return 2.0*(e-U(x))

# численное вычисление производной, ф-ла (19)
def deriv(Y, h, m):
    return (Y[m-2]-Y[m+2]+8.0*(Y[m+1]-Y[m-1]))/(12.0*h)

# вычисление разности производных в узле сшивки, формула (18)
def f_fun(e, r, n):
    F = np.array([c*q(e, X[i]) for i in np.arange(n)])
    Psi[0] = 0.0
    Fi[n-1] = 0.0
    Psi[1] = d1
    Fi[n-2] = d2

    # решение задачи Коши "вперед" методом Нумерова
    for i in np.arange(1, n-1, 1):
        p1 = 2.0*(1.0 - 5.0*F[i])*Psi[i]
        p2 = (1.0 + F[i-1])*Psi[i-1]
        Psi[i+1] = (p1 - p2)/(1.0 + F[i+1])

    # решение задачи Коши "назад" методом Нумерова
    for i in np.arange(n-2, 0, -1):
        f1 = 2.0*(1.0 - 5.0*F[i])*Fi[i]
        f2 = (1.0 + F[i+1])*Fi[i+1]
        Fi[i-1] = (f1 - f2)/(1.0 + F[i-1])

    # поиск максимального по величине элемента Psi
    p1 = np.abs(Psi).max()
    p2 = np.abs(Psi).min()
    big = p1 if p1 > p2 else p2

    # масштабирование Psi
    Psi[:] = Psi[:]/big

    # математическая нормировка Fi для достижения равенства F[r]=Psi[r]
    coef = Psi[r]/Fi[r]
    Fi[:] = coef*Fi[:]

    # вычисление f(E) для узла сшивки, формула (18)
    f = deriv(Psi, h, r) - deriv(Fi, h, r)
    return f

# задание отрезка [A, B] (края ямы)
L = 2.0
A = -L
B = +L
# кол-во узлов сетки на [A, B]
n = 501
# шаг сетки
h = (B-A)/(n-1)
# константа для использования в методе Нумерова
c = h**2/12.0
# минимальное значение потенциальной функции
U0 = -1.0
# максимальное значение потенциальной функции на графике
W = 4.0

Psi = np.zeros(n)
Fi = np.zeros(n)
F = np.zeros(n)
Psi2 = np.zeros(n)
X = np.linspace(A, B, n)

# номер узла сшивки
r = (n-1)//2 + 15

d1 = 1.e-9
d2 = d1

# ввод пристрелочного значения энергии
e = float(input("Energy="))
print("e=", e)
f = f_fun(e, r, n)
print("f=", f)

Upot = np.array([U(X[i]) for i in np.arange(n)])
# построение графика
plt.axis([A, B, U0, W])
Zero = np.zeros(n, dtype=float)
plt.plot(X, Zero, 'k-', linewidth=1.0)
plt.plot(X, Upot, 'g-', linewidth=6.0, label="U(x)")
plt.plot(X[1:n-1], Psi[1:n-1], 'r-', linewidth=2.0, label="Psi(x)")
plt.plot(X[1:n-1], Fi[1:n-1], 'b-', linewidth=2.0, label="Phi(x)")

plt.xlabel("X", fontsize=20, color="k")
plt.ylabel("Psi(x),Φ(x),U(x)", fontsize=20, color="k")
plt.grid(True)
plt.legend(fontsize=16, shadow=True, fancybox=True)
plt.plot([X[r]], [Psi[r]], color='red', marker='o', markersize=10)
string1 = "E = " + str(e)
string2 = "f(E) = " + str(f)
plt.text(-1.5, 2.1, string1, fontsize=16, color='black')
plt.text(-1.5, 1.7, string2, fontsize=16, color="black")
# сохранение графика в файл
plt.savefig('Schrodinger-1M.pdf', dpi=300)
# вывод графика в окно
plt.show()
