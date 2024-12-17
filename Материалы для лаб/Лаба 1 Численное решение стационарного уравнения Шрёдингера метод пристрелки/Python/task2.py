"""
Вычисление энергий квантовых состояний
частицы в прямоугольной потенциальной яме
с бесконечными стенками по формуле (8)
"""
import numpy as np

def Ek(k):
    return Umin + np.pi**2 * (k**2 / (8 * L**2))

L = 2.0
Umin = -1.0

tt = "----------------------------------------"
tau = " E, a.u."
tev = " E, eV"
print(tt)
print("k", " ", tau, " ", tev)
print(tt)

for k in range(1, 6):
    Ea = Ek(k)
    Ev = 27.212 * Ea
    print("{:1d}   {:12.7f}   {:12.7f}".format(k, Ea, Ev))
