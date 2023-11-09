import numpy as np
from dsp import quadro_method, kernel_approx, galkin_petrov
import matplotlib.pyplot as plt
def task_1():
    h = 0.1
    a = 0
    b = 1 + 0.001
    lam = 0.5
    x = np.arange(a, b, h)

    K = lambda x1, s: x1 * s * lam
    f = lambda x1: 5 / 6 * x1
    y_exact = lambda x1: x1

    y = []  # точное решение
    for i in range(len(x)):
        y.append([])  # создаем пустую строку
        y[i].append(y_exact(x[i]))
    y = np.array(y).reshape(len(x), 1)  # точное решение

    y_approx = quadro_method(K, f, a, b, h)
    plt.plot(x, y, '-g', linewidth=2, label='y_exact')  # график точного решения
    plt.plot(x, y_approx, 'or', label='y_approx')  # график найденного решения
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(bbox_to_anchor=(1, 1), loc='best')
    plt.ylim([0, max(y) + 2])
    plt.show()

def task_2():
    a = 0
    b = 1.001
    h = 0.05
    Lambda = -1
    x = np.arange(a, b, h)
    f = lambda t: np.exp(t) - t
    y_exact = lambda t: 1  # точное решение

    y = []  # точное решение
    for i in range(len(x)):
        y.append([])
        y[i].append(y_exact(x[i]))
    y = np.array(y).reshape(len(x), 1)

    y_approx = kernel_approx(a, b, f, x, Lambda)
    plt.plot(x, y, '-g', linewidth=2, label='y_exact')
    plt.plot(x, y_approx, 'or', label='y_approx')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(0, 1.1)
    plt.legend('1', fontsize=12)
    plt.legend(bbox_to_anchor=(1, 1), loc='best')
    plt.show()


def task_3():
    phi = [
        lambda x: x,
        lambda x: x ** 2
    ]

    psi = [
        lambda x: 1,
        lambda x: x
    ]

    K = lambda x, s: (x ** 2 + x * s)
    f = lambda x: 1
    lam = 1

    a = np.zeros([2, 2])
    b = np.zeros(2)

    a, b = galkin_petrov(a=a, b=b, psi=psi, phi=phi, K=K, lam=lam, f=f)

    x = np.linspace(-1, 1, 10)
    c = np.linalg.solve(a, b)

    plt.plot(x, 1 + 6 * x ** 2, '-g', linewidth=2, label='y_exact')
    plt.plot(x, 1 + c[0] * phi[0](x) + c[1] * phi[1](x), 'or', label='y_approx')
    plt.legend()
    plt.show()


task_1()