import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def eq_quiver(rhs, limits, N=16):
    xlims, ylims = limits
    xs = np.linspace(xlims[0], xlims[1], N)
    ys = np.linspace(ylims[0], ylims[1], N)
    U = np.zeros((N, N))
    V = np.zeros((N, N))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            vfield = rhs(0.0, [x, y])
            u, v = vfield
            U[i][j] = u
            V[i][j] = v
    return xs, ys, U, V


def func():
    def rhs(t, X):
        x, y = X
        return (y, x ** 5 - 5.0 * x ** 3 + 4.0 * x)

    return rhs


def matrix(x):
    px = 0.0  # 
    py = 1.0
    qx = 5.0 * x ** 4 - 15.0 * x ** 2 + 4.0
    qy = 0.0
    return [[px, py], [qx, qy]]


def plotonPlane(rhs, limits):
    plt.close()
    xlims, ylims = limits
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    xs, ys, U, V = eq_quiver(rhs, limits)
    plt.quiver(xs, ys, U, V, alpha=0.8)

    eps = 0.001

    eigen = np.linalg.eig(matrix(0.))
<<<<<<< HEAD
    # print(eigen)
    sol = solve_ivp(rhs, [1, 9.7], [eps * eigen[1][1][1], eps * eigen[1][1][1]], method='RK45', rtol=1e-12, atol=1e-10)
=======
    #print(eigen)
    sol = solve_ivp(rhs, [1., 100.], [eps * eigen[1][0][0], eps * eigen[1][0][1]],
                    method='RK45',
                    rtol=1e-12,
                    atol=1e-10)
>>>>>>> 0e65d656c0b287a7c17ece29a67e2acb40f52025
    x, y = sol.y
    plt.plot(x, y, 'r-')
    sol = solve_ivp(rhs, [1, 9.7], [-eps * eigen[1][1][1], -eps * eigen[1][1][1]], method='RK45', rtol=1e-12,
                    atol=1e-10)
    x, y = sol.y
    plt.plot(x, y, 'r-')
<<<<<<< HEAD

    eigen = np.linalg.eig(matrix(2.))
    # print(eigen)
    sol = solve_ivp(rhs, [1, 10.], [2. + eps * eigen[1][1][0], eps * eigen[1][1][0]], method='RK45', rtol=1e-12,
                    atol=1e-10)
    x, y = sol.y
    plt.plot(x, y, 'r-')
    sol = solve_ivp(rhs, [10., 1], [2. + eps * eigen[1][1][1], -eps * eigen[1][1][1]], method='RK45', rtol=1e-12,
                    atol=1e-10)
    x, y = sol.y
    plt.plot(x, y, 'r-')
    sol = solve_ivp(rhs, [1, 5.], [2. - eps * eigen[1][1][1], -eps * eigen[1][1][1]], method='RK45', rtol=1e-12,
                    atol=1e-10)
    x, y = sol.y
    plt.plot(x, y, 'r-')
    sol = solve_ivp(rhs, [5., 1], [2. - eps * eigen[1][1][1], eps * eigen[1][1][1]], method='RK45', rtol=1e-12,
                    atol=1e-10)
    x, y = sol.y
    plt.plot(x, y, 'r-')

    eigen = np.linalg.eig(matrix(2.))
    # print(eigen)
    sol = solve_ivp(rhs, [1, 5.], [-2. + eps * eigen[1][1][0], eps * eigen[1][1][0]], method='RK45', rtol=1e-12,
                    atol=1e-10)
=======
    
    sol = solve_ivp(rhs, [1., 100.], [2., eps], method='RK45', rtol=1e-12, atol=1e-10)
>>>>>>> 0e65d656c0b287a7c17ece29a67e2acb40f52025
    x, y = sol.y
    plt.plot(x, y, 'r-')
    sol = solve_ivp(rhs, [5., 1], [-2. + eps * eigen[1][1][1], -eps * eigen[1][1][1]], method='RK45', rtol=1e-12,
                    atol=1e-10)
    x, y = sol.y
    plt.plot(x, y, 'r-')
    sol = solve_ivp(rhs, [1, 10.], [-2. - eps * eigen[1][1][1], -eps * eigen[1][1][1]], method='RK45', rtol=1e-12,
                    atol=1e-10)
    x, y = sol.y
    plt.plot(x, y, 'r-')
    sol = solve_ivp(rhs, [10., 1], [-2. - eps * eigen[1][1][1], eps * eigen[1][1][1]], method='RK45', rtol=1e-12,
                    atol=1e-10)
    x, y = sol.y
    plt.plot(x, y, 'r-')


plotonPlane(func(), [(-2.7, 2.7), (-2.7, 2.7)])