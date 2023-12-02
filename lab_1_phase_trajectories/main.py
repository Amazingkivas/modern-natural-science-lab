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


def plotonPlane(rhs, limits, time_lims, points, colors):
    plt.close()
    xlims, ylims = limits
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    xs, ys, U, V = eq_quiver(rhs, limits)
    plt.quiver(xs, ys, U, V, alpha=0.8)

    eps = 0.001

    for time, point, color in zip(time_lims, points, colors):
        eigen = np.linalg.eig(matrix(point[0]))
        #print(eigen)
        
        reverse_time = [time[1], time[0]]
        offset = eps * eigen[1][1][0] if (color == 'r-') else 0
        
        sol = solve_ivp(rhs, time, 
                        [point[0] + offset, point[1] + offset], 
                        method='RK45', rtol=1e-12, atol=1e-10)
        x, y = sol.y
        plt.plot(x, y, color)
        sol = solve_ivp(rhs, reverse_time, 
                        [point[0] + offset, -point[1] - offset], 
                        method='RK45', rtol=1e-12, atol=1e-10)
        x, y = sol.y
        plt.plot(x, y, color)
        sol = solve_ivp(rhs, time, 
                        [point[0] - offset, -point[1] - offset], 
                        method='RK45', rtol=1e-12, atol=1e-10)
        x, y = sol.y
        plt.plot(x, y, color)
        sol = solve_ivp(rhs, reverse_time, 
                        [point[0] - offset, point[1] + offset], 
                        method='RK45', rtol=1e-12, atol=1e-10)
        x, y = sol.y
        plt.plot(x, y, color)
    

pnts = [[0., 0.], [2., 0.], [-2., 0.], [-2.5, 0.], [2.5, 0.], [-0.5, 0.], [0.5, 0.], [2., 1.5], [0., 1.]]

tms = [[-4., 4.]] + [[-2., 2.]] * 2 + [[-1., 1.]] * 6

clrs = ['r-'] * 3 + ['g-'] * 6
    
## sep:
#plotonPlane(func(), [( -2.0025, -1.9975), (-0.0027, 0.0027)], tms, pnts, clrs) 
#plotonPlane(func(), [( 1.9975, 2.0025), (-0.0027, 0.0027)], tms, pnts, clrs)
#plotonPlane(func(), [( -0.0027, 0.0027), (-0.0027, 0.0027)], tms, pnts, clrs)
##

#plotonPlane(func(), [( -2.0025, -1.9975), (1.4975, 1.5025)], tms, pnts, clrs)
#plotonPlane(func(), [( 1.9975, 2.0025), (1.4975, 1.5025)], tms, pnts, clrs)

#plotonPlane(func(), [( -0.004, 0.004), (1-0.0027, 1+0.0027)], tms, pnts, clrs)

#plotonPlane(func(), [( -2.0025-0.5, -1.9975-0.5), (-0.0027, 0.0027)], tms, pnts, clrs)
#plotonPlane(func(), [( 1.9975+0.5, 2.0025+0.5), (-0.0027, 0.0027)], tms, pnts, clrs)

plotonPlane(func(), [( -2.7, 2.7), (-2.7, 2.7)], tms, pnts, clrs)


####
def plot_by_time(rhs, limits, time, point, offset):
    plt.close()
    xlims, ylims = limits
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    
    #print(eigen)
    sol = solve_ivp(rhs, time, [point[0] + offset, point[1] + offset], 
                    method='RK45', rtol=1e-12, atol=1e-10)
    
    x = sol.y[0]
    t = sol.t
    plt.plot(t, x)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.grid(True)

####
eps = 0.001
eigen = np.linalg.eig(matrix(0.))
plot_by_time(func(), [(-3., 6.), ( -2., 0.5)], [-3. , 6.], [0., 0.], -eps * eigen[1][1][0])

####
eigen = np.linalg.eig(matrix(0.))
plot_by_time(func(), [(-3., 6.), ( -0.5, 2)], [-3. , 6.], [0., 0.], eps * eigen[1][1][0])

####
eigen = np.linalg.eig(matrix(-2.))
plot_by_time(func(), [(-0.5, 2.), ( -5., 0)], [2. , -0.5], [-2., 0.], -eps * eigen[1][1][0])

####
eigen = np.linalg.eig(matrix(2.))
plot_by_time(func(), [(-0.5, 2.), ( 0., 5)], [-0.5 , 2.], [2., 0.], eps * eigen[1][1][0])

####
plot_by_time(func(), [(-3., 6.), ( 0., 2)], [-3. , 6.], [0.5, 0.], 0)

####
plot_by_time(func(), [(-3., 6.), ( -2., 0)], [-3. , 6.], [-0.5, 0.], 0)

####
plot_by_time(func(), [(-3, 6.), ( -3., 3)], [-3. , 6.], [0., 1.], 0)

####
plot_by_time(func(), [(-3, 6.), ( -3., 3)], [-3. , 6.], [0., -1.], 0)
