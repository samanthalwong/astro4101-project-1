import numpy as np
import math
from matplotlib import pyplot as plt

#QUESTION 1
def rk2(theta, zeta, v, n, h, derivsRK):
    """Runge-Kutta integrator (2nd order)
       Input arguments -
        theta = independent variable
        zeta = current value of dependent variable
        v = du/dzeta
        n = polytropic index
        h = step size (delta x)
        derivsRK = right hand side of the ODE; derivsRK is the
                  name of the function which returns dx/dt
                  Calling format derivsRK (x,t,param).
       Output arguments -
        zeta_out = new value of zeta after a step of size tau
    """
    zeta_temp = zeta
    zeta = zeta + h
    v = v + h*odes(n, [theta, zeta_temp, v])[1]
    theta = theta + h*odes(n, [theta, zeta_temp, v])[0]
    return [v, theta, zeta]

def odes(n,s):
    """
    Returns RHS of coupled ODE drho_bar/dr_bar
    inputs:
    n: polytropic index
    s: state vector [theta, zeta, v]
    output:
    derivatives [du/dzeta, dv/dzeta]
    """
    #unpack state vector
    theta = s[0]
    zeta = s[1]
    v = s[2]

    u = theta

    #solve derivatives
    du_dzeta = v
    dv_dzeta = -u**n-2*v/zeta

    return np.array([du_dzeta, dv_dzeta])

#SOLUTION FOR n = 1.5
poly_index = [1.5, 3, 3.25] #initial conditions from polytropes lecture
h = 1e-3 #timestep
nsteps = 10000

for n in poly_index:
    zeta = 0.000001
    u = 1
    v = 0
    index = 0
    theta_array = np.empty(nsteps)
    zeta_array = np.empty(nsteps)
    for i in range(nsteps):
        #update arrays with solutions from previous iteration
        theta_array[i] = u
        zeta_array[i] = zeta
        #find new state vector using rk2 solver
        state = rk2(u, zeta, v, n, h, odes)
        u = state[1]
        v = state[0]
        zeta = state[2]
        if u < 0.00001 or math.isnan(u):
            index = index - 1
            break
        index = index + 1
    plt.plot(zeta_array[0:index], theta_array[0:index], label="n = %0.1f" %n)

#ANALYTIC SOLUTIONS FOR n = 0, 1, 5
zeta = np.linspace(0.000001,10,10000)
theta_0 = 1 - (zeta**2)/6
theta_1 = np.sin(zeta)/zeta
theta_5 = 1/(np.sqrt(1+(zeta**2)/3))

theta_0 = theta_0[0: np.where(theta_0 < 0)[0][0]]

plt.plot(zeta[0:theta_0.size], theta_0, label="n = 0")
plt.plot(zeta[0:theta_1.size], theta_1, label="n = 1")
plt.plot(zeta[0:theta_5.size], theta_5, label="n = 5")

plt.xlabel("zeta")
plt.ylabel("theta")
plt.grid()
plt.legend()

plt.show()



