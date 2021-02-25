import numpy as np
import math
from matplotlib import pyplot as plt

#QUESTION 1
def rk2(theta, xi, v, n, h, derivsRK):
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
    xi_temp = xi
    xi = xi + h #increment zeta by the timestep value
    v = v + h*odes(n, [theta, xi_temp, v])[1] #use midpoint method to compute new v and u
    theta = theta + h*odes(n, [theta, xi_temp, v])[0]
    return [v, theta, xi]

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
    xi = s[1]
    v = s[2]

    u = theta

    #solve derivatives
    du_dxi = v
    dv_dxi = -u**n-2*v/xi

    return np.array([du_dxi, dv_dxi])

#SOLUTION FOR n = 1.5
poly_index = [1.5, 3, 3.25] #initial conditions from polytropes lecture
h = 1e-3 #timestep
nsteps = 10000 #number of steps to iterate through

for n in poly_index:
    #initial conditions
    xi = 0.000001 #slightly more than zero to avoid divergence at the origin
    u = 1
    v = 0
    index = 0 #index to keep track of when theta = 0
    theta_array = np.empty(nsteps)
    xi_array = np.empty(nsteps)
    v_array = np.empty(nsteps)
    for i in range(nsteps):
        #update solution arrays with solutions from previous iteration
        theta_array[i] = u
        xi_array[i] = xi
        v_array = v
        #find new state vector using rk2 solver
        state = rk2(u, xi, v, n, h, odes)
        u = state[1]
        v = state[0]
        xi = state[2]
        if u < 0.00001 or math.isnan(u): #if theta is very close or less than zero, exit loop
            index = index - 1
            break
        index = index + 1
    plt.plot(xi_array[0:index], theta_array[0:index], label="n = %0.1f" %n) #plot results

#ANALYTIC SOLUTIONS FOR n = 0, 1, 5
xi_analytic = np.linspace(0.000001,10,10000)
theta_0 = 1 - (xi_analytic**2)/6
theta_1 = np.sin(xi_analytic)/xi_analytic
theta_5 = 1/(np.sqrt(1+(xi_analytic**2)/3))

theta_0 = theta_0[0: np.where(theta_0 < 0)[0][0]] #stop computing when theta < 0

plt.plot(xi_analytic[0:theta_0.size], theta_0, label="n = 0")
plt.plot(xi_analytic[0:theta_1.size], theta_1, label="n = 1")
plt.plot(xi_analytic[0:theta_5.size], theta_5, label="n = 5")

plt.xlabel("xi")
plt.ylabel("theta")
plt.grid()
plt.legend()

plt.show()

#QUESTION 2

#a)
M = 1.99e33
r = 6.96e10
rho_c = (-M*xi)/(4*np.pi*r**3*v_array)
print("Central Density of the Sun: %0.1f g/cm^3" %rho_c)

#b)
alpha = r/xi
print("Length Scale, Alpha: %0.1f cm" %alpha)

#c)
n = 3.25
G = 6.67e-8
K = (4*np.pi*G*alpha**2)/((n+1)*rho_c**((1-n)/n))
print("Polytropic Constant, K: %0.1f " %K)

#d)
P_c = (G*M**2)/(r**4*4*np.pi*(n+1)*v**2)
print("Central Pressure: %0.1f Ba" %P_c)

#e)
mH = 1.00784
m = 0.6*mH
k = 1.38064852e-23
T_c = (P_c*M*m)/(rho_c*k)
print("Central Temperature: %f K" %T_c) #this number is WAY too big

#QUESTION 3
r_model = np.loadtxt('solartable.csv', dtype = float, delimiter=',', skiprows=1, usecols=(0))
rho_model = np.loadtxt('solartable.csv', dtype = float, delimiter=',', skiprows=1, usecols=(4))

r_rsun = r_model/r
plt.plot(rho_model,r_rsun, 'k', label="Standard Solar Model")
plt.show()