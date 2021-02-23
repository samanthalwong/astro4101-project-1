import numpy as np

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
        xout = new value of x after a step of size tau
    """
    half_h = 0.5 * h
    F1 = derivsRK(n, [theta, zeta, v])
    theta_half = theta + half_h
    zeta_temp = zeta + half_h * F1
    F2 = derivsRK(n, [theta_half, zeta_temp, v])
    zeta_out = zeta + h * F2
    return zeta_out

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

#SOLUTION FOR n = 0
zeta = 0; u = 1; v = 0 #initial conditions from polytropes lecture


