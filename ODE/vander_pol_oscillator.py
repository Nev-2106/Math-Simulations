import numpy as np
import matplotlib.pyplot as plt

#initial conditions
mu = 1.0  # nonlinearity parameter

def Vander_Pol(t, y):
    x, dxdt = y
    d2xdt2 = mu * (1 - x**2) * dxdt - x
    return [dxdt, d2xdt2]

def solve_ivp(func, t_span, y0, t_eval, method):
    from scipy.integrate import solve_ivp
    return solve_ivp(func, t_span, y0, t_eval=t_eval, method=method)

t = np.linspace(0, 20, 1000)
sol = solve_ivp(Vander_Pol, [0, 20], [2, 0], t_eval=t, method='RK45')
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='x(t)', lw=2)
plt.plot(sol.t, sol.y[1], label='dx/dt', lw=2)
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("Van der Pol Oscillator Simulation Time Series")
plt.legend()
plt.grid(True)
plt.show()

