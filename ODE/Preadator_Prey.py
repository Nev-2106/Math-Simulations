import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#initial conditions
alpha = 0.1  # prey growth rate
beta = 0.02  # predation rate
delta = 0.01  # predator growth rate per prey eaten
gamma = 0.1  # predator death rate
t = np.linspace(0, 10, 1000)


def predator_prey(t, y):
    x, y = y
    dxdt = alpha*x - beta*x*y
    dydt = delta*x - gamma*y
    return [dxdt, dydt]

# Solve the ODE
sol = solve_ivp(predator_prey, [0, 10], [10, 10], t_eval=t, method='RK45')

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

# Plot time series
axes[0].plot(sol.t, sol.y[0], label='Prey', lw=2)
axes[0].plot(sol.t, sol.y[1], label='Predator', lw=2)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Population")
axes[0].set_title("Predator-Prey Model Simulation Time Series")
axes[0].legend()
axes[0].grid(True)

# Phase portrait
axes[1].plot(sol.y[0], sol.y[1], lw=2)
axes[1].set_xlabel("Prey Population")
axes[1].set_ylabel("Predator Population")
axes[1].set_title("Phase Portrait") 
axes[1].grid(True)

plt.tight_layout()
plt.show()