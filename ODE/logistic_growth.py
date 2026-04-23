import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#intial conditions
r = 0.1  # growth rate
K = 100  # carrying capacity
t=np.linspace(0, K, 1000)

def logistic_growth(t, y):
    x = y[0]
    dxdt = r * x * (1 - x / K)
    return [dxdt]

# Solve the ODE
sol = solve_ivp(logistic_growth, [0, K], [1], t_eval=t, method='RK45')

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

# Plot time series
axes[0].plot(sol.t, sol.y[0], label='Population', lw=2)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Population")
axes[0].set_title("Logistic Growth Simulation Time Series")
axes[0].legend()
axes[0].grid(True)

# Phase portrait
axes[1].plot(sol.y[0], lw=2)
axes[1].set_xlabel("Population")
axes[1].set_ylabel("Population growth rate")
axes[1].set_title("Phase Portrait")
axes[1].grid(True)

plt.tight_layout()
plt.show()