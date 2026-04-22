import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
omega = 2.0          # angular frequency (rad/s)
x0 = 1.0             # initial displacement (m)
v0 = 0.0             # initial velocity (m/s)
t_span = (0, 10)     # time range (s)
t_eval = np.linspace(*t_span, 1000)

def harmonic_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

# Solve
sol = solve_ivp(harmonic_oscillator, t_span, [x0, v0], t_eval=t_eval, method='RK45')

x_analytical = x0 * np.cos(omega * t_eval) + (v0 / omega) * np.sin(omega * t_eval)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

# Time series
axes[0].plot(sol.t, sol.y[0], label='Numerical', lw=2)
axes[0].plot(t_eval, x_analytical, '--', label='Analytical', lw=2)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Displacement x (m)")
axes[0].set_title(f"Simple Harmonic Oscillator  (ω = {omega} rad/s)")
axes[0].legend()
axes[0].grid(True)

# Phase portrait
axes[1].plot(sol.y[0], sol.y[1], lw=2)
axes[1].set_xlabel("Displacement x (m)")
axes[1].set_ylabel("Velocity ẋ (m/s)")
axes[1].set_title("Phase Portrait")
axes[1].grid(True)

plt.tight_layout()
plt.show()