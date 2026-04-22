import numpy as np
import matplotlib.pyplot as plt

g = 9.8
L = 2
mu = 0.1

def field(theta, theta_dot):
    dtheta = theta_dot
    dtheta_dot = -mu * theta_dot - (g/L) * np.sin(theta)
    return dtheta, dtheta_dot

# grid
theta_vals = np.linspace(-4*np.pi, 4*np.pi, 30)
theta_dot_vals = np.linspace(-10, 10, 30)

T, V = np.meshgrid(theta_vals, theta_dot_vals)

dT, dV = field(T, V)

plt.figure(figsize=(8,6))
plt.quiver(T, V, dT, dV)

plt.xlabel("distance")
plt.ylabel("period ")
plt.title("Phase Space Vector Field (Damped Pendulum)")
plt.grid()
plt.show()