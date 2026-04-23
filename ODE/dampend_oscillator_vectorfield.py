import numpy as np 
import matplotlib.pyplot as plt

#intial conditions 
omega = 2.0
x0 = 1.0
v0 = 0.0
mass = 1.0
ar = 1.0 # damping coefficient
t=np.linspace(0, 10, 1000)
g=9.81


def damped_oscillator(t, y):
    force = -mass * g * np.sin(y[0]) - ar * y[1]
    x, v = y
    dxdt = v
    dvdt = (force*np.cos(omega*t)-v*ar-omega**2*x)
    return [dxdt, dvdt]

#grid
# grid
theta_vals = np.linspace(-4*np.pi, 4*np.pi, 30)
theta_dot_vals = np.linspace(-10, 10, 30)
T, V = np.meshgrid(theta_vals, theta_dot_vals)
dT, dV = damped_oscillator(0, [T, V])

#plotting
plt.figure(figsize=(8,6))
plt.quiver(T, V, dT, dV)
plt.xlabel("displacement")
plt.ylabel("velocity")
plt.title("Phase Space Vector Field (Damped Harmonic Oscillator)")
plt.grid()
plt.show()