import numpy as np 
import matplotlib.pyplot as plt

#initial conditions
x0 = 1.0
v0 = 0.0
omega = 2.0
t = np.linspace(0, 10, 1000)

def harmonic_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]


#grid
x_vals = np.linspace(-2, 2, 20)
v_vals = np.linspace(-4, 4, 20)
T, V = np.meshgrid(x_vals, v_vals)
dT, dV = harmonic_oscillator(0, [T, V])

#plotiing 
plt.figure(figsize=(8,6))
plt.quiver(T, V, dT, dV)
plt.xlabel("displacement")
plt.ylabel("velocity")
plt.title("Phase Space Vector Field (Simple Harmonic Oscillator)")
plt.grid()
plt.show()
