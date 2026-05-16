import numpy as np
import matplotlib.pyplot as plt

#initial conditions
mu = 1.0  # nonlinearity parameter
t = np.linspace(0, 20, 1000)


def Vander_Pol(t, y):
    x, y = y
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

#grid
x_vals = np.linspace(-3, 3, 30)
y_vals = np.linspace(-3, 3, 30)
X, Y = np.meshgrid(x_vals, y_vals)
dX, dY = Vander_Pol(0, [X, Y])

#plot
plt.figure(figsize=(8,6))
plt.quiver(X, Y, dX, dY)
plt.xlabel("x")
plt.ylabel("dx/dt")
plt.title("Phase Space Vector Field (Van der Pol Oscillator)")
plt.grid()
plt.show()


