import numpy as np 
import matplotlib.pyplot as plt

#intial conditions 
r = 1.0  # growth rate
K = 10.0  # carrying capacity
t=np.linspace(0, 10, 1000)

def logistic_growth(t, y):
    x = y[0]
    dxdt = r * x * (1 - x / K)
    return [dxdt]

# grid
x_vals = np.linspace(0, K, 30)
dxdt_vals = np.linspace(-10, 10, 30)
T, V = np.meshgrid(x_vals, dxdt_vals)
dV = logistic_growth(0, [T])

#plotting
plt.figure(figsize=(8,6))
plt.quiver(T, V, dV)
plt.xlabel("time")
plt.ylabel("population")
plt.title("Phase Space Vector Field (Logistic Growth)")
plt.grid()
plt.show()