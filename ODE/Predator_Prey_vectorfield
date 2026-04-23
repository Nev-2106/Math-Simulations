import numpy as np 
import matplotlib.pyplot as plt

#initial conditions
alpha = 0.1  # prey growth rate
beta = 0.02  # predation rate
delta = 0.01  # predator growth rate per prey eaten
gamma = 0.1  # predator death rate
t = np.linspace(0, 10, 1000)



#modeling
def predator_prey(t, y):
    x, y = y
    dxdt = alpha*x - beta*x*y
    dydt = delta*x - gamma*y
    return [dxdt, dydt]



#grid
x_vals = np.linspace(0, 10, 30)
y_vals = np.linspace(0, 100, 30)
X, Y = np.meshgrid(x_vals, y_vals)
dX, dY = predator_prey(0, [X, Y])

#plot 
plt.figure(figsize=(8,6))
plt.quiver(X, Y, dX, dY)
plt.xlabel("prey population")
plt.ylabel("predator population")
plt.title("Phase Space Vector Field (Predator-Prey Model)")
plt.grid()
plt.show()

    