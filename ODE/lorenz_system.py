import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D           
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

# Initial conditions
SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0
mass = np.array([1.0,1.0,1.0])

gravity = 1.5    # gravitational coupling  (0 = pure Lorenz, >0 = coupled)
softening = 0.5    # softening length        (prevents 1/r singularity)

def lorenz(t, y):
    x, y, z = y
    dxdt = SIGMA * (y - x) + gravity * (y - x) / ((x - y)**2 + softening**2)**1.5
    dydt = x * (RHO - z) - y + gravity * (x - y) / ((x - y)**2 + softening**2)**1.5
    dzdt = x * y - BETA * z
    return [dxdt, dydt, dzdt]

t_span = (0, 30)
t_eval = np.linspace(*t_span, 15000)
y0 = [8.0, 8.0, 20.0]
solution = solve_ivp(lorenz, t_span, y0, t_eval=t_eval,method='RK45')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(solution.y[0], solution.y[1], solution.y[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz System with Gravitational Coupling')
plt.show()
