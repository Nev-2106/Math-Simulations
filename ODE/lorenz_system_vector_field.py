import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Initial conditions
SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0


gravity = 1.5    # gravitational coupling  (0 = pure Lorenz, >0 = coupled)
softening = 0.5    # softening length        (prevents 1/r singularity)
mass = np.array([1.0,1.0,1.0,1.0])
t_span = (0, 30)
t_eval = np.linspace(*t_span, 400)
colors = ["black", "red", "blue", "green"]
labels = ["m1", "m2", "m3", "m4"]

#define 4 masses their initial positions and velocities
angles = np.linspace(0,2*np.pi,4,endpoint=False)
R = 8.0
x_list = [(R*np.cos(a),R*np.sin(a),20+i*0.5) for i, a in enumerate(angles)]
y0 = np.array([c for xyz in x_list for c in xyz], dtype=float)

# Lorenz system with gravity coupling
def lorenz(t, y):
    n = 4
    pos = y.reshape(n, 3)          # shape (4, 3)
    dydt = np.empty_like(pos)

    for i in range(n):
        xi, yi, zi = pos[i]

        # Lorenz terms
        dxi = SIGMA * (yi - xi)
        dyi = xi * (RHO - zi) - yi
        dzi = xi * yi - BETA * zi

        # Gravity from all other bodies
        gx = gy = gz = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            r2 = dx*dx + dy*dy + dz*dz + softening**2
            inv_r3 = gravity * mass[j] / (r2 * np.sqrt(r2))
            gx += inv_r3 * dx
            gy += inv_r3 * dy
            gz += inv_r3 * dz

        dydt[i] = [dxi + gx, dyi + gy, dzi + gz]

    return dydt.flatten()

#vector field for the coupled system
X, Y, Z = np.meshgrid(np.linspace(-20, 20, 10), np.linspace(-30, 30, 10), np.linspace(0, 50, 10))
U = SIGMA * (Y - X)
V = X * (RHO - Z) - Y
W = X * Y - BETA * Z
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, length=0.5, normalize=True, color='gray', alpha=0.5)
ax.set_title("Vector Field of Coupled Lorenz System")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()  