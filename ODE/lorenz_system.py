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

#solveing system of ODEs
solution = solve_ivp(lorenz, t_span, y0, t_eval=t_eval,method='RK45')
trajectories = solution.y.reshape(4, 3, -1)  # shape (4, 3, n_pts)

# pairwise distances for plotting
distances = np.zeros((4, 4, len(t_eval)))
for i in range(4):
    for j in range(i+1, 4):
        dx = trajectories[i, 0] - trajectories[j, 0]
        dy = trajectories[i, 1] - trajectories[j, 1]
        dz = trajectories[i, 2] - trajectories[j, 2]
        distances[i, j] = np.sqrt(dx**2 + dy**2 + dz**2)
        distances[j, i] = distances[i, j]  # symmetric


# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(4):
    ax.plot(trajectories[i, 0], trajectories[i, 1], trajectories[i, 2],
            color=colors[i], label=labels[i], lw=0.5)
ax.set_title("Coupled Lorenz-Gravitational System")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()

#plotting pairwise distances
fig2, ax2 = plt.subplots(figsize=(10, 6))
for i in range(4):
    for j in range(i+1, 4):
        ax2.plot(t_eval, distances[i, j], label=f"{labels[i]}-{labels[j]}")
ax2.set_title("Pairwise Distances Over Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Distance")
ax2.legend()
plt.show()