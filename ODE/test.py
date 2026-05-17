"""
Three-Body Gravitational + Lorenz Chaos Simulator
==================================================
Each body evolves under:
  - Lorenz chaotic dynamics  (sigma, rho, beta)
  - Newtonian mutual gravity (G, softened by eps)

Equations of motion for body i:
  dx_i/dt = sigma*(y_i - x_i)         + sum_j [ G*m_j*(x_j-x_i) / (r_ij^2 + eps^2)^(3/2) ]
  dy_i/dt = x_i*(rho - z_i) - y_i     + ...
  dz_i/dt = x_i*y_i - beta*z_i        + ...

Dependencies: numpy, scipy, matplotlib
  pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

# ── Parameters ────────────────────────────────────────────────────────────────
SIGMA = 10.0
RHO   = 28.0
BETA  = 8.0 / 3.0

G     = 1.5    # gravitational coupling  (0 = pure Lorenz, >0 = coupled)
EPS   = 0.5    # softening length        (prevents 1/r singularity)
MASSES = np.array([1.0, 1.0, 1.0])   # m1, m2, m3

T_END = 30.0
N_PTS = 15_000
COLORS = ['#378ADD', '#D85A30', '#1D9E75']
LABELS = ['m₁', 'm₂', 'm₃']

# ── Initial conditions ─────────────────────────────────────────────────────────
# Place bodies at vertices of an equilateral triangle in XY, offset in Z
R0 = 8.0
angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
x0_list = [(R0 * np.cos(a), R0 * np.sin(a), 20.0 + i * 0.5)
           for i, a in enumerate(angles)]

# State vector: [x1,y1,z1, x2,y2,z2, x3,y3,z3]  (9 values)
y0 = np.array([c for xyz in x0_list for c in xyz], dtype=float)


# ── ODE ───────────────────────────────────────────────────────────────────────
def lorenz_gravity(t, y, masses, G, eps, sigma, rho, beta):
    """Vectorised RHS for 3 coupled bodies."""
    n = 3
    pos = y.reshape(n, 3)          # shape (3, 3)
    dydt = np.empty_like(pos)

    for i in range(n):
        xi, yi, zi = pos[i]

        # Lorenz terms
        dxi = sigma * (yi - xi)
        dyi = xi * (rho - zi) - yi
        dzi = xi * yi - beta * zi

        # Gravity from all other bodies
        gx = gy = gz = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi
            dz = pos[j, 2] - zi
            r2 = dx*dx + dy*dy + dz*dz + eps**2
            r3 = r2 ** 1.5
            fac = G * masses[j] / r3
            gx += fac * dx
            gy += fac * dy
            gz += fac * dz

        dydt[i] = [dxi + gx, dyi + gy, dzi + gz]

    return dydt.ravel()


# ── Solve ─────────────────────────────────────────────────────────────────────
print("Integrating … (RK45, dense output)")
t_eval = np.linspace(0, T_END, N_PTS)

sol = solve_ivp(
    lorenz_gravity,
    [0, T_END],
    y0,
    t_eval=t_eval,
    method='RK45',
    args=(MASSES, G, EPS, SIGMA, RHO, BETA),
    rtol=1e-9,
    atol=1e-9,
)

if not sol.success:
    raise RuntimeError(f"Integration failed: {sol.message}")

# Unpack  shape → (3, 3, N_PTS)
traj = sol.y.reshape(3, 3, -1)   # [body, coord, time]
t    = sol.t


# ── Utility: coloured trail (alpha gradient) ───────────────────────────────────
def gradient_line(ax, xs, ys, zs, color, lw=0.6, n_seg=400):
    """Draw a 3-D trail that fades from transparent to opaque."""
    step = max(1, len(xs) // n_seg)
    xs, ys, zs = xs[::step], ys[::step], zs[::step]
    alphas = np.linspace(0.05, 1.0, len(xs) - 1)
    for k in range(len(xs) - 1):
        ax.plot(xs[k:k+2], ys[k:k+2], zs[k:k+2],
                color=color, alpha=float(alphas[k]), lw=lw)


# ── Pairwise distances ─────────────────────────────────────────────────────────
def pair_dist(i, j):
    d = traj[i] - traj[j]         # (3, N)
    return np.sqrt((d**2).sum(axis=0))

d12 = pair_dist(0, 1)
d13 = pair_dist(0, 2)
d23 = pair_dist(1, 2)


# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Three-Body: Lorenz Chaos + Newtonian Gravity", fontsize=14, y=0.98)

gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# --- 3-D attractor ---
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
for i in range(3):
    gradient_line(ax3d,
                  traj[i, 0], traj[i, 1], traj[i, 2],
                  COLORS[i], lw=0.5)
    ax3d.scatter(*traj[i, :, -1], color=COLORS[i], s=40, zorder=5)
    ax3d.text(*traj[i, :, -1], f" {LABELS[i]}",
              color=COLORS[i], fontsize=9)

ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")
ax3d.set_title("Phase space (XYZ)")

# --- XY projection ---
axXY = fig.add_subplot(gs[0, 1])
for i in range(3):
    axXY.plot(traj[i, 0], traj[i, 1], color=COLORS[i], lw=0.5, alpha=0.8, label=LABELS[i])
    axXY.scatter(traj[i, 0, -1], traj[i, 1, -1], color=COLORS[i], s=30, zorder=5)
axXY.set_xlabel("X"); axXY.set_ylabel("Y"); axXY.set_title("XY projection")
axXY.legend(fontsize=8, loc='upper right')

# --- XZ projection ---
axXZ = fig.add_subplot(gs[0, 2])
for i in range(3):
    axXZ.plot(traj[i, 0], traj[i, 2], color=COLORS[i], lw=0.5, alpha=0.8)
    axXZ.scatter(traj[i, 0, -1], traj[i, 2, -1], color=COLORS[i], s=30, zorder=5)
axXZ.set_xlabel("X"); axXZ.set_ylabel("Z"); axXZ.set_title("XZ projection")

# --- Pairwise distances over time ---
axD = fig.add_subplot(gs[1, 1])
axD.plot(t, d12, color=COLORS[0], lw=0.8, label='m₁↔m₂')
axD.plot(t, d13, color=COLORS[1], lw=0.8, label='m₁↔m₃')
axD.plot(t, d23, color=COLORS[2], lw=0.8, label='m₂↔m₃')
axD.set_xlabel("t"); axD.set_ylabel("Distance")
axD.set_title("Pairwise separation")
axD.legend(fontsize=8)

# --- Gravitational PE over time ---
axE = fig.add_subplot(gs[1, 2])
pe = np.zeros(len(t))
pairs = [(0,1,d12), (0,2,d13), (1,2,d23)]
for i, j, d in pairs:
    pe -= G * MASSES[i] * MASSES[j] / (d + EPS)
axE.plot(t, pe, color='#888780', lw=0.8)
axE.set_xlabel("t"); axE.set_ylabel("Gravitational PE")
axE.set_title("Potential energy")
axE.fill_between(t, pe, alpha=0.15, color='#888780')

plt.savefig("three_body_lorenz.png", dpi=150, bbox_inches='tight')
print("Saved → three_body_lorenz.png")
plt.show()


# ── Optional: sensitivity to initial conditions ────────────────────────────────
def run_perturbed(delta=1e-5):
    y0p = y0.copy()
    y0p[0] += delta    # nudge x of body 1
    sol2 = solve_ivp(
        lorenz_gravity, [0, T_END], y0p,
        t_eval=t_eval, method='RK45',
        args=(MASSES, G, EPS, SIGMA, RHO, BETA),
        rtol=1e-9, atol=1e-9,
    )
    return sol2.y.reshape(3, 3, -1)

print("Running perturbed trajectory for Lyapunov estimate …")
traj_p = run_perturbed(delta=1e-5)

# Separation between original and perturbed body 1
sep = np.sqrt(((traj[0] - traj_p[0])**2).sum(axis=0))
sep = np.clip(sep, 1e-15, None)

fig2, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(t, sep, color=COLORS[0], lw=0.9)
ax.set_xlabel("t")
ax.set_ylabel("||δ(t)|| (log scale)")
ax.set_title("Butterfly effect — divergence of nearby trajectories (body m₁)")
ax.axhline(1.0, ls='--', color='gray', lw=0.7, label='macroscopic scale')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("lyapunov_divergence.png", dpi=150, bbox_inches='tight')
print("Saved → lyapunov_divergence.png")
plt.show()