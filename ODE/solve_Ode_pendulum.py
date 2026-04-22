import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

g = 9.8
L = 2
mu = 0.1

def get_theta(theta,theta_dot):
    return -mu*theta_dot -(g/L)*np.sin(theta)

def simulate(t_max):
    theta_0 = np.pi/3
    theta_dot_0 = 0
    theta = theta_0
    theta_dot = theta_dot_0
    d_t = 0.01
    
    t_vals = []
    x_vals = []
    y_vals = []
    
    for time in np.arange(0,t_max,d_t):
        theta += theta_dot * d_t
        theta_dot += get_theta(theta,theta_dot) * d_t
        
        x = L * np.sin(theta)
        y = -L * np.cos(theta)
        
        t_vals.append(time)
        x_vals.append(x)
        y_vals.append(y)
    return t_vals, x_vals, y_vals

t,x,y = simulate(100)

plt.plot(x,y)
plt.xlabel('distance (rad)')
plt.ylabel('Period (rad/s)')
plt.title('Pendulum Simulation')
plt.axis('equal')
plt.grid()
plt.show()

fig, ax = plt.subplots()
ax.set_xlim(-L, L)
ax.set_ylim(-L, L)

line, = ax.plot([], [], 'o-', lw=2)

def update(frame):
    line.set_data([0, x[frame]], [0, y[frame]])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(x), interval=20)
plt.show()