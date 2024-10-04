import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import picos
from matplotlib.animation import FuncAnimation

# Configuración del gráfico
vertices = {1, 2, 3, 4, 5, 6}
edges = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]

# Posiciones iniciales
p1 = np.array([0, 0, 1])
p2 = np.array([0, 1, 0])
p3 = np.array([0, -1, 0])
p4 = np.array([1, 0, 0])
p5 = np.array([-1, 0, 0])
p6 = np.array([0, 0, -1])

p = np.array([p1, p2, p3, p4, p5, p6])
pbar = np.column_stack((p, np.ones(len(vertices))))

# Matriz de Incidencia
H = np.zeros([len(vertices), len(edges)])
for i, edge in enumerate(edges):
    H[edge[0] - 1, i] = 1
    H[edge[1] - 1, i] = -1

E = pbar.T @ H @ np.diag(H[0, :])
for i in range(1, len(vertices)):
    E = np.append(E, pbar.T @ H @ np.diag(H.T[:, i]), axis=0)

# SVD
U, S, Vt = np.linalg.svd(E)
rank = np.linalg.matrix_rank(E)
Z = Vt[rank:, :].T

Up, Sp, Vtp = np.linalg.svd(pbar)
d = len(p[0])
U1 = Up[:, :1 + d]
U2 = Up[:, 1 + d:]

M = [U2.T @ H @ np.diag(Z[:, i]) @ H.T @ U2 for i in range(Z.shape[1])]

# Problema de LMI con PICOS
P = picos.Problem()
c = picos.RealVariable("c", len(M))
M_i = [picos.Constant(Mi) for Mi in M]
Sum = picos.sum(c[i] * M_i[i] for i in range(len(M_i)))
P.add_constraint(Sum >> 0)
P.solve()

ci = np.array([c.np])

try:
    w = sum(ci[i] * Z[:,i] for i in range(len(c)))
except:
    w = [sum(Z[:,i]*ci[0][i] for i in range(Z.shape[1]))]
    w = w[0]

Omega = H @ np.diag(w) @ H.T
I = np.eye(3)
Lbar = np.kron(Omega, I)

pstack = np.array([p.flatten()]).T

n = len(vertices)

## Traslación
v = np.array([[1,1,1]]).T
Mtras = np.kron(np.ones((n,1)), v)

## Rotación

def angvel(x,y,z):
    angvelx = x*np.pi/180
    angvely = y*np.pi/180
    angvelz = z*np.pi/180
    return angvelx,angvely,angvelz

angvelx,angvely,angvelz = angvel(1,1,1)

def Wtot(x, y, z):
    return np.array([[0,-z*np.pi/180, y*np.pi/180],[z*np.pi/180, 0, -x*np.pi/180],[-y*np.pi/180, x*np.pi/180, 0]]) # Matriz de velocidad angular en 2D

def Mrbargen(n,W):
    return np.kron(np.eye(n),W) # Rotación

Mrbar = Mrbargen(n, Wtot(0.2,0.1,1))

## Escalado
Msbar = np.eye(n*3)

## Shearing (Cizallamiento)
hxy = 1
hyx = 1
hxz = 1
hzx = 1
hyz = 1
hzy = 1
S = np.array([[0,hxy, hxz],[hyx,0, hyz],[hzx, hzy, 0]])
Mslbar = np.kron(np.eye(n), S)

# Simulación
dt = 0.01
tf = 100
tp = tf / 1000
t = 0
tp2 = tf*0.9

h = 0.4
kt = 0.05 # Ganancia de traslación
ks = -0.0085 # Ganancia del scaling
kr = 3
ksh = 0.015

trajx, trajy, trajz = [], [], []

# Pt = 15 * np.random.rand(len(pstack), 1)
Pt = pstack.copy()

while t <= tf:
    t += dt
    if t >= tp:
        # ax.plot(Pt[0::3, 0], Pt[1::3, 0], Pt[2::3, 0], '.', markersize=0.5)
        trajx.append(Pt[0::3, 0])
        trajy.append(Pt[1::3, 0])
        trajz.append(Pt[2::3, 0])
        tp += tf / 1000
    # consenso lo ejecuto siempre como referencia
    # Pt = Pt - h*Lbar@Pt*dt
    # Pt = Pt - (h*Lbar@Pt + kt*Mtras)*dt 
    # Pt = Pt - (h*Lbar@Pt + kr*Mrbar@Pt)*dt 
    # Pt = Pt - (h*Lbar@Pt + ks*Msbar@Pt)*dt
    Pt = Pt - (h*Lbar@Pt + ksh*Mslbar@Pt)*dt
    # Pt = Pt - (h*Lbar@Pt + (ks*Msbar+ kr*Mrbar)@Pt + kt*Mtras)*dt

# Animación 3D
x = np.array(trajx)[:, :]
y = np.array(trajy)[:, :]
z = np.array(trajz)[:, :]
npuntos = x.shape[0]

x_min, x_max = np.min(x) * 1, np.max(x) * 1
y_min, y_max = np.min(y) * 1, np.max(y) * 1
z_min, z_max = np.min(z) * 1, np.max(z) * 1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmap = plt.get_cmap('tab10')
norm = plt.Normalize(vmin=0, vmax=x.shape[1] - 1)
colors = cmap(norm(np.arange(x.shape[1])))
ax.set_prop_cycle(color=colors)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

edge_lines = [ax.plot([], [], [], '--', color="black", linewidth=0.8)[0] for _ in range(len(edges))]
lines = [ax.plot([], [], [], lw=2)[0] for _ in range(6)]
dots = [ax.plot([], [], [], 'o', markersize=3)[0] for _ in range(6)]

def init():
    for line, dot in zip(lines, dots):
        line.set_data([], [])
        line.set_3d_properties([])
        dot.set_data([], [])
        dot.set_3d_properties([])
    for edge_line in edge_lines:
        edge_line.set_data([], [])
        edge_line.set_3d_properties([])
    return lines + dots + edge_lines

def update(frame):
    for i, (line, dot) in enumerate(zip(lines, dots)):
        line.set_data(x[:frame, i], y[:frame, i])
        line.set_3d_properties(z[:frame, i])
        dot.set_data(x[frame - 1, i], y[frame - 1, i])
        dot.set_3d_properties(z[frame - 1, i])
    for i, edge in enumerate(edges):
        x_edge = [x[frame - 1, edge[0] - 1], x[frame - 1, edge[1] - 1]]
        y_edge = [y[frame - 1, edge[0] - 1], y[frame - 1, edge[1] - 1]]
        z_edge = [z[frame - 1, edge[0] - 1], z[frame - 1, edge[1] - 1]]
        edge_lines[i].set_data(x_edge, y_edge)
        edge_lines[i].set_3d_properties(z_edge)
    return lines + dots + edge_lines

ani = FuncAnimation(fig, update, frames=npuntos, init_func=init, blit=True, interval=0.1)
# plt.title("Traslación, rotación y escalado en 3D desde posiciones arbitrarias")
plt.title("Cizallamiento")
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel("z")
plt.grid(alpha=0.7, linestyle='--', linewidth=0.3)

plt.show()
