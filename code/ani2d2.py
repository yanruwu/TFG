import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, Matrix
import picos
from matplotlib.animation import FuncAnimation

# Definición de los parámetros
n = 18
k = 9
h = 2

# Generar posiciones de los vértices
angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
p = np.array([(np.cos(a), np.sin(a)) for a in angles])

# Generar índices de vértices como un conjunto
vertices = set(range(1, n + 1))

# Generar aristas para el n-gon regular
edges = []
vertices_list = list(vertices)  # Convertir conjunto en lista para indexar
for i in range(n):
    edges.append((vertices_list[i], vertices_list[(i + 1) % n]))

# Generar aristas para el polígono estrellado con tamaño de paso k
for i in range(n):
    edges.append((vertices_list[i], vertices_list[(i + k) % n]))

pbar = np.column_stack((p, np.ones(len(vertices))))

# Matriz de incidencia
H = np.zeros([len(vertices), len(edges)])
for i in enumerate(edges):
    H[i[1][0] - 1, i[0]] = 1
    H[i[1][1] - 1, i[0]] = -1

E = pbar.T @ H @ np.diag(H[0, :])
for i in range(1, len(vertices)):
    E = np.append(E, pbar.T @ H @ np.diag(H.T[:, i]), axis=0)

# Descomposición en valores singulares (SVD)
U, S, Vt = np.linalg.svd(E)
rank = np.linalg.matrix_rank(E)
Z = Vt[rank:, :].T

Up, Sp, Vtp = np.linalg.svd(pbar)
d = len(p[0])
U1 = Up[:, :1 + d]
U2 = Up[:, 1 + d:]

# Crear lista de matrices M
M = []
for i in range(Z.shape[1]):
    M.append(U2.T @ H @ np.diag(Z[:, i]) @ H.T @ U2)

# Resolver el problema LMI con PICOS
P = picos.Problem()
c = picos.RealVariable("c", len(M))
M_i = [picos.Constant(Mi) for Mi in M]
Sum = picos.sum(c[i] * M_i[i] for i in range(len(M_i)))
P.add_constraint(Sum >> 0)
P.solve()

ci = np.array([c.np])
try:
    w = sum(ci[i] * Z[:, i] for i in range(len(c)))
except:
    w = [sum(Z[:, i] * ci[0][i] for i in range(Z.shape[1]))]
    w = w[0]

Omega = H @ np.diag(w) @ H.T

I = np.eye(2)
Lbar = np.kron(Omega, I)
h = 1  # Ganancia arbitraria
pstack = np.array([p.flatten()]).T

# Inicialización de la posición
Pt = 100 * np.random.rand(len(pstack), 1)

dt = 0.006
tf = 20
tp = 0
tp2 = tf / 2
t = 0.1

trajx = []  # Para almacenar las trayectorias
trajy = []

v = np.array([[1, 1]]).T
Mtras = np.kron(np.ones((n, 1)), v)

while t <= tf:
    t += dt
    if t >= tp:
        trajx.append(Pt[0::2, 0])
        trajy.append(Pt[1::2, 0])
        tp += tf / 800

    Pt = Pt - h * Lbar @ Pt * dt

x = np.array(trajx)[:, :]
y = np.array(trajy)[:, :]
npuntos = x.shape[0]

# Calcular los límites de los ejes con un margen del 10%
x_min = -np.max(x) * 0
x_max = np.max(x) * 1.5
y_min = -np.max(y) * 0
y_max = np.max(y) * 1.5

# Inicializar la figura y los ejes con los límites calculados
fig, ax = plt.subplots()
cmap = plt.get_cmap('tab10')
norm = plt.Normalize(vmin=0, vmax=np.array(trajx).shape[1] - 1)
colors = cmap(norm(np.arange(np.array(trajx).shape[1])))
plt.gca().set_prop_cycle(color=colors)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Crear una lista de líneas para representar las aristas del grafo
edge_lines = [ax.plot([], [], '--', color="black", linewidth=0.8)[0] for _ in range(len(edges))]

# Crear una lista de líneas vacías que se actualizarán en cada cuadro de la animación
lines = [ax.plot([], [], lw=0.6)[0] for _ in range(n)]

# Crear una lista de puntos para representar los nodos al final de cada trayectoria
dots = [ax.plot([], [], 'o', markersize=3)[0] for _ in range(n)]

# Función de inicialización: crea líneas y puntos vacíos para todas las trayectorias y aristas del grafo
def init():
    for line, dot in zip(lines, dots):
        line.set_data([], [])
        dot.set_data([], [])
    for edge_line in edge_lines:
        edge_line.set_data([], [])
    return lines + dots + edge_lines

# Función de actualización: actualiza los datos de las líneas de las trayectorias, los puntos y las aristas del grafo
def update(frame):
    for i, (line, dot) in enumerate(zip(lines, dots)):
        line.set_data(x[:frame, i], y[:frame, i])
        dot.set_data(x[frame-1, i], y[frame-1, i])  # Coloca el punto al final de cada trayectoria
    for i, edge in enumerate(edges):
        x_edge = [x[frame-1, edge[0]-1], x[frame-1, edge[1]-1]]
        y_edge = [y[frame-1, edge[0]-1], y[frame-1, edge[1]-1]]
        edge_lines[i].set_data(x_edge, y_edge)
    return lines + dots + edge_lines

# Crear la animación
ani = FuncAnimation(fig, update, frames=npuntos, init_func=init, blit=True, interval=10)

# Mostrar la animación
plt.title("Estabilización")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha=0.7, linestyle='--', linewidth=0.3)
plt.show()
