import numpy as np
import matplotlib.pyplot as plt
import math as math

from sympy import symbols, Matrix

import picos

## Graph setup
vertices = {1,2,3,4,5,6} # Vertices
edges = [(1,2),(1,3),(1,5),(1,6),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6),(4,5),(4,6)] # Edges


## Positions
p1 = np.array([-1,-1])
p2 = np.array([-1,1])
p3 = -p2
p4 = -p1
p5 = np.array([2,0])
p6 = -p5

# p1 = np.array([-1,1])
# p2 = np.array([-1,-1])
# p3 = -p1
# p4 = -p2

p = np.array([p1,p2,p3,p4,p5,p6]) 
# p = np.array([p1,p2,p3,p4]) 

pbar = np.column_stack((p,np.ones(len(vertices))))

## Incidence Matrix 
H = np.zeros([len(vertices),len(edges)]) # H is defined as a +-1 matrix, where the rows indicate the vertices and the columns, the edges. The sign is due to convention an indicates that it is a directed graph.
for i in enumerate(edges): # First element from enumerate indicates edge index, the other is the vertices connected by it
    H[i[1][0]-1, i[0]] = 1 # i[1][0]-1 takes the first vertex index connected by the edge, the second element associates an edge to it. +1 due to convention
    H[i[1][1]-1, i[0]] = -1 # i[1][0]-1 takes the second vertex index connected by the edge, the second element associates an edge to it. -1 due to convention

E =pbar.T @ H @ np.diag(H[0,:]) # Creating the first element from which the matrix will be built.

for i in range(1, len(vertices)): # Using a loop with a range from 1 to add the next elements
    E = np.append(E,pbar.T @ H @ np.diag(H.T[:,i]), axis = 0)


## SVD (Singular Value Decomposition)
U, S, Vt = np.linalg.svd(E)

rank = np.linalg.matrix_rank(E)
Z = Vt[rank:,:].T # This z should represent a basis of the nullspace of E. To check it: E @ Z = 0

Up, Sp, Vtp = np.linalg.svd(pbar)

d = len(p1) # Dimensions
U1 = Up[:,:1+d] # The first 1+d columns are U1
U2 = Up[:, 1+d:] # The rest of the columns are U2

# In this case Z only has 1 column, but in case there were more, we have to make a loop for each column.
M = [] # We create it as a list so it will be storing each matrix resulting from each i separatedly, that way we will be able to call it after individually.
for i in range(Z.shape[1]): # Loops for each column of Z
    M.append(U2.T @ H @ np.diag(Z[:,i]) @ H.T @ U2)


## LMI problem M_i * c_i > 0, solve with PICOS

P = picos.Problem()
c = picos.RealVariable("c", len(M))
M_i = [picos.Constant(Mi) for Mi in M]
Sum = picos.sum(c[i] * M_i[i] for i in range(len(M_i)))
P.add_constraint(Sum>>0)
print(P)
P.solve()

ci = np.array([c.np])

try:
    w = sum(ci[i] * Z[:,i] for i in range(len(c)))
except:
    w = [sum(Z[:,i]*ci[0][i] for i in range(Z.shape[1]))]
    w = w[0]

Omega = H @ np.diag(w) @ H.T

I = np.eye(2)
Lbar = np.kron(Omega, I)
h = 1 # Arbitrary constant gain
pstack = np.array([p.flatten()]).T

n = len(vertices)

## Traslación
v = np.array([[1,0]]).T
Mtras = np.kron(np.ones((n,1)), v)

## Rotación
angvel = np.pi/180 # 1 º/s
W = np.array([[0,-angvel],[angvel, 0]]) # Matriz de velocidad angular en 2D
Mrbar = np.kron(np.eye(n),W) # Rotación

## Escalado
Msbar = np.eye(n*2)

## Shearing (Cizallamiento)
hxy = 1
hyx = 1
S = np.array([[0,hxy],[hyx,0]])
Mslbar = np.kron(np.eye(n), S)


h = 0.5 # Arbitrary constant gain
pstack = np.array([p.flatten()]).T
Pt = pstack.copy()

Pt = 100*np.random.rand(len(pstack),1)

dt = 0.006
tf = 20
tp = 0
tp2 = tf/2
t = 0.1
kt = -2 # Ganancia de traslación
ks = -0.08 # Ganancia del scaling
kr = 15 # Ganancia de la rotación
ksh = 0.095

# plt.xlim([min(Pt) - 0.1, max(Pt) + 0.1])
# plt.ylim([min(Pt) - 0.1, max(Pt) + 0.1])

trajx = [] # Para futuro, almacenar trayectorias para dibujarlas en líneas
trajy = []

v = np.array([[1,1]]).T
Mtras = np.kron(np.ones((n,1)), v)


while t <= tf:
    t += dt


    if t >= tp:
        trajx.append(Pt[0::2,0])
        trajy.append(Pt[1::2,0])
        tp += tf/800

    Pt = Pt - h*Lbar@Pt*dt
    # Pt = Pt - (h*Lbar@Pt + kt*Mtras)*dt 
    # Pt = Pt - (h*Lbar@Pt + kr*Mrbar@Pt)*dt 
    # Pt = Pt - (h*Lbar@Pt + ks*Msbar@Pt)*dt
    # Pt = Pt - (h*Lbar@Pt + ksh*Mslbar@Pt)*dt
    # Pt = Pt - (h*Lbar@Pt + (ks*Msbar+ kr*Mrbar)@Pt + kt*Mtras)*dt



from matplotlib.animation import FuncAnimation, PillowWriter

x = np.array(trajx)[:,:]
y = np.array(trajy)[:,:]
npuntos = x.shape[0]
# Definir las aristas del grafo
edges = [(1,2),(1,3),(1,5),(1,6),(2,4),(2,5),(2,6),(3,4),(3,5),(3,6),(4,5),(4,6)]  # Edges

# Calcular los límites de los ejes con un margen del 10%
# x_min = np.min(x) * 1.5
x_min = -np.max(x) * 0
x_max = np.max(x) * 1.5
# y_min = np.min(y) * 1.5
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
edge_lines = [ax.plot([], [], '--', color = "black", linewidth = 0.8)[0] for _ in range(len(edges))]

# Crear una lista de líneas vacías que se actualizarán en cada cuadro de la animación
lines = [ax.plot([], [], lw=2)[0] for _ in range(6)]

# Crear una lista de puntos para representar los nodos al final de cada trayectoria
dots = [ax.plot([], [], 'o', markersize=3)[0] for _ in range(6)]

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
ani = FuncAnimation(fig, update, frames=npuntos, init_func=init, blit=True, interval = 0.01)

# Mostrar la animación
# plt.title("Traslación, rotación y escalado desde posiciones arbitrarias")
plt.title("Estabilización")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(alpha = 0.7 ,linestyle='--', linewidth=0.3)


# writer = PillowWriter(fps=15)
# ani.save('2d.gif', writer=writer)

plt.show()
