import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Definición de los datos
w = np.arange(0, 2 * np.pi + np.pi / 20, np.pi / 100)
cx = 2 * np.cos(w)
cy = np.sin(2 * w)

# Crear la figura y el eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Inicializar los gráficos
line1, = ax.plot([], [], [], 'r', label = r'$^{aug}\mathcal{P}$')
line2, = ax.plot([], [], [], 'b', label = r'$^{phys}\mathcal{P}$')

# Crear el wireframe
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 1, 100)
xm, Tm = np.meshgrid(x, w)
xm, lmx = np.meshgrid(x, cx)
ym, lmy = np.meshgrid(y, cy)
wireframe = ax.plot_wireframe(lmx, ym, Tm, color='gray', linewidth = 0.5, alpha=0.5, label = r'$\phi_1(\xi)$')
wireframe2 = ax.plot_wireframe(xm, lmy, Tm, linewidth = 0.5, alpha=0.5, label = r'$\phi_2(\xi)$')

# Configuración del gráfico
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 2 * np.pi)

# Función de inicialización
def init():
    line1.set_data([], [])
    line1.set_3d_properties([])
    line2.set_data([], [])
    line2.set_3d_properties([])
    return line1, line2, wireframe, wireframe2

# Función de animación
def animate(i):
    # Actualizar la primera línea
    line1.set_data(cx[:i], cy[:i])
    line1.set_3d_properties(w[:i])
    
    # Actualizar la segunda línea
    line2.set_data(cx[:i], cy[:i])
    line2.set_3d_properties(np.zeros(w[:i].shape))
    
    return line1, line2, wireframe, wireframe2

# Crear la animación
ani = FuncAnimation(fig, animate, frames=len(w), init_func=init, blit=True, interval=10)

# Mostrar la animación
plt.legend()
plt.show()
