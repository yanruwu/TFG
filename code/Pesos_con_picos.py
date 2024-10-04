# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 19:39:47 2024

@author: abierto
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:33:38 2020
Ejemplo de obtención vector de stress para el paper de Zhao
Usando picos
@author: juanjimenez
"""
import numpy as np
import matplotlib 
import matplotlib.pyplot as pl
import picos

#creamos la matriz de incidencia en su caso tiene 7 vertices
V = {1,2,3,4,5,6,7} 
#y 12 aristas
A = [(1,2),(1,3),(1,4),(1,5),(2,4),(2,7),(3,5),(3,6),(4,5),(4,6),(5,7),(6,7)] 
# Hemos orientado el grafo por tanto.
# creamos la configuracion, le damos directamente la forma de P(r)
Pr = np.array([[2,0],[1,1],[1,-1],[0,1],[0,-1],[-1,1],[-1,-1]])

H=np.zeros([len(V),len(A)])
for i in enumerate(A):
    H[i[1][0]-1,i[0]] = -1
    H[i[1][1]-1,i[0]] = 1
#Nuestra matriz de incidencia es la traspueta de la que define Zhao!!!


Pr_b = np.append(Pr,np.ones([Pr.shape[0],1]),axis=1)

#calculamos la matriz E 
E =Pr_b.T @ H @ np.diag(H[0,:])
for i in range(1,len(V)):
    E =np.append(E,Pr_b.T @ H @ np.diag(H[i,:]),axis=0)

#calculamos ahora el subespacio nulo de E
#primero calculamos el rango de E, así sabemos a partir de qué vector
#columna de la matriz V E=U@S@V.T, empieza la base del kernel de A.
rango = np.linalg.matrix_rank(E)

#calculamos ahora la descomposicion en valores singulares de E,
U,S,Vt = np.linalg.svd(E,full_matrices= True) #no es necesario full matrices
filas = Vt.shape[0]
#y el espacio nullo de E (Cada columna de Z un vector de la base del
# espacio Nulo)
Z = Vt[rango:filas,:].T
#comprbación de que Z expande el Null(E)
print(E@Z)

#calculamos la descomposicion SVD de Pr_b
#y creamos una lista de matrices Mi
M = []
Up,Sp,Vtp = np.linalg.svd(Pr_b)
U2 = Up[:,Pr.shape[1]+1:]
for i in Z.T:
    M.append(U2.T @ H @ np.diag(i) @ H.T @ U2)
    
#planteamos un bonito problema LMI en picos    
P = picos.Problem()

c = picos.RealVariable("c",len(M))
M_L = [picos.Constant(Mi) for Mi in M]
S = picos.sum(M_L[i]*c[i] for i in range(len(M_L)))
P.add_constraint(S>>0)
print(P)
P.solve()

#esto puede dar errores ojo
cn = np.array([c.np])

#Construimos el vector de pesos y la matriz de pesos.
w = [sum(Z[:,i]*cn[i] for i in range(Z.shape[1]))]
w = w[0]
Omg = H @ np.diag(w) @ H.T    
#dibujamos que siempre ayuda...
pl.plot(Pr[:,0],Pr[:,1],'o')
for i in enumerate(Pr):
    pl.gca().text(i[1][0]+0.01,i[1][1]+0.01,str(i[0]+1))
for i in enumerate(A):
    pl.plot(Pr[[i[1][0]-1,i[1][1]-1],0],Pr[[i[1][0]-1,i[1][1]-1],1])
    pl.gca().text(np.mean(Pr[[i[1][0]-1,i[1][1]-1],0]),\
        np.mean(Pr[[i[1][0]-1,i[1][1]-1],1]),str(w[i[0]])[0:6],\
            rotation=180*np.arctan2(Pr[i[1][0]-1,1]-Pr[i[1][1]-1,1],\
                                Pr[i[1][0]-1,0]-Pr[i[1][1]-1,0])/np.pi)
#que pasa si empezamos en la configuracion deseada
pl.figure()

Prs = np.reshape(Pr,[2*Pr.shape[0],1]) #posisciones de referencia apiladas
P = Prs.copy()
pl.plot(P[0::2,0],P[1::2,0],'^k',linewidth =0.1)
dt = 0.01

tf = 100
tp = tf/1000
t = 0
I = np.eye(2)

Pc = P

################no se mueve estan en el kernel de los pesos####################
while t <= tf:
    t += dt
    if t >= tp:
        pl.plot(Pc[0::2,0],Pc[1::2,0],'.',linewidth =0.1)
        tp += tf/1000
    #consenso lo ejecuto siempre como referencia
        
    Pc = -10*np.kron(Omg,I)@Pc*dt + Pc    
    

Pst = np.array([Pc[0::2,0],Pc[1::2,0]]).T
for i in enumerate(Pst):
    pl.gca().text(i[1][0],i[1][1],str(i[0]+1))
for i in enumerate(A):
    pl.plot(Pst[[i[1][0]-1,i[1][1]-1],0],Pst[[i[1][0]-1,i[1][1]-1],1])
# ###################################################################

#por último podemos probar por euler que empezando en codiciones arbitrarias el
#sistema converge a la configuraación deseada
pl.figure()
P  = 10*np.random.rand(14,1)-5 #posiciones iniciales
Prs = np.reshape(Pr,[2*Pr.shape[0],1]) #posisciones de referencia apiladas

pl.plot(P[0::2,0],P[1::2,0],'^k',linewidth =0.1)
dt = 0.01

tf = 100
tp = tf/1000
t = 0
I = np.eye(2)

Pc = P

################consenso######################################
while t <= tf:
    t += dt
    if t >= tp:
        pl.plot(Pc[0::2,0],Pc[1::2,0],'.',linewidth =0.1)
        tp += tf/1000
    #consenso lo ejecuto siempre como referencia
        
    Pc = -10*np.kron(Omg,I)@Pc*dt + Pc    
    

Pst = np.array([Pc[0::2,0],Pc[1::2,0]]).T
for i in enumerate(Pst):
    pl.gca().text(i[1][0],i[1][1],str(i[0]+1))
for i in enumerate(A):
    pl.plot(Pst[[i[1][0]-1,i[1][1]-1],0],Pst[[i[1][0]-1,i[1][1]-1],1])
# ###################################################################