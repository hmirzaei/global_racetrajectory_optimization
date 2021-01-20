import numpy as np
from matplotlib import pyplot as plt
from numpy import sin, cos, sqrt, arctan, arctan2, pi, tan, hypot
from matplotlib import patches
import pickle

wx,wy=pickle.load(open('/home/hamid/eskf_csv_to_ref_line/output/ref_line.p','rb'))
plt.figure()
plt.plot(wx,wy,'.')

x = wx
y = wy
th = -np.arctan2(y[-1]-y[0],x[-1]-x[0])
x2 = np.array(x) * 0
y2 = np.array(y) * 0
for i in range(len(x)):
    x2[i] = (x[i] - x[0]) * cos(th) - (y[i] - y[0]) * sin(th) 
    y2[i] = (x[i] - x[0]) * sin(th) + (y[i] - y[0]) * cos(th)


xa = list(x2)
ya = list(y2)

N = 4
l_x = x2[-1]
l_y = y2[-1]
for i in range(N-1):
    th = (i+1) * 2*np.pi/N
    for j in range(len(x2)):
        xa.append(l_x + x2[j] * cos(th) - y2[j] * sin(th))
        ya.append(l_y + x2[j] * sin(th) + y2[j] * cos(th))
    l_x = xa[-1]
    l_y = ya[-1]

plt.figure()
plt.plot(x2,y2,'.')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid() 

a=['%f,%f,1.5,1.5'%(x,y) for (x,y) in zip(xa[:-1],ya[:-1])]
print()
with open('/home/hamid/global_racetrajectory_optimization/inputs/tracks/spg.csv','w') as f:
    f.write("# x_m,y_m,w_tr_right_m,w_tr_left_m\n")
    f.write('\n'.join(a))
