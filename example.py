import matplotlib.pyplot as plt
from lambdaforge.forge import Forge, Geometric, CGBW_tree
from lambdaforge import production as pr, visualization as vs, reduction as rd
import numpy as np
k = 16
n = 2**k
forge = Forge(n,
              2*n)
l0 = forge.craft()
fig, ax = plt.subplots(1, 1)
vs.draw(l0,fig,ax,edge_width = 0.5,binding_width = 2,edge_color='lightblue',dot_size=2,binding_alpha=0.1)
fig.set_size_inches(10, 20)
#print(pr.polish_de_bruijin(l))
#vs.draw(rd.reduce_normal_order_step(l),fig,ax2)

l=l0
t = 5000
k = 50
area = np.zeros(t//k)
mass = np.zeros(t//k)

for i in range(t):
    
    
    if i%k == 0 :
        print(i)
        print(l.size)
        fig, ax = plt.subplots(1, 1)
        vs.draw(l,fig,ax,edge_width = 0.5,binding_width = 1.6,edge_color='lightblue',dot_size=2,binding_alpha=0.1)
        fig.set_size_inches(10, 20)
        plt.savefig(f"plot_{i+5000}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        area[i//k] = np.sum(l.height())
        mass[i//k] = l.size
    l = rd.reduce_applicative_order_step(l)
    if l.size > 4*n : 
        break

plt.plot(np.arange(t//k),area)
plt.plot(np.arange(t//k),mass)

#l2.draw(fig,ax2,edge_color='lightblue',dot_size=500,binding_alpha=0.5)
#ax2.patch.set_facecolor('black')


#fig.patch.set_facecolor('black')
#fig.set_size_inches(40, 80)
''' 


fig, axs = plt.subplots(nrows=2, ncols=1)
plt.show()


x_pos,y_pos = l.layout()
y_pos += 0*np.max(y_pos)
theta = np.pi*0.0+ 1.0 *np.pi *(x_pos-np.min(x_pos))/np.max(x_pos)
x_pol = y_pos * np.cos(theta)
y_pol = y_pos * np.sin(theta)
fig, ax = l.draw(layout = (x_pos,y_pos),edge_width = 2,binding_width = 5,edge_color='lightblue',dot_size=500,binding_alpha=0.5)
t = np.pi*0.0+ 1.0 *np.pi * np.linspace(0,1,100)
m = np.max(y_pos)
#◘plt.plot(m*np.cos(t),m*np.sin(t))
fig.set_size_inches(40, 40)
ax.patch.set_facecolor('black')
fig.patch.set_facecolor('black')
plt.show()
'''
