import matplotlib.pyplot as plt
from lambdaforge.forge import Forge
from lambdaforge import visualization as vs

k = 16
n = 2**k
l = Forge(n,2*n).craft()
fig, ax = plt.subplots(1, 1)
vs.draw(l,fig,ax,edge_width = 0.1,binding_width = 0.3,edge_color='lightblue',dot_size=5,binding_alpha=0.2)
fig.set_size_inches(5, 20)