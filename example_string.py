import matplotlib.pyplot as plt
import numpy as np
from lambdaforge.forge import Forge, Geometric,CGBW_tree
from lambdaforge import visualization as vs, production as pr, visualization as vs

f = Forge(4,8)
f.de_bruijn_sampler = Geometric(0.8)
f.tree_sampler = CGBW_tree(0.4,0.3)
l = f.craft()
print(pr.polish_de_bruijin(l))
print(pr.parenthesis_de_bruijin(l))
voc_free = np.array(['a','b','c','d','e','f','g','h','j','k','l','m','n','o','p','q'])
voc_bounded = np.array(['u','v','w','x','y'])
print(pr.polish_named(l,voc_bounded,voc_free))
print(pr.parenthesis_named(l,voc_bounded,voc_free))
fig, ax = plt.subplots(1, 1)
vs.draw(l,fig,ax,edge_width = 1,binding_width = 5,edge_color='lightblue',dot_size=100,binding_alpha=0.5)
fig.set_size_inches(5, 20)