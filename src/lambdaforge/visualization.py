import numpy as np
import matplotlib.collections as plt_collections
from lambdaforge.lambda_term import Lambda

# draw the nodes of the lambda term
def draw_nodes(l:Lambda, ax, positions_x, positions_y, abs_color='yellow', app_color='blue', var_color='red', size=5):
    var = l.variables()
    abstractions = l.abstractions() 
    # Determine the color for each node based on whether it is a variable, abstraction or application
    colors = np.where(var, var_color, np.where(abstractions, abs_color, app_color))
    ax.scatter(positions_x, positions_y, c=colors, zorder=2, s=size)

# draw the edges of the lambda term
def draw_edges(l:Lambda, ax, positions_x, positions_y, color='blue', width=1):
    # Get the parents of each node in the lambda term
    parents = l.parents()
    # Create an array of edges connecting each parent to its child node
    edges = np.zeros([l.size, 2, 2], float)
    edges[:, 0, 0] = positions_x[parents]
    edges[:, 0, 1] = positions_y[parents]
    edges[:, 1, 0] = positions_x
    edges[:, 1, 1] = positions_y

    lc = plt_collections.LineCollection(edges[1:, :, :], zorder=1, colors=color, linewidth=width)
    ax.add_collection(lc)
    
def draw_bindings(l:Lambda,ax,positions_x,positions_y,color = 'red',alpha=0.2,width = 3):
    bv = np.nonzero( l.bounded_var())[0]
    bindings = l.bindings()
    n = bv.size
    edges = np.zeros([n,2,2],float)
    edges[:,0,0] = positions_x[bindings[bv]]
    edges[:,0,1] = positions_y[bindings[bv]]
    edges[:,1,0] = positions_x[bv]
    edges[:,1,1] = positions_y[bv]
    lc = plt_collections.LineCollection(edges[:,:,:],zorder=0, colors=color,alpha=alpha,linewidth = width)
    ax.add_collection(lc)
    
def draw(l:Lambda,
         fig,
         ax,
         layout='default',
         abs_color='yellow',
         app_color='blue',
         var_color='red', 
         dot_size=5,
         edge_color = 'lightblue',
         edge_width = 1,
         binding_color = 'red',binding_alpha=0.2,binding_width = 4): 
    if(layout=='default'):
        x_pos,y_pos = l.layout()
    else:
        x_pos,y_pos = layout
    
    draw_nodes(l,ax,x_pos,y_pos,abs_color=abs_color,app_color=app_color,var_color=var_color, size=dot_size)
    draw_edges(l,ax,x_pos,y_pos,color = edge_color,width = edge_width)
    draw_bindings(l,ax,x_pos,y_pos,color = binding_color,alpha=binding_alpha,width = binding_width)
    ax.patch.set_facecolor('black')
    fig.patch.set_facecolor('black')
    #fig.set_size_inches(10, 25)
    return fig, ax
    
    
    



    
