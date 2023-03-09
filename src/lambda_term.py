import numpy as np
import lambdaforge.computation as comp 
from dataclasses import dataclass

@dataclass
class Lambda:
    size:np.int64
    abstraction_kernel:np.array(np.int64) # the index of the closest parent abstraction of each node
    application_kernel:np.array(np.int64) # the index of the closest parent application for which the node is a subterm of the function
    variable_kernel:np.array(np.int64)   # the right most variable of the subtree attached at each node
    de_bruijn_indices:np.array(np.int64) # the de Bruijn index of each variable in the lambda term. Value for non-variable node is irrelevant
    
    # a static method that creates a Lambda object from a given kind vector (variable:-1,abstraction:0,applicatoin:1) and de Bruijn indices
    def from_kind(kind:np.array(np.int64),de_bruijn_indices:np.array(np.int64)):
        abstraction_kernel,application_kernel,variable_kernel=comp.compute_kernel(kind)
        return Lambda(kind.size, abstraction_kernel, application_kernel, variable_kernel, de_bruijn_indices)
           
    # kind of each node (variable:-1,abstraction:0,applicatoin:1)
    def node_kind(self): 
        dq = np.roll(self.application_kernel,-1)-self.application_kernel # a numpy array of the difference between the application kernel and its shifted version
        dq[self.size-1] = -1 # set the last element of dq to -1
        dq = np.sign(dq) # set each element of dq to its sign
        return dq
    
    # returns the subterm of the lambda term rooted at a given index
    def subterm(self,index:int):
        var = self.variable_kernel[index] # the index of the closest parent variable of the given index
        size = var+1-index # the size of the subterm rooted at the given index
        abstraction_kernel = self.abstraction_kernel[index:var+1]-index # a numpy array of the indices of the parent abstractions of the nodes in the subterm
        abstraction_kernel = np.where(abstraction_kernel<0,-1,abstraction_kernel) # set any negative element of abstraction_kernel to -1
        application_kernel = self.application_kernel[index:var+1]-index # a numpy array of the indices of the left-most child applications of the nodes in the subterm
        application_kernel = np.where(application_kernel<0,-1,application_kernel) # set any negative element of application_kernel to -1
        variable_kernel = self.variable_kernel[index:var+1]-index # a numpy array of the indices of the parent variables of the nodes in the subterm
        de_bruijn_indices = self.de_bruijn_indices[index:var+1] # a numpy array of the de Bruijn indices of the variables in the subterm
        return Lambda(size, abstraction_kernel, application_kernel, variable_kernel, de_bruijn_indices)
     
    # returns a new lambda term obtained by replacing the subterm at a given index with another lambda term that may capture variables
    def replace(self,index:int,other):
        var = self.variable_kernel[index]
        abstraction_kernel = comp.forward_insert(self.abstraction_kernel, other.abstraction_kernel, index, var)
        application_kernel = comp.forward_insert(self.application_kernel, other.application_kernel, index, var)
        variable_kernel = comp.backward_insert(self.variable_kernel, other.variable_kernel, index, var)
        de_bruijn_indices = comp.de_bruijn_insert(self.de_bruijn_indices,other.de_bruijn_indices,index, var)
        
        return Lambda(abstraction_kernel.size, abstraction_kernel, application_kernel, variable_kernel, de_bruijn_indices)

    # capture avoiding substitution of 'other' for each variable in 'variables'
    def substitute(self,variables,other):
        abstraction_kernel = comp.forward_substitute(self.abstraction_kernel, other.abstraction_kernel,  variables)
        application_kernel = comp.forward_substitute(self.application_kernel, other.application_kernel, variables)
        variable_kernel = comp.backward_substitute(self.variable_kernel, other.variable_kernel, variables)
        de_bruijn_indices = self.de_bruijn_indices
        habs = self.height_abs()
        de_bruijn_indices = np.where(de_bruijn_indices>habs,de_bruijn_indices-1,de_bruijn_indices)
        de_bruijn_indices =  comp.de_bruijn_substitute(de_bruijn_indices, other.de_bruijn_indices, variables, habs, other.free_var())
        return Lambda(abstraction_kernel.size, abstraction_kernel, application_kernel, variable_kernel, de_bruijn_indices)
        
    # indicator vector of applications
    def applications(self): return np.clip(self.node_kind(),0,1)

    # indicator vector of abstractions
    def abstractions(self): return 1-np.abs(self.node_kind())

    # indicator vector of variables
    def variables(self): return np.clip(-self.node_kind(),0,1)
    # indices of applications
    def applications_node(self): return self.applications() * np.arange(self.size)

    # indices of abstractions
    def abstractions_node(self): return self.abstractions() * np.arange(self.size)

    # indices of variables
    def variables_node(self): return self.variables() * np.arange(self.size)
    # vector with the index of the parent of each node, -1 for the root
    def parents(self): 
        ap = np.max([self.abstraction_kernel,self.application_kernel,np.roll(self.application_kernel,1)],axis=0)
        return ap
    # vector with the index of the abstraction of each bounded variable, and the amount of necessary
    # abstraction beyond the root of the term to make a free variable bounded
    def bindings(self): 
        # search for abstractions at a certain distance, in place, initialize at 1
        abs_at_distance = self.abstraction_kernel 
        # get the variables that are bounded by abstractions
        bounded_var = self.bounded_var() 
        # get the maximum De Bruijn index of bounded variables
        max_depth = np.max(bounded_var * self.de_bruijn_indices) 
        
        # initialize future bindings
        b = np.zeros(self.size, int) 
        
        # for each depth level, look for abstractions at distance +1 and report their position to the variable
        for i in range(max_depth):
            # determine whether or not the variable at this index has De Bruijn index i+1
            to_replace = (self.de_bruijn_indices == i+1) * bounded_var 
            # update the bindings where to_replace is 1
            b = to_replace * abs_at_distance + (1-to_replace) * b
            # search for the abstractions at distance +1
            abs_at_distance = self.abstraction_kernel[abs_at_distance] 
        
        # determine the number of abstractions before the root that should be added to close bind the free variable
        free_bindings = (self.de_bruijn_indices-self.height_abs()+1) * self.free_var()
        # store this information with negative integers in b
        b -= free_bindings
        return b  

    # for each node i, sum the input vector v between v and the root
    def forward_integral(self, v: np.array(np.int64)): 
        return comp.forward_integral(self.application_kernel, self.node_kind(), v)

    # for each node i, sum the input vector v on  the subtree at index i
    def backward_integral(self, v: np.array(np.int64)): 
        return comp.backward_integral(self.variable_kernel, self.node_kind(), v)

    # get the height process of the tree
    def height(self) -> np.array(int):
        ones = np.ones(self.size, dtype=np.int64)
        h = self.forward_integral(ones)
        return h

    # get the number of bastraction below each node
    def height_abs(self): 
        return self.forward_integral(self.abstractions())

    # get the variables that are bounded by abstractions
    def bounded_var(self): 
        return (self.de_bruijn_indices <= self.height_abs()) * self.variables()

    # get the free variables
    def free_var(self): 
        return (self.de_bruijn_indices > self.height_abs()) * self.variables()

    # compute a nice layout for the tree
    def layout(self):
        var = self.variables()
        x_pos = self.backward_integral(np.cumsum(var) * var) /self.backward_integral(var)
        y_pos = self.height()
        return x_pos, y_pos

    # compute the redex of the tree
    def redex(self):
        return self.applications() * np.roll(self.abstractions(),-1)
            