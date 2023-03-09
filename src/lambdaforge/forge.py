import numpy.random as rd
import numpy as np
from dataclasses import dataclass
from .lambda_term import Lambda

# a sampler for critical Bienaymé-Galton-Watson trees
# with reproduction distribution b*delta_0 + a*delta_1 + b*delta_2
@dataclass
class CGBW_tree:
    a:float = 0.2
    b:float = 0.5 * 0.8
    
    # method to sample nodes from the tree
    def sample_nodes(self, size: np.int64):
        v = rd.rand(size)
        return (v > (1 - self.b)) * 1 - (v < self.b) * 1
        
# a sampler for De Bruijn indices with geometric distribution
@dataclass
class Geometric:
    p:float = 0.05
    
    # method to sample De Bruijn indices
    def sample_de_bruijn(self, size: np.int64):
        return rd.geometric(self.p, size)

# the lambda term sampler. It generates terms of size between minimum and maximum,
# according to the tree_sampler and the de_bruijn_sampler specified
@dataclass
class Forge:
    minimum: int = 10
    maximum: int = 100
    tree_sampler = CGBW_tree()
    de_bruijn_sampler = Geometric()
    
    # method to sample the tree structure through the node kinds. (variables:-1,abstractions:0,application:1)
    # a valid set of kinds is such that #variables-#applications=1
    def nodes_may_fail(self):
        # sample nodes
        inc = self.tree_sampler.sample_nodes(self.maximum)
        # compute the cumulative sum
        s = np.cumsum(inc)
        # check if the list of increments reaches the minimum size
        if s[self.minimum - 1] == -1:
            return inc[:self.minimum], s[:self.minimum]
        else:
            # search for the first -1 after the minimum size
            i = np.argmax(s[self.minimum:] == -1) + self.minimum
            if s[i] == -1:
                return inc[:i+1], s[:i+1]
            else:
                return None
    
    # try sampling a valid set of nodes until it find a valid one
    def nodes(self):
        l = self.nodes_may_fail()
        while l == None:
            l = self.nodes_may_fail()
        inc, s = l
        return inc, s
    
    # reroot the potential lambda term in the (unique way) that it forms a unique connected term
    # (see Dvoretzky-Motzkin cycle lemma)
    def rotation(self, inc, s):
        # find the position of the first -1
        i = np.argmin(s) + 1
        l = s.size
        # rotate the list of increments and compute the new cumulative sum
        inc2 = np.concatenate((inc[i:l], inc[0:i]))
        return inc2, np.cumsum(inc2)
    
    # sample a lambda term
    def craft(self) -> Lambda:
        inc, s = self.nodes()
        inc, s = self.rotation(inc, s)
        # sample De Bruijn indices
        var_ind = self.de_bruijn_sampler.sample_de_bruijn(inc.size)
        # create the Lambda term from the kinds and the indices
        return Lambda.from_kind(inc, var_ind)
