import numpy as np
from lambdaforge.lambda_term import Lambda

# perform one step of beta reduction at a specified index in the lambda term
def reduce_at(l:Lambda, index):
    # get the subterm under the abstraction of the redex
    l1 = l.subterm(index + 2)
    # get the number of abstractions under each nodes in l1
    habs = l1.height_abs()
    # get the variables in l1 that bind just below the root
    variables = np.flatnonzero((l1.de_bruijn_indices == habs + 1) * l1.variables())
    # get the subterm in argument position
    l2 = l.subterm(l.variable_kernel[index + 1] + 1)
    # Substitute the bounded variables in l1 with the argument
    l3 = l1.substitute(variables, l2)
    # Replace the redex with the result of the substitution
    return l.replace(index, l3)

# perform one step of beta reduction using normal order
def reduce_normal_order_step(l:Lambda):
    # if there are no redexes in the lambda term, return the term unchanged
    if np.max(l.redex()) == 0:
        return l
    # find the leftmost redex
    i = np.argmax(l.redex())
    # reduce at the leftmost redex
    return reduce_at(l, i)

# perform one step of beta reduction using applicative order
def reduce_applicative_order_step(l:Lambda):
    # if there are no redexes in the lambda term, return the term unchanged
    if np.max(l.redex()) == 0:
        return l
    # find the rightmost redex
    r = l.redex()[::-1] # reverse the redex array
    i = len(r) - np.argmax(r) - 1 # find the index of the rightmost redex in the original array
    # reduce at the rightmost redex
    return reduce_at(l, i)
