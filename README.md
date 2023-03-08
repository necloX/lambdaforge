# Lambda Forge
`lambdaforge` is a Python package for generating random lambda based on the theory of random process. Lambda terms enjoy an efficient contiguous representation in memory, encoded by four `numpy` array. It also provides a set of tools for visualizing, reducting and printing lambda terms.

## Installation

You can install it by cloning this repository and running pip install `path_to_your_clone`.

## Usage

To generate a random lambda term, use the `craft` method of the `Forge`

```
import lambdaforge as lf
import lambdaforge.production as pr
l = lf.forge.Forge(8,16).craft()
print(pr.parenthesis_de_bruijin(l))
```
This sample a lambda term of size between 8 and 16 and print it with De Bruijn indices.

## Random process and random trees

Fundamentaly, we see a lambda term as a tree where the leafs (the `variable`) are decorated by De Bruijn indices. Every other nodes is either an `abstraction`, which have a unique child, or an `application`, which have two child, the argument and the function.

For a reference on random trees encoded by random processes see T. Duquesne et J.-F. Le Gall, Random Trees, Lévy Processes and Spatial Branching Processes. A related notion is the one of Bolzman samplers (see ). While they are fundamentaly equivalent, Boltzman samplers emphasises the recursive nature of trees, while trees encoded by process emphasis the imperative nature of tree traversal.

### How the encoding work
Here, the term process refers to a vector of size `n` that encode a given tree. Trees underlying lambda terms Traditionnaly, in the study of random trees, we encode them via their `height` process: nodes are visited in a depth first manner and we keep track of their distance from the root, their 'height'. It can be shown that this process totally encode a given tree. For instance, we can check if a node `a` is on the path from the root to `b` by checking if the height process reach a minimum between the visit of the two. 

Another way to encode a tree, is to keep track of the closest node of size 2, eg applications, from which we are a subterm on the function side. We call that vector the `application_kernel`. This process also totally encode the tree. We can recover the height process by *integrating* the vector `(np.ones(n)` against this kernel where *integrating* which means computing the folowing function:

```
@nb.njit
def forward_integral(application_kernel:np.array(np.int64),kind:np.array(np.int64),v:np.array(np.int64)):
    n = application_kernel.size
    h = np.zeros(n,dtype= np.int64)
    for i in range(n-1):
        if kind[i] == 0:
            h[i+1] = h[i]+v[i]
        elif kind[i] == 1:
            h[i+1] = h[i]+v[i]
        else:
            h[i+1] = h[application_kernel[i]]+v[i]
    return h
``` 
This `forward_integral` method iteratively compute the cumulative sum of `v` along the ancestors node of a given node. For convenience, we also define the `abstraction_kernel` and the `variable_kernel`, even if our `application_kernel` totally encode the tree. Integrating diverse vectors is a fundamental way of interracting with `lambdaforge`. 

### Resolving bindings by integration process
A fundamental example gives us a way to compute the bindings of a lambda terms. If `abs` is an indicator vector of abstractions, integrating `abs` against the 

## License

This project is licensed under the MIT License.
