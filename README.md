# Lambda Forge
`lambdaforge` is a Python package for sampling large random lambda terms. It is based on the theory of random processes, which ensures an efficient representation with a contiguous block of memory. Each term is encoded by four numpy arrays. The package provides tools for visualizing, reducing, and printing lambda terms.

![alt text](https://github.com/necloX/lambdaforge/blob/main/lambda_size_5720903.png?raw=true)

## Installation

You can install it by cloning this repository and running `pip install path_to_your_clone`.

## Usage

To generate a random lambda term, use the `craft` method of the `Forge`

```
import lambdaforge as lf
import lambdaforge.production as pr
l = lf.forge.Forge(8,16).craft()
print(pr.parenthesis_de_bruijn(l))
```
This will sample a lambda term of size between 8 and 16 and print it with De Bruijn indices. The underlying tree is a critical Bienaymé-Galton-Watson tree.

## Random process and random trees

A lambda term is viewed as a tree with leaves (variables) decorated by De Bruijn indices. All other nodes are either abstractions (which have a unique child) or applications (which have two children: the argument and the function).

For a reference on random trees encoded by random processes, see T. Duquesne and J.-F. Le Gall, [Random Trees, Lévy Processes and Spatial Branching Processes](https://www.imo.universite-paris-saclay.fr/~jean-francois.le-gall/Mono-revised.pdf)). A related notion is that of Boltzmann samplers, see Pierre Lescanne, [Boltzmann samplers for random generation of lambda terms](https://hal-ens-lyon.archives-ouvertes.fr/ensl-00979074v2). While these are fundamentally equivalent, Boltzmann samplers enphasize the recursive nature of trees, while the encoding process emphasize the iterative nature of tree traversal.

### How the encoding works
The term "process" refers to a vector of integers of size n that encodes a given tree of size n. Traditionally, random trees are encoded via their "height" process: nodes are visited in a depth-first manner, and the process represent their distance from the root (their "height"). It can be shown that this process totally encodes a given tree. For instance, we can check if a node `a` is on the path from the root to `b` by checking if the height process reaches a minimum between the visits of the two. 

A celebrated result concerning critical Bienaymé-Galton-Watson trees is the convergence of such processes toward Brownian excursions, which lead in the 90s to David Aldous' theory of continuum random trees.

Another way to encode a tree is to keep track, for each index `i` of the closest application `application_kernel[i]` for which `i` is in the subterm on the function side. This process also totally encodes the tree. We can recover the height process by "integrating" the vector `numpy.ones(n)` against this kernel,  where integrating means computing the following function, which is at the heart of this project:
```
# the kind of each node (-1 for variables, 0 for abstractions and 1 for applications) can be easily computed from application_kernel
@nb.njit
def forward_integral(application_kernel:np.array(np.int64),kind:np.array(np.int64),v:np.array(np.int64)):
    n = application_kernel.size
    h = np.zeros(n,dtype= np.int64)
    for i in range(n-1):
        if kind[i] > -1:
            h[i+1] = h[i]+v[i]
        else:
            h[i+1] = h[application_kernel[i]]+v[i]
    return h
``` 
This `forward_integral` method iteratively computes the cumulative sum of `v` along the ancestors of a given node. For convenience, we also define an `abstraction_kernel` and a `variable_kernel`, even if our `application_kernel` totally encode the tree. Integrating diverse vectors against diverse kernels is a fundamental way of interacting with lambdaforge as illustrated in the following example.

### Resolving bindings by integrating processes
If `abs` is an indicator vector of abstractions, integrating `abs` gives us `habs`, a process representing the number of abstraction under each node. Note that `variable_kernel` gives for each index `a`, the rightmost variable in the subterm at `a`.

To know if a variable at index `v` is binded to an abstraction at index `a`, we only need to check  that 
- `a <= v <= variable_kernel[a]` which means that `v` is in the subterm at index `a`
- `de_bruijn_indices[v] == habs[v]-habs[a]+1`

## Plan for future development
- The obvious next thing to do is to implement other samplers than critical Bienaymé-Galton-Watson trees and geometric De Bruijn indices. The way the `Forge` works makes it rather easy, just write classes that implement `sample_nodes`for trees and `sample_de_bruijn` for indices.
- There is potential for expanding `lambdaforge` to support other process calculi, such as the pi calculus or pattern calculus.
- Another area for future improvement is in the representation of reduction in the system. Currently, `lambdaforge` recomputes the kernels at each step of reduction, which can be computationally expensive. Ideally we would like a representation of reduction through a view of the original term, which would be compositional and with a small memory footprint.

## License

This project is licensed under the MIT License.