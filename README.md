# Lambda Forge
`lambdaforge` is a Python package for sampling large random lambda terms. It is based on the theory of random processes, which ensures an efficient contiguous representation in memory. Each term is encoded by four numpy arrays. The package provides tools for visualizing, reducing, and printing lambda terms.

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
This will sample a lambda term of size between 8 and 16 and print it with De Bruijn indices.

## Random process and random trees

A lambda term is viewed as a tree with leaves (variables) decorated by De Bruijn indices. All other nodes are either abstractions (which have a unique child) or applications (which have two children: the argument and the function).

For a reference on random trees encoded by random processes, see T. Duquesne et J.-F. Le Gall, [Random Trees, Lévy Processes and Spatial Branching Processes]([https://link-url-here.org](https://www.imo.universite-paris-saclay.fr/~jean-francois.le-gall/Mono-revised.pdf)). A related notion is that of Boltzmann samplers, see Pierre Lescanne, [Boltzmann samplers for random generation of lambda terms](https://hal-ens-lyon.archives-ouvertes.fr/ensl-00979074v2). While these are fundamentally equivalent, Boltzmann samplers emphasize the recursive nature of trees, while the encoding process emphasizes the imperative nature of tree traversal.

### How the encoding work
The term "process" refers to a vector of size n that encodes a given tree of size n. Traditionally, random trees are encoded via their "height" process: nodes are visited in a depth-first manner, and the process represent their distance from the root (their "height"). It can be shown that this process totally encodes a given tree. For instance, we can check if a node a is on the path from the root to b by checking if the height process reaches a minimum between the visits of the two.

Another way to encode a tree, the `application_kernel`, is to keep track, for each index `i` of the closest node `application_kernel[i]` of size 2 (e.g., applications) for which `i` is in the subterm on the function side of the application. This process also totally encodes the tree. We can recover the height process by "integrating" the vector (np.ones(n)) against this kernel  where integrating means computing the following function, which is at the heart of this project:
```
# the kind of each node (-1 for variables, 0 for abstractions and 1 for applications) can be easily computed from application_kernel
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
This `forward_integral` method iteratively compute the cumulative sum of `v` along the ancestors of a given node. For convenience, we also define the `abstraction_kernel` and the `variable_kernel`, even if our `application_kernel` totally encode the tree. Integrating diverse vectors against diverse kernels is a fundamental way of interacting with lambdaforge as illustrated in the following example.

### Resolving bindings by integration process
If `abs` is an indicator vector of abstractions, integrating `abs` gives us `habs`, a process representing the number of abstraction under each node. Note that `variable_kernel` gives for each index `a`, the rightmost variable in the subterm at `a`.

To know if a variable at index `v` is binded to an abstraction at index `a`, we only need to check  that 
- `a <= v <= variable_kernel[a]` which means that `v` is in the subterm at index `a`
- `de_bruijn_indices[v] == habs[v]-habs[a]`

## License

This project is licensed under the MIT License.
