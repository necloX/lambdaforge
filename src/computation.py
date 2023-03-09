import numpy as np
import numba as nb

@nb.njit
def compute_kernel(kind:np.array(np.int64)):
    n = kind.size
    abs_kernel = -np.ones(n, dtype=np.int64)
    app_kernel = -np.ones(n, dtype=np.int64)
    var_kernel = -np.ones(n, dtype=np.int64)
    for i in range(1,n):
        if kind[i-1] == 0:
            app_kernel[i] = app_kernel[i-1]
            abs_kernel[i] = i-1
        elif kind[i-1] == 1:
            app_kernel[i] = i-1
            abs_kernel[i] = abs_kernel[i-1]
        else:
            app_kernel[i] = app_kernel[app_kernel[i-1]]
            abs_kernel[i] = abs_kernel[app_kernel[i-1]]
    for i in range(n-1,-1,-1):
        if kind[i] == 0:
            var_kernel[i] = var_kernel[i+1]
        elif kind[i] == 1:
            var_kernel[i] = var_kernel[var_kernel[i+1]+1]
        else:
            var_kernel[i] = i
    return abs_kernel,app_kernel,var_kernel
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
@nb.njit
def backward_integral(variable_kernel:np.array(np.int64),kind:np.array(np.int64),v:np.array(np.int64)):
    n = variable_kernel.size
    m = np.zeros(n, dtype=np.int64)
    for i in range(n-1,-1,-1):
        if kind[i] == 0:
            m[i] = m[i+1] + v[i]
        elif kind[i] == 1:
            m[i] = m[i+1]+ m[variable_kernel[i+1]+1] + v[i]
        else:
            m[i] = v[i]
    return m

def forward_substitute(kernel1,kernel2,variables):
    if variables.size == 0: return kernel1
    size = kernel1.size + variables.size * (kernel2.size-1)
    kernel = np.zeros(size,int)
    shift = kernel2.size-1
    shifted_variables = variables + np.arange(variables.size) * shift
    p = np.arange(kernel1.size)
    p[p.size-1] = -1
    kernel[0:variables[0]] = kernel1[0:variables[0]]
    kernel[variables[0]:variables[0]+shift+1] = np.where(kernel2==-1,kernel1[variables[0]],kernel2+variables[0])
    for i in range(variables.size-1):
        p[variables[i]:p.size-1] += shift
        kernel[shifted_variables[i]+shift+1:shifted_variables[i+1]] =  p[kernel1[variables[i]+1:variables[i+1]]]
        kernel[shifted_variables[i+1]:shifted_variables[i+1]+shift+1] = np.where(kernel2==-1,p[kernel1[variables[i+1]]],kernel2+shifted_variables[i+1])
    p[variables[variables.size-1]:p.size-1] += shift
    kernel[shifted_variables[variables.size-1]+shift+1:] = p[kernel1[variables[variables.size-1]+1:]]
    return kernel

def backward_substitute(kernel1,kernel2,variables):
    if variables.size == 0: return kernel1
    size = kernel1.size + variables.size * (kernel2.size-1)
    kernel = np.zeros(size,int)
    shift = kernel2.size-1
    shifted_variables = variables + np.arange(variables.size) * shift
    
    p = np.arange(kernel1.size)+variables.size * shift
    
    kernel[shifted_variables[variables.size-1]+shift+1:] = p[kernel1[variables[variables.size-1]+1:]]
    kernel[shifted_variables[variables.size-1]:shifted_variables[variables.size-1]+shift+1] = kernel2 + shifted_variables[variables.size-1]
    for i in range(variables.size-2,-1,-1):
        p[:variables[i+1]] -= shift
        kernel[shifted_variables[i]+shift+1:shifted_variables[i+1]] =  p[kernel1[variables[i]+1:variables[i+1]]]
        kernel[shifted_variables[i]:shifted_variables[i]+shift+1] = kernel2+shifted_variables[i]
    p[:variables[0]] -= shift
    kernel[0:variables[0]] = p[kernel1[0:variables[0]]]
    return kernel
def de_bruijn_substitute  (indices1,indices2,variables,habs1,free_var2):
    if variables.size == 0: return indices1
    size = indices1.size + variables.size * (indices2.size-1)
    indices = np.zeros(size,int)
    shift = indices2.size-1
    shifted_variables = variables + np.arange(variables.size) * shift

    indices[0:variables[0]] = indices1[0:variables[0]]
    indices[variables[0]:variables[0]+shift+1] = indices2+free_var2 * (habs1[variables[0]])
    for i in range(variables.size-1):
        indices[shifted_variables[i]+shift+1:shifted_variables[i+1]] =  indices1[variables[i]+1:variables[i+1]]
        indices[shifted_variables[i+1]:shifted_variables[i+1]+shift+1] = indices2+free_var2*(habs1[variables[i+1]])
    indices[shifted_variables[variables.size-1]+shift+1:] = indices1[variables[variables.size-1]+1:]
    return indices
    
def forward_insert(kernel1,kernel2,start,end):
    shift = kernel2.size-(end-start + 1)
    size = kernel1.size + shift
    kernel = np.zeros(size,int)
    kernel[0:start] = kernel1[0:start]
    kernel[start:start+kernel2.size] = np.where(kernel2==-1,kernel1[start],kernel2+start)
    kernel[start+kernel2.size:] = np.where(kernel1[end+1:]>end, kernel1[end+1:]+shift,kernel1[end+1:])
    return kernel
def backward_insert(kernel1,kernel2,start,end):
    shift = kernel2.size-(end-start + 1)
    size = kernel1.size + shift
    kernel = np.zeros(size,int)
    kernel[0:start] = np.where(kernel1[0:start]>end, kernel1[0:start]+shift, np.where(kernel1[0:start]>=start,kernel2[0]+start, kernel1[0:start]))
    kernel[start:start+kernel2.size] = kernel2+start
    kernel[start+kernel2.size:] = kernel1[end+1:]+shift
    return kernel
def de_bruijn_insert(indices1,indices2,start,end):
    shift = indices2.size-(end-start + 1)
    size = indices1.size + shift
    indices = np.zeros(size,int)
    indices[0:start] = indices1[0:start]
    indices[start:start+indices2.size] = indices2
    indices[start+indices2.size:] = indices1[end+1:]
    return indices
    