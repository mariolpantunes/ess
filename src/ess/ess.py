import math
import numba
import hnswlib
import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def clip(a, vmin, vmax):
    return np.maximum(np.minimum(a, vmax), vmin)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def mean(a):
    return np.divide(np.sum(a), a.shape[0])


def _scale(arr, min_val=None, max_val=None):
    if min_val is None:
        min_val = min(arr)
    
    if max_val is None:
        max_val = max(arr)

    scl_arr = (arr - min_val) / (max_val - min_val)
    return scl_arr, min_val, max_val


def _inv_scale(scl_arr, min_val, max_val):
    return scl_arr*(max_val - min_val) + min_val


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _force(sigma, d):
    """
    Optimized Force function.
    """
    #ratio = sigma / d # Reuse this computation
    #np.clip(ratio, a_min=None, a_max=3.1622, out=ratio)  # Avoids overflow
    #ratio = clip(sigma / d, 0, 3.1622)
    ratio = np.minimum(sigma/d, 3.1622)
    #attrac = ratio ** 6
    #np.clip(attrac, a_min=None, a_max=1000, out=attrac)  # Avoids overflow
    #attrac = clip(attrac, 0, 1000)
    attrac = np.minimum(ratio ** 6, 1000)
    
    return np.abs(6 * (2 * attrac ** 2 - attrac) / d)


def _elastic(es, neighbors, neighbors_dist):
    """
    Optimized Elastic force with vectorization.
    """
    sigma = mean(neighbors_dist) / 5.0
    neighbors_dist = np.maximum(neighbors_dist, 0.001)  # Avoids distances < 0.001

    # Vectorized force computation
    forces = _force(sigma, neighbors_dist)

    # Vectorized displacement computation
    vecs = (es - neighbors) / neighbors_dist[:, np.newaxis]
    
    # Compute the directional force
    direc = np.sum(vecs * forces[:, np.newaxis], axis=0)

    return direc


def esa(samples, bounds, *, n:int=None, epochs:int = 64, lr:float = 0.01, seed:int=None):
    '''
    apply esa in the experiment
    '''
    min_val = bounds[:,0]
    max_val = bounds[:,1]
    samples, _, _ = _scale(samples, min_val, max_val)
    samples = samples.astype(np.float32)

    if n is None:
        n = len(samples)

    dim = samples.shape[1]
    # computed experimentally to get the values 
    # mentioned on the library
    M = int(math.ceil(6*math.log2(dim)))
    max_elements=len(samples)+n

    neigh = hnswlib.Index(space='l2', dim=dim)
    if seed is None:
        neigh.init_index(max_elements=max_elements, ef_construction=200, M=M)
        rng = np.random.default_rng()
    else:
        neigh.init_index(max_elements=max_elements, ef_construction=200, M=M, 
        random_seed = seed)
        rng = np.random.default_rng(seed=seed)
    
    coors = rng.uniform(0, 1, (n, samples.shape[1])).astype(np.float32)
    # increase the sample pool and keep original size as idx
    idx = len(samples)
    samples = np.concatenate((samples, coors), axis=0)
    neigh.add_items(samples)

    for _ in range(epochs):
        for i in range(idx, len(samples)):
            p = samples[i]
            
            adjs_, distances_ = neigh.knn_query(p, k=min(samples.shape[1]+2, max_elements))        
            direc = _elastic(p, samples[adjs_[0, 1:]], distances_[0, 1:])
            p += (direc/np.linalg.norm(direc)) * lr
            
            samples[i] = p
        
        samples = clip(samples, 0, 1)
        neigh = hnswlib.Index(space='l2', dim=dim)
        if seed is not None:
            neigh.init_index(max_elements=max_elements, ef_construction=200, M=M, 
            random_seed = seed)
        else:
            neigh.init_index(max_elements=max_elements, ef_construction=200, M=M)
        neigh.add_items(samples)
    
    rv = samples[idx:]
    rv = _inv_scale(rv, min_val=min_val, max_val=max_val)
    
    return rv


def ess(samples, bounds, *, n:int=None, epochs:int = 64, lr:float = 0.01, seed:int=None):
    if type(samples) is not np.ndarray:
        samples = np.array(samples).astype(np.float32)
    rv = esa(samples=samples, bounds=bounds, n=n, seed=seed)
    return np.concatenate((samples, rv), axis=0)
