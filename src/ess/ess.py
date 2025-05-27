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
    #TODO: check this one
    direc = np.sum(vecs * forces[:, np.newaxis], axis=0)

    return direc


def _empty_center(coor, data, neigh, movestep, iternum:int=100, bounds=np.array([[-1, 1]])):
    """
    Empty center search process.
    """
    
    for i in range(iternum):
        adjs_, distances_ = neigh.knn_query(coor, k=data.shape[1]+1)

        logger.debug(f'Empty Centers {adjs_} {distances_}')
        
        direc = _elastic(coor, data[adjs_[0]], distances_[0])
        mag = np.linalg.norm(direc)
        if mag < 1e-7:
            break
        direc /= mag
        coor += direc * movestep

        # TODO (4): should the bounds be fixed to [0, 1]?
        # may help code here 
        if (coor < bounds[:, 0]).any() or (coor > bounds[:, 1]).any():
            np.clip(coor, bounds[:, 0], bounds[:, 1], out=coor)
            break

    return coor


def _esa_01(samples, bounds, n:int=None, seed:int=None):
    '''
    apply esa in the experiment
    '''
    min_val = bounds[:,0]
    max_val = bounds[:,1]
    samples, _, _ = _scale(samples, min_val, max_val)

    neigh = hnswlib.Index(space='l2', dim=samples.shape[1])
    if seed is not None:
        neigh.init_index(max_elements=len(samples)+n, ef_construction = 200, M=48, 
        random_seed = seed)
    else:
        neigh.init_index(max_elements=len(samples)+n, ef_construction = 200, M=48)
    neigh.add_items(samples)
    
    #TODO (2): improve by adding one point at a time (avoiding clustering points together) 
    coors = np.random.uniform(0, 1, (n, samples.shape[1]))
    logger.debug(f'Coors({n}, {samples.shape[1]})\n{coors}')
    es_params = []
    logger.debug(f'Samples\n{samples}')
    es_params = [_empty_center(coor.reshape(1, -1), samples, neigh, 
    movestep=0.01, iternum=100, bounds=np.array([[0, 1]]))[0] for coor in coors]
    logger.debug(f'Params({len(es_params)})\n{es_params}')
    #rv = np.array(es_params)[:n]
    rv = np.array(es_params)
    rv = _inv_scale(rv, min_val=min_val, max_val=max_val)
    
    logger.debug(f'RV({rv.shape})\n{rv}')

    return rv


def _esa_02(samples, bounds, n:int=None, seed:int=None):
    '''
    apply esa in the experiment
    '''
    min_val = bounds[:,0]
    max_val = bounds[:,1]
    samples, _, _ = _scale(samples, min_val, max_val)

    neigh = hnswlib.Index(space='l2', dim=samples.shape[1])
    if seed is not None:
        neigh.init_index(max_elements=len(samples)+n, ef_construction = 200, M=48, 
        random_seed = seed)
    else:
        neigh.init_index(max_elements=len(samples)+n, ef_construction = 200, M=48)
    neigh.add_items(samples)
    
    #TODO (2): improve by adding one point at a time (avoiding clustering points together) 
    coors = np.random.uniform(0, 1, (n, samples.shape[1]))
    logger.debug(f'Coors({n}, {samples.shape[1]})\n{coors}')
    es_params = []
    logger.debug(f'Samples\n{samples}')
    for c in coors:
        es_param = _empty_center(c.reshape(1, -1), samples, neigh, 
        movestep=0.01, iternum=100, bounds=np.array([[0, 1]]))
        es_params.append(es_param[0])
        samples = np.concatenate((samples, es_param), axis=0)
        #samples = np.append(samples, es_param)
        logger.debug(f'Samples\n{samples}')
        neigh.add_items(es_param)
    #es_params = [_empty_center(coor.reshape(1, -1), samples, neigh, 
    #movestep=0.01, iternum=100, bounds=np.array([[0, 1]]))[0] for coor in coors]
    logger.debug(f'Params({len(es_params)})\n{es_params}')
    #rv = np.array(es_params)[:n]
    rv = np.array(es_params)
    rv = _inv_scale(rv, min_val=min_val, max_val=max_val)
    
    logger.debug(f'RV({rv.shape})\n{rv}')

    return rv


def esa(samples, bounds, n:int=None, seed:int=None):
    '''
    apply esa in the experiment
    '''
    min_val = bounds[:,0]
    max_val = bounds[:,1]
    samples, _, _ = _scale(samples, min_val, max_val)
    samples = samples.astype(np.float32)

    neigh = hnswlib.Index(space='l2', dim=samples.shape[1])
    if seed is not None:
        neigh.init_index(max_elements=len(samples)+n, ef_construction = 200, M=48, 
        random_seed = seed)
    else:
        neigh.init_index(max_elements=len(samples)+n, ef_construction = 200, M=48)
    
    #TODO: apply seed number here
    coors = np.random.uniform(0, 1, (n, samples.shape[1])).astype(np.float32)
    # increase the sample pool and keep original size as idx
    idx = len(samples)
    samples = np.concatenate((samples, coors), axis=0)
    neigh.add_items(samples)

    iternum = 100
    movestep=0.01

    for _ in range(iternum):
        for i in range(idx, len(samples)):
            p = samples[i]
            
            adjs_, distances_ = neigh.knn_query(p, k=samples.shape[1]+2)        
            direc = _elastic(p, samples[adjs_[0, 1:]], distances_[0, 1:])
            p += (direc/np.linalg.norm(direc)) * movestep
            
            samples[i] = p
        
        samples = clip(samples, 0, 1)
        neigh = hnswlib.Index(space='l2', dim=samples.shape[1])
        if seed is not None:
            neigh.init_index(max_elements=len(samples)+n, ef_construction = 200, M=48, 
            random_seed = seed)
        else:
            neigh.init_index(max_elements=len(samples)+n, ef_construction = 200, M=48)
        neigh.add_items(samples)
    
    rv = samples[idx:]
    rv = _inv_scale(rv, min_val=min_val, max_val=max_val)
    
    return rv


def ess(samples, bounds, n:int=None, seed:int=None):
    if type(samples) is not np.ndarray:
        samples = np.array(samples).astype(np.float32)
    rv = esa(samples=samples, bounds=bounds, n=n, seed=seed)
    return np.concatenate((samples, rv), axis=0)
