import hnswlib
import numpy as np

def _scale(arr, min_val=None, max_val=None):
    if min_val is None:
        min_val = min(arr)
    
    if max_val is None:
        max_val = max(arr)

    scl_arr = (arr - min_val) / (max_val - min_val)
    return scl_arr, min_val, max_val


def _inv_scale(scl_arr, min_val, max_val):
    return scl_arr*(max_val - min_val) + min_val


def _empty_center(coor, data, neigh, movestep, iternum, bounds=np.array([[-1, 1]])):
    """
    Empty center search process.
    """
    
    es_configs = []
    for i in range(iternum):
        adjs_,  distances_= adjs_, distances_ = neigh.knn_query(coor, k=1)
        
        direc = _elastic(coor, data[adjs_[0]], distances_[0])
        mag = np.linalg.norm(direc)
        if mag < 1e-7:
            break
        direc /= mag
        coor += direc * movestep

        if (coor < bounds[:, 0]).any() or (coor > bounds[:, 1]).any():
            np.clip(coor, bounds[:, 0], bounds[:, 1], out=coor)
            es_configs.extend(coor.tolist())
            break

    return coor


def _force(sigma, d):
    """
    Optimized Force function.
    """
    ratio = sigma / d  # Reuse this computation
    # TODO: check this clip
    ratio = np.clip(ratio, a_min=None, a_max=3.1622)  # Avoids overflow
    attrac = ratio ** 6
    # TODO: check this clip
    attrac = np.clip(attrac, a_min=None, a_max=1000)  # Avoids overflow
    
    return 6 * (2 * attrac ** 2 - attrac) / d


def _elastic(es, neighbors, neighbors_dist):
    """
    Optimized Elastic force with vectorization.
    """
    sigma = np.mean(neighbors_dist) / 2
    neighbors_dist = np.clip(neighbors_dist, a_min=0.001, a_max=None)  # Avoids distances < 0.001

    # Vectorized force computation
    forces = _force(sigma, neighbors_dist)

    # Vectorized displacement computation
    vecs = (es - neighbors) / neighbors_dist[:, np.newaxis]
    
    # Compute the directional force
    direc = np.sum(vecs * forces[:, np.newaxis], axis=0)
    
    return direc


def esa(samples, bounds, n:int=None, seed:int=None):
    '''
    apply esa in the experiment
    '''
    min_val = bounds[:,0]
    max_val = bounds[:,1]
    samples, _, _ = _scale(samples, min_val, max_val)

    coors = np.random.uniform(0, 1, (n, samples.shape[1]))
    
    neigh = hnswlib.Index(space='l2', dim=samples.shape[1])
    if seed is not None:
        neigh.init_index(max_elements=len(samples), ef_construction = 200, M=48, random_seed = seed)
    else:
        neigh.init_index(max_elements=len(samples), ef_construction = 200, M=48)
    neigh.add_items(samples)
    
    es_params = [_empty_center(coor.reshape(1, -1), samples, neigh, 
    movestep=0.01, iternum=100, bounds=np.array([[0, 1]]))[0] for coor in coors]

    rv = np.array(es_params)[:n] 
    rv = _inv_scale(rv, min_val=min_val, max_val=max_val)
    
    return rv


def ess(samples, bounds, n:int=None, seed:int=None):
    if type(samples) is not np.ndarray:
        samples = np.array(samples)
    rv = esa(samples=samples, bounds=bounds, n=n, seed=seed)
    return np.concatenate((samples, rv), axis=0)


points = [[1,1], [3,3], [5,3], [3,5]]

points = ess(points, np.array([[0,5],[0,5]]), 5)

print(f'{points}')