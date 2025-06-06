import numpy as np



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