import time
import logging
import argparse
import numpy as np
import ess.ess as ess
import ess.utils as utils
import ess.legacy as legacy
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

plt.set_loglevel("error")
logging.getLogger('PIL').setLevel(logging.WARNING)


def main(args):
    points0 = [[0,0,0], [5,5,5], [5,0,0], [0,5,0], [0,0,5], [0,5,5], [5,0,5], [5,5,0], [2,2,2]]
    bounds = np.array([[0,5], [0,5], [0,5]])

    s = time.process_time()
    points1 = legacy._esa_01(points0, bounds, n=args.n)
    e = time.process_time()
    all_points = np.concatenate((points0, points1), axis=0)
    coverage = utils.calculate_grid_coverage(all_points, bounds=bounds, grid=args.g)
    logger.info(f'ESA01 ({e-s:.3f} seconds) coverage {coverage:.2f}')


    s = time.process_time()
    points2 = legacy._esa_02(points0, bounds, n=args.n)
    e = time.process_time()
    all_points = np.concatenate((points0, points2), axis=0)
    coverage = utils.calculate_grid_coverage(all_points, bounds=bounds, grid=args.g)
    logger.info(f'ESA02 ({e-s:.3f} seconds) coverage {coverage:.2f}')

    s = time.process_time()
    points3 = ess.esa(points0, bounds, n=args.n)
    e = time.process_time()
    all_points = np.concatenate((points0, points3), axis=0)
    coverage = utils.calculate_grid_coverage(all_points, bounds=bounds, grid=args.g)
    logger.info(f'ESA03 ({e-s:.3f} seconds) coverage {coverage:.2f}')


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))

    ax1.scatter(*zip(*points0))
    ax1.scatter(*zip(*points1))
    ax2.scatter(*zip(*points0))
    ax2.scatter(*zip(*points2))
    ax3.scatter(*zip(*points0))
    ax3.scatter(*zip(*points3))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample for 2D ESA')
    parser.add_argument('-n', type=int, help='number of points', default=10)
    parser.add_argument('-g', type=int, help='number of points in grid', default=7)
    args = parser.parse_args()

    main(args)
