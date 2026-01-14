import argparse
import logging
import time

import matplotlib.pyplot as plt
import numpy as np

import ess.ess as ess
import ess.legacy as legacy
import ess.utils as utils

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

plt.set_loglevel("error")
logging.getLogger("PIL").setLevel(logging.WARNING)


def main(args):
    points0 = np.array([[0, 0], [5, 5], [5, 0], [0, 5], [2.5, 2.5]])
    bounds = np.array([[0, 5], [0, 5]])

    s = time.perf_counter()
    points1 = legacy._esa_01(points0, bounds, n=args.n)
    e = time.perf_counter()
    all_points = np.concatenate((points0, points1), axis=0)
    coverage = utils.calculate_grid_coverage(all_points, bounds=bounds, grid=args.g)
    logger.info(f"ESA01 ({e - s:.3f} seconds) coverage {coverage:.2f}")

    s = time.perf_counter()
    points2 = legacy._esa_02(points0, bounds, n=args.n)
    e = time.perf_counter()
    all_points = np.concatenate((points0, points2), axis=0)
    coverage = utils.calculate_grid_coverage(all_points, bounds=bounds, grid=args.g)
    logger.info(f"ESA02 ({e - s:.3f} seconds) coverage {coverage:.2f}")

    s = time.perf_counter()
    points3 = ess.esa(points0, bounds, n=args.n)
    e = time.perf_counter()
    all_points = np.concatenate((points0, points3), axis=0)
    coverage = utils.calculate_grid_coverage(all_points, bounds=bounds, grid=args.g)
    logger.info(f"ESA03 ({e - s:.3f} seconds) coverage {coverage:.2f}")

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.scatter(points0[:, 0], points0[:, 1])
    ax1.scatter(points1[:, 0], points1[:, 1])
    ax2.scatter(points0[:, 0], points0[:, 1])
    ax2.scatter(points2[:, 0], points2[:, 1])
    ax3.scatter(points0[:, 0], points0[:, 1])
    ax3.scatter(points3[:, 0], points3[:, 1])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample for 2D ESA")
    parser.add_argument("-n", type=int, help="number of points", default=50)
    parser.add_argument("-g", type=int, help="number of points in grid", default=10)
    args = parser.parse_args()

    main(args)
