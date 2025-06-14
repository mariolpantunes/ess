import logging
import argparse
import numpy as np
import ess.ess as ess
import ess.legacy as legacy
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

plt.set_loglevel("error")
logging.getLogger('PIL').setLevel(logging.WARNING)


def main(args):
    points0 = np.array([[0,0], [5,5], [5,0], [0,5], [2.5,2.5]])
    bounds = np.array([[0,5], [0,5]])
    points1 = legacy._esa_01(points0, bounds, n=args.n)
    points2 = legacy._esa_02(points0, bounds, n=args.n)
    points3 = ess.esa(points0, bounds, n=args.n)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

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
    args = parser.parse_args()
    
    main(args)
