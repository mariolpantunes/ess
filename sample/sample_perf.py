import logging
import argparse
import numpy as np
import src.ess as ess
import cProfile, pstats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

plt.set_loglevel("error")
logging.getLogger('PIL').setLevel(logging.WARNING)



points = np.array([[0,0], [5,5], 
[5,0], [0,5], [2.5,2.5]])
logger.info(f'START')
points2 = ess.esa(points, np.array([[0,5], [0,5]]), 10)
logger.info(f'ESA')
#with cProfile.Profile() as pr:
#    points3 = ess.esa2(points, np.array([[0,10], [0,10]]), 300)
#    pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
points3 = ess.esa2(points, np.array([[0,5], [0,5]]), 10)
logger.info(f'ESA2')


fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(*zip(*points))
ax1.scatter(*zip(*points2))
ax2.scatter(*zip(*points))
ax2.scatter(*zip(*points3))
plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi Knee evaluation app')
    parser.add_argument('-i', type=str, required=True, help='input file')
    parser.add_argument('-a', help='add even spaced points', action='store_true')
    parser.add_argument('-r', type=float, help='RDP reconstruction threshold', default=0.01)
    parser.add_argument('-t', type=float, help='clustering threshold', default=0.05)
    parser.add_argument('-c', type=float, help='corner threshold', default=0.33)
    parser.add_argument('-o', help='store output (debug)', action='store_true')
    parser.add_argument('-g', help='display output (debug)', action='store_true')
    parser.add_argument('-k', help='Knee ranking method', type=knee_ranking.ClusterRanking, choices=list(knee_ranking.ClusterRanking), default='hull')
    args = parser.parse_args()
    
    main(args)