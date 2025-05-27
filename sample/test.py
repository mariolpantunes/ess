import logging
import numpy as np
import src.ess as ess
import cProfile, pstats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

plt.set_loglevel("error")
logging.getLogger('PIL').setLevel(logging.WARNING)

### 2D TEST

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


### 3D TEST

points = [[0,0,0], [5,5,5], [5,0,0], [0,5,0], 
[0,0,5], [0,5,5], [5,0,5], [5,5,0], [2,2,2]]
with cProfile.Profile() as pr:
    points2 = ess.esa2(points, np.array([[0,5], [0,5], [0,5]]), 10)
    pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(30)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

ax.scatter(*zip(*points))
ax.scatter(*zip(*points2))
plt.show()
