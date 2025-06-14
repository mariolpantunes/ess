import logging
import numpy as np
import src.ess as ess
import cProfile, pstats
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

plt.set_loglevel("error")
logging.getLogger('PIL').setLevel(logging.WARNING)

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
