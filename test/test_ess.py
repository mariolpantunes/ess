# coding: utf-8

__author__ = 'Mário Antunes'
__version__ = '0.1'
__email__ = 'mario.antunes@ua.pt'
__status__ = 'Development'
__license__ = 'MIT'
__copyright__ = '''
Copyright (c) 2021-2023 Stony Brook University
Copyright (c) 2021-2023 The Research Foundation of SUNY
'''


import unittest
import numpy as np
import ess.ess as ess
import ess.utils as utils


class TestESS(unittest.TestCase):

    def test_ess_00(self):
        points = np.array([[0,0], [5,5], [5,0], [0,5], [2.5,2.5]])
        bounds = np.array([[0,5], [0,5]])
        n_points = 100
        grid = 10
        
        rnd_points = np.random.uniform(bounds[0, 0], bounds[0, 1], 
        size=(n_points, points.shape[1]))
        b = np.concatenate((points, rnd_points), axis=0)
        
        a = ess.ess(points, bounds, n=n_points, seed=42)

        coverage_a = utils.calculate_grid_coverage(a, bounds=bounds, grid=grid)
        coverage_b = utils.calculate_grid_coverage(b, bounds=bounds, grid=grid)

        self.assertGreater(coverage_a, coverage_b)
    
    def test_ess_01(self):
        points = np.array([[0,0], [5,5], [5,0], [0,5], [2.5,2.5]])
        bounds = np.array([[0, 100], [0, 100]])
        n_points = 100
        grid = 10
        
        rnd_points = np.random.uniform(bounds[0, 0], bounds[0, 1], 
        size=(n_points, points.shape[1]))
        b = np.concatenate((points, rnd_points), axis=0)
        
        a = ess.ess(points, bounds, n=n_points, seed=42)

        coverage_a = utils.calculate_grid_coverage(a, bounds=bounds, grid=grid)
        coverage_b = utils.calculate_grid_coverage(b, bounds=bounds, grid=grid)

        self.assertGreater(coverage_a, coverage_b)
    
    def test_ess_02(self):
        points = np.array([[0,0,0], [5,5,5], [5,0,0], [0,5,0], 
        [0,0,5], [0,5,5], [5,0,5], [5,5,0], [2,2,2]])
        bounds = np.array([[0, 5], [0, 5], [0, 5]])
        n_points = 500
        grid = 10
        
        rnd_points = np.random.uniform(bounds[0, 0], bounds[0, 1], 
        size=(n_points, points.shape[1]))
        b = np.concatenate((points, rnd_points), axis=0)
        
        a = ess.ess(points, bounds, n=n_points, seed=42)

        coverage_a = utils.calculate_grid_coverage(a, bounds=bounds, grid=grid)
        coverage_b = utils.calculate_grid_coverage(b, bounds=bounds, grid=grid)

        self.assertGreater(coverage_a, coverage_b)

    def test_ess_03(self):
        points = np.array([[0,0,0], [5,5,5], [5,0,0], [0,5,0], 
        [0,0,5], [0,5,5], [5,0,5], [5,5,0], [2,2,2]])
        bounds = np.array([[0, 10], [0, 10], [0, 10]])
        n_points = 500
        grid = 10
        
        rnd_points = np.random.uniform(bounds[0, 0], bounds[0, 1], 
        size=(n_points, points.shape[1]))
        b = np.concatenate((points, rnd_points), axis=0)
        
        a = ess.ess(points, bounds, n=n_points, seed=42)

        coverage_a = utils.calculate_grid_coverage(a, bounds=bounds, grid=grid)
        coverage_b = utils.calculate_grid_coverage(b, bounds=bounds, grid=grid)

        self.assertGreater(coverage_a, coverage_b)
    
    def test_ess_04(self):
        points = np.array([[0,0,0], [5,5,5]])
        bounds = np.array([[0, 5], [0, 5], [0, 5]])
        n_points = 1

        a = ess.esa(points, bounds, n=n_points, seed=42)
    
    def test_ess_05(self):
        points = np.array([[0,0,0], [5,5,5]])
        bounds = np.array([[0, 5], [0, 5], [0, 5]])
        n_points = 10

        a = ess.esa(points, bounds, n=n_points)