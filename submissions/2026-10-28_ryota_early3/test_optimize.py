import unittest
from itertools import combinations
import numpy as np
from optimize import enclosing, cents, Grid, execute

class GeometryTests(unittest.TestCase):
    def test_random_disks_against_exhaustive_boundaries(self):
        rng=np.random.default_rng(440)
        for _ in range(40):
            pts=rng.normal(size=(8,2))*100
            possible=[]
            for a,b in combinations(pts,2):
                c=(a+b)/2;possible.append((c,np.linalg.norm(a-c)))
            for a,b,c in combinations(pts,3):
                matrix=2*np.vstack((b-a,c-a))
                if abs(np.linalg.det(matrix))<1e-10:continue
                center=np.linalg.solve(matrix,np.array([b@b-a@a,c@c-a@a]))
                possible.append((center,np.linalg.norm(center-a)))
            best=min(r for c,r in possible if np.all(np.linalg.norm(pts-c,axis=1)<=r+1e-7))
            center,radius=enclosing(pts)
            self.assertAlmostEqual(radius,best+1e-6,places=6)
    def test_known_disks(self):
        for pts,expected in [([[0,0],[2,0]],1),([[0,0],[2,0],[1,3**.5]],2/3**.5),
                              ([[0,0],[2,0],[2,2],[0,2]],2**.5),([[0,0],[1,0],[3,0]],1.5),
                              ([[1,1],[1,1]],0)]:
            c,r=enclosing(pts)
            self.assertAlmostEqual(r,expected+1e-6,places=7)
            self.assertTrue(np.all(np.linalg.norm(np.array(pts)-c,axis=1)<=r))
    def test_grid_matches_bruteforce(self):
        rng=np.random.default_rng(440);xy=rng.uniform(-200,200,(1000,2));grid=Grid(xy)
        for _ in range(30):
            c=rng.uniform(-200,200,2);r=rng.uniform(0,300)
            self.assertEqual(set(grid.query(c,r)),set(np.flatnonzero(np.linalg.norm(xy-c,axis=1)<=r)))
    def test_cent_rounding(self):
        self.assertEqual(cents(1.005),101)
        self.assertEqual(cents(1.004),100)
    def test_leaks_after_construction_completion_count(self):
        class Tiny:
            p=[0,1];start=np.array([False,False]);iron=np.array([True,False])
            def cost(self,u,state):return 100
        unit=(np.zeros(2),1,np.array([0]),1)
        result=execute(Tiny(),[unit],np.array([0,20]),np.array([10000,20000]))
        self.assertTrue(result['feasible'])
        self.assertEqual(result['leaks'],1)
        self.assertEqual(result['repair'],200)
        self.assertEqual(result['finish'],2027)
    def test_budget_carryover_and_first_unit_blocking(self):
        class Tiny:
            p=[0];start=np.array([False]);iron=np.array([True])
            def cost(self,u,state):return 1_500_000_000
        result=execute(Tiny(),[(None,1,np.array([0]),1)])
        self.assertEqual(result['finish'],2028)
        self.assertEqual(result['construction'],15_000_000)

if __name__=='__main__':unittest.main()
