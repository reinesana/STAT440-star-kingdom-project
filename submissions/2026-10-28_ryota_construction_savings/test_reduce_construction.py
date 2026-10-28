import unittest
import numpy as np
from reduce_construction import generate,trace,Problem
from optimize import Grid

class Toy:
    unit=Problem.unit
    cost=Problem.cost
    def __init__(self):
        self.p=[0,1];self.start=np.zeros(2,bool)
        self.a=np.array([[0.,0.],[0.,.1]]);self.b=np.array([[2.,0.],[2.,.1]])
        self.mid=(self.a+self.b)/2;self.grid=Grid(self.mid)
        self.length=np.array([2.,2.]);self.surface=np.array(['grassland','grassland'])

class ConstructionTests(unittest.TestCase):
    def test_parallel_pipes_merge_preserves_coverage_and_saves(self):
        pr=Toy();units=[pr.unit([i]) for i in range(2)]
        candidate,audit,search=generate(pr,units,lambda y:np.zeros(2))
        self.assertEqual(len(candidate),1)
        self.assertEqual(set(candidate[0][2]),{0,1})
        self.assertEqual(len(audit),1)
        self.assertLess(trace(pr,candidate)[3],trace(pr,units)[3])
        np.testing.assert_array_less(trace(pr,candidate)[1],trace(pr,units)[1]+1)

    def test_different_planned_years_not_merged(self):
        class Expensive(Toy):
            def cost(self,u,state):return 1_200_000_000
        pr=Expensive();units=[pr.unit([i]) for i in range(2)]
        candidate,audit,_=generate(pr,units,lambda y:np.zeros(2))
        self.assertEqual(len(candidate),2)
        self.assertEqual(audit,[])

if __name__=='__main__':unittest.main()
