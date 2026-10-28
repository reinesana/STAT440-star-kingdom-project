import unittest
import numpy as np
from merge_three_circles import generate,trace,Problem
from optimize import Grid

class CrossToy:
    unit=Problem.unit
    cost=Problem.cost
    def __init__(self):
        centers=np.array([[0.,0.],[2.,0.],[1.,3**.5]])
        self.a=np.array([c+v for c in centers for v in [(-1.,0.),(0.,-1.)]])
        self.b=np.array([c+v for c in centers for v in [(1.,0.),(0.,1.)]])
        self.p=list(range(6));self.start=np.zeros(6,bool);self.mid=(self.a+self.b)/2
        self.length=np.full(6,2.);self.surface=np.full(6,'structure');self.grid=Grid(self.mid)

class TripleTests(unittest.TestCase):
    def test_three_way_saving_when_no_pair_saves_fixed_cost(self):
        pr=CrossToy();units=[pr.unit([2*i,2*i+1]) for i in range(3)]
        candidate,audit,search=generate(pr,units,lambda y:np.zeros(6))
        self.assertEqual(len(candidate),1)
        self.assertEqual(set(candidate[0][2]),set(range(6)))
        self.assertEqual(search['triple_only_fixed_savings_cases'],1)
        self.assertLess(trace(pr,candidate)[3],trace(pr,units)[3])

    def test_different_years_not_merged(self):
        class Expensive(CrossToy):
            def cost(self,u,state):return 1_200_000_000
        pr=Expensive();units=[pr.unit([2*i,2*i+1]) for i in range(3)]
        candidate,audit,_=generate(pr,units,lambda y:np.zeros(6))
        self.assertEqual(len(candidate),3)
        self.assertEqual(audit,[])

if __name__=='__main__':unittest.main()
