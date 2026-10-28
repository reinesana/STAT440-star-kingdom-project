import unittest
import numpy as np
from shrink_after_work import generate,trace,Problem
from optimize import Grid

class Toy:
    unit=Problem.unit
    cost=Problem.cost
    def __init__(self):
        self.p=[0,1];self.start=np.zeros(2,bool)
        self.a=np.array([[-1.,0.],[9.,0.]]);self.b=np.array([[1.,0.],[11.,0.]])
        self.mid=(self.a+self.b)/2;self.grid=Grid(self.mid)
        self.length=np.array([2.,2.]);self.surface=np.array(['grassland','grassland'])

class ShrinkTests(unittest.TestCase):
    def test_shrink_after_previous_pipe_replaced(self):
        pr=Toy();units=[pr.unit([0]),pr.unit([0,1])]
        candidate,audit,_=generate(pr,units)
        self.assertEqual(len(candidate),2)
        self.assertEqual(len(audit),1)
        self.assertLess(candidate[1][1],units[1][1])
        self.assertEqual(set(candidate[1][2]),{1})
        self.assertLess(trace(pr,candidate)[3],trace(pr,units)[3])
        np.testing.assert_array_equal(trace(pr,candidate)[1],trace(pr,units)[1])

    def test_initial_polyurethane_coverage_not_silently_removed(self):
        pr=Toy();pr.start[0]=True;units=[pr.unit([0,1])]
        candidate,audit,_=generate(pr,units)
        self.assertEqual(audit,[])
        self.assertEqual(set(candidate[0][2]),{0,1})

if __name__=='__main__':unittest.main()
