import unittest
import numpy as np
from early_three_years import benefits, early_schedule

class EarlyTests(unittest.TestCase):
    def test_three_year_probability_uses_survival(self):
        value=benefits(lambda year:np.array([.1,0.]),np.array([100.,200.]),2027,2029)
        np.testing.assert_allclose(value,[27.1,0.])

    def test_risk_efficient_group_and_isolated_pipe(self):
        class Tiny:
            p=[0,1,2];start=np.zeros(3,bool);iron=np.ones(3,bool)
            def cost(self,u,state):return int(u[3]*100)
        def unit(ids,cost,radius):return (np.zeros(2),radius,np.array(ids),cost)
        singles={0:unit([0],6_000_000,1),1:unit([1],6_000_000,1),2:unit([2],6_000_000,1)}
        combined=unit([0,1],7_000_000,2)
        units,audit,n=early_schedule(Tiny(),singles,[(500_000_000,(0,1),combined)],
            lambda year:np.array([.9,.9,.01]),np.full(3,100.))
        self.assertEqual(set(units[0][2]),{0,1})
        self.assertEqual(set(units[1][2]),{2})
        self.assertEqual(len(units),2)
        self.assertEqual(n,2)
        self.assertEqual([r['planned_year'] for r in audit],[2027,2028])

if __name__=='__main__':unittest.main()
