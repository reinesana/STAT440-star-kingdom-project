import sys, math, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from build_cost_submissions import validate
import simulate_score as sim
import search_orders as orders

class CurrentRules(unittest.TestCase):
    def test_surface_polyurethane_overlap_and_simulator_agreement(self):
        # Two grass pipes and a polyurethane structure pipe: rate is 3+50,
        # not 3+3+50; only the two live 2m pipes contribute replacement length.
        a=np.array([[-1.,0.],[-1.,1.],[-1.,-1.]])
        b=np.array([[1.,0.],[1.,1.],[1.,-1.]])
        lengths=np.array([2.,2.,2.])
        pipes=pd.DataFrame({'Pipe ID':['a','b','c'], 'Material':['cast iron','copper','polyurethane'],
                            'Surface':['grassland','grassland','structure'],
                            'mat':['cast iron','copper','polyurethane'], 'leak_date':[pd.NaT]*3})
        history=pd.DataFrame({'Pipe ID':pd.Series(dtype=str)})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'units.csv'
            path.write_text('\"(0, 0)\",2\n\"(0, 0)\",2\n')
            rep=validate(path,pipes,history,a,b,(a+b)/2,lengths,np.array([True,False,False]))
        expected=round(100+53*(math.pi*4)**.85+800,2)
        self.assertEqual(rep['construction_cost'],expected)
        self.assertEqual(rep['skipped_units'],1)
        self.assertEqual(rep['cost_model_version'],'2026-10-26-area-exponent')
        inside=[np.arange(3),np.arange(3)]
        result=sim.simulate(inside,np.array([2.,2.]),pipes,lengths,np.zeros((3,len(sim.YEARS))),[np.array([0.])]*3,1)
        self.assertAlmostEqual(result.iloc[0].construction,expected,places=2)
        live,cost=orders.circle_terms(inside,np.array([2.,2.]),pipes,lengths)
        self.assertAlmostEqual(cost[0],100+53*(math.pi*4)**.85+800)
        self.assertNotAlmostEqual(expected,round(100+(53*math.pi*4)**.85+800,2))

if __name__=='__main__':unittest.main()
