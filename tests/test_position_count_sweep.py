import numpy as np
from scripts.position_count_sweep import score


def test_position_matching_ignores_marks_but_is_one_to_one():
    gt=np.array([[0.,0.,-100.,-100.]])
    pred=np.array([[.1,0.,100.,100.],[.2,0.,200.,200.]])
    result=score(pred,gt)
    assert abs(result['precision']-.5)<1e-6
    assert abs(result['recall']-1.)<1e-6
    assert abs(result['f1']-2/3)<1e-6
    pred[:,2:]=-300
    assert score(pred,gt)==result


def test_position_gate_excludes_distant_returns():
    result=score(np.array([[2.,0.,0.,0.]]),np.array([[0.,0.,0.,0.]]))
    assert result==dict(precision=0.,recall=0.,f1=0.)
