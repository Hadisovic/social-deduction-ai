"""Pure math preflight for a future trainer; no games, optimizer or final cases."""
import numpy as np
import pytest
from impostor_training_math import gae


def test_zero_terminal_has_no_bootstrap_or_cross_episode_trace():
    advantages,returns=gae([13.,2.],[3.,1.],[999.,999.],[1.,1.],[True,True],[0.,.6])
    np.testing.assert_allclose(advantages,[10.,1.]);np.testing.assert_allclose(returns,[13.,2.])


def test_positive_durations_match_frozen_legacy_exactly():
    from impostor_research.learning import gae as legacy
    args=([1.,2.,3.],[.1,.2,.3],[.2,.3,0.],[1.,1.,1.],[False,False,True],[.6,3.,.2])
    for a,b in zip(gae(*args,.995),legacy(*args,.995)):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('duration,done',[(-.1,True),(0.,False)])
def test_reject_invalid_or_zero_nonterminal_time(duration,done):
    with pytest.raises(ValueError):gae([1.],[0.],[0.],[1.],[done],[duration])
