import numpy as np

import util


class TestUtil(object):
    def test_safelog(self):
        assert util.safelog(5) == np.log(5)
        assert util.safelog(-5) is None
        assert len(util.safelog(np.array([1, 2, 3]))) == 3
