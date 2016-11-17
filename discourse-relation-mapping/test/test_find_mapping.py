import numpy.testing as npt

from app.find_mapping import Lexconn
import numpy as np
import unittest

class Test_Lexconn (unittest.TestCase):
    def test_numpy(self):
        a = np.arange(15).reshape((3, 5))
        print(a.shape)
        print(a.shape[0])
        print(np.max(a, axis=0))
        print(np.max(a, ))

    def test_mapping_when_one_dc_one_entry(self):
        emission = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float32)
        print(emission)
        entries = np.array([[0, 0], [0, 1], [1, 1], [1, 2]])
        lexconn = Lexconn(emission, entries)
        got_mapping = lexconn.get_mapping()
        print(got_mapping)
        expected_mapping = np.array([[0.4267, 0.5733, 0.0], [0.0, 0.3963, 0.6037]])
        npt.assert_almost_equal(got_mapping, expected_mapping, decimal=4)
        pass

if __name__ == "__main__":
    unittest.main()
