import unittest
from simsopt.field import BiotSavart, coils_via_symmetries
from scipy.special import eval_chebyt
from tools import *

class Testing(unittest.TestCase):
    # @unittest.skip
    def test_get_w7x_coils_desc(self):
        """Confirms that the W7-X coils loaded into DESC produce the same field
        as from SIMSOPT.
        """ 
        desc_coils = get_w7x_coils_desc()
        simsopt_curves, simsopt_currents, _ = get_w7x_data()
        simsopt_coils = coils_via_symmetries(simsopt_curves, simsopt_currents, 5, True)

        test_pts = [[0, 0, 0], [1, 1, 1], [-1, 1, 1], [10, 5, 0]]
        desc_field = desc_coils.compute_magnetic_field(test_pts, basis="xyz")
        sims_field = BiotSavart(simsopt_coils).set_points(np.array(test_pts)).B()

        np.testing.assert_allclose(desc_field, sims_field, atol=1e-8)

    # @unittest.skip
    def test_get_x_lines(self):
        # we've shown visually in the document "II: Real Stellarators.ipynb"
        # that the 'xyz' basis returns the correct x-lines for the W7-X
        # stellarator. therefore, we must only show that the 'rtz' basis
        # is self-consistent with the 'xyz' basis.

        xs, ys, zs = get_x_lines(basis='xyz')
        rs, ts, zs = get_x_lines(basis='rpz')

        np.testing.assert_array_equal(xs, rs * np.cos(ts))
        np.testing.assert_array_equal(ys, rs * np.sin(ts))
        np.testing.assert_array_equal(zs, zs)
        
if __name__ == "__main__":
    unittest.main()