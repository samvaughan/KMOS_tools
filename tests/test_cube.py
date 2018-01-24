import unittest
from ..cube_tools import Cube
import numpy as np 

class TestCubeFunctions(unittest.TestCase):
   
    """Testing the Cube_tools Class"""
    def setUp(self):
        self.cube=Cube('/Data/KCLASH/Data/Sci/FINAL/COMBINE_SCI_RECONSTRUCTED_MACS1931_BCG_59407.fits')

    def test_arcesond_interpolation_shape(self):

        """Test to see if the output cube from interpolate_point_1_arcsec_sampling is exactly double the size of the original data"""

        old_shape=(self.cube.nx, self.cube.ny)
        new_cube, new_noise=self.cube.interpolate_point_1_arcsec_sampling()

        new_shape=(cube.nx, cube.ny)

        self.assertEqual(((2*new_shape[0], 2*new_shape[1]), old_shape, "Cube dimensions haven't doubled"))

    def test_arcesond_interpolation_flux_sum(self):

        """Test to check that the flux in consereved after interpolation"""
        old_data=self.cube.data.copy()

        new_cube, new_noise=self.cube.interpolate_point_1_arcsec_sampling()

        self.assertEqual(np.nansum(np.nansum(old_data, axis=2), axis=1), np.nansum(np.nansum(new_cube, axis=2), axis=1), "Sum of the flux in the old cube and interpolated cube is different")

    def test_arcesond_interpolation_noise_sum(self):

        """Test to check that the noise in consereved after interpolation"""
        old_noise=self.cube.noise.copy()

        new_cube, new_noise=self.cube.interpolate_point_1_arcsec_sampling()

        self.assertEqual(np.nansum(np.nansum(old_noise, axis=2), axis=1), np.nansum(np.nansum(new_cube, axis=2), axis=1), "Sum of the noise in the old cube and interpolated cube is different")

    
    def test_collapse_cube_flag(self):
        """Test to see that the collapse flag has been set"""

        old_flag=self.cube.has_been_collapsed
        self.cube.collapse()
        new_flag=self.cube.has_been_collapsed

        self.assertNotEqual(old_flag, new_flag, "Flag hasn't changed after calling cube.collapse")

    def test_collapse_cube_data_shape(self):

        """The collapsed Cube should be 2d"""

        self.cube.collapse()
        self.assertEqual(len(self.cube.collapse.shape), 2, "Collapsed Cube isn't 2D")




class TestCubeStaticMethods(unittest.TestCase):
    def setUp(self):
        pass

    def test_fits_file_exists(self):

        """Test to check that the fits file with all the data in exists"""

        result=Cube.get_KMOS_fits_data('MACS1931_BCG_59407')
        self.assertIsNotNone(result)




if __name__ == '__main__':

    suite_1 = unittest.TestLoader().loadTestsFromTestCase(TestCubeFunctions)
    suite_2 = unittest.TestLoader().loadTestsFromTestCase(TestCubeStaticMethods)

    alltests = unittest.TestSuite([suite_1, suite_2])
    unittest.TextTestRunner().run(alltests)
