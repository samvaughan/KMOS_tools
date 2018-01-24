import unittest
from ..kinematics import Cube

class TestKinematics(unittest.TestCase):
   
   """Testing the Kinematics Class"""


    def setUp(self):
        self.kins=CubeKinematics('/Data/KCLASH/Data/Sci/FINAL/COMBINE_SCI_RECONSTRUCTED_MACS1931_BCG_59407.fits')


    def test_load_gas_templates(self, kins):
        """load_gas_templates function"""

        gas_templates, line_names, line_wave, lamRange_template = self.kins.load_gas_templates([5000, 8000], velscale=50.0, FWHM_gal=2.0)
        #Write tests with appropriate numbers

    def test_voronoi_binning_output_shapes(self, kins)

        x, y, bins, nPixels=self.kins.voronoi_bin_cube(SN_TARGET=5.0, save=False)

        self.assertTrue(x.shape==y.shape==bins.shape)

    




if __name__ == '__main__':
    unittest.main()
