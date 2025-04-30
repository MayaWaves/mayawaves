import io
import shutil
import numpy as np
from unittest import TestCase
from unittest import mock
from mayawaves.utils.catalogutils import Catalog
from mayawaves.utils.catalogutils import Parameter
import os
import datetime as dt
from freezegun import freeze_time


class TestCatalogUtils(TestCase):
    CURR_DIR = os.path.dirname(__file__)

    @classmethod
    def setUpClass(cls) -> None:
        metadata_path = os.path.join(TestCatalogUtils.CURR_DIR, 'resources')
        metadata_file_path = os.path.join(metadata_path, "MAYAmetadata.pkl")
        with mock.patch('os.path.dirname', return_value=metadata_path):
            with freeze_time(dt.datetime.fromtimestamp(os.path.getmtime(metadata_file_path)).strftime('%Y-%m-%d')):
                TestCatalogUtils.catalog = Catalog()
        TestCatalogUtils.output_directory = os.path.join(TestCatalogUtils.CURR_DIR, "resources/test_output")
        if os.path.exists(TestCatalogUtils.output_directory):
            shutil.rmtree(TestCatalogUtils.output_directory)
        os.mkdir(TestCatalogUtils.output_directory)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(TestCatalogUtils.output_directory):
            shutil.rmtree(TestCatalogUtils.output_directory)
            
    def test_nonspinning_simulations(self):
        expected_total = 103
        total = len(TestCatalogUtils.catalog.nonspinning_simulations)
        self.assertEqual(expected_total, total)

    def test_aligned_spin_simulations(self):
        expected_total = 157
        total = len(TestCatalogUtils.catalog.aligned_spin_simulations)
        self.assertEqual(expected_total, total)

    def test_precessing_simulations(self):
        expected_total = 370
        total = len(TestCatalogUtils.catalog.precessing_simulations)
        self.assertEqual(expected_total, total)

        expected_total = 635
        spin_total = len(TestCatalogUtils.catalog.aligned_spin_simulations) \
                     + len(TestCatalogUtils.catalog.nonspinning_simulations) \
                     + len(TestCatalogUtils.catalog.precessing_simulations)
        known_NaNs = 5
        self.assertEqual(expected_total, spin_total + known_NaNs)

    def test__check_input(self):
        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.SYMMETRIC_MASS_RATIO, 0.1))

        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.MASS_RATIO, 2))

        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.MASS_1, 0.9))
        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.MASS_2, 0.8))

        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.IRREDUCIBLE_MASS_1, 0.2))
        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.IRREDUCIBLE_MASS_2, 0.65))

        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_X_1, 0.3))
        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_Y_1, 0.7))
        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_X_1, -0.5))

        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_X_2, 0.8))
        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_Y_2, -0.3))
        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_X_2, +0.6))

        self.assertTrue(TestCatalogUtils.catalog._check_input(Parameter.F_LOWER_AT_1MSUN, 10))

        # test some invalid input for each possible parameter
        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.SYMMETRIC_MASS_RATIO, 0.3))

        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.MASS_RATIO, 1 / 2))

        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.MASS_1, 1.2))
        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.MASS_2, 2.1))

        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.IRREDUCIBLE_MASS_1, 3))
        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.IRREDUCIBLE_MASS_2, 5))

        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_X_1, 1.1))
        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_Y_1, 1.01))
        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_X_1, -1.001))

        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_X_2, 9))
        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_Y_2, 3))
        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.DIMENSIONLESS_SPIN_X_2, 5))

        self.assertFalse(TestCatalogUtils.catalog._check_input(Parameter.F_LOWER_AT_1MSUN, -20))

    def test__print_parameters(self):
        with mock.patch('sys.stdout', new_callable=io.StringIO) as stdout_mock:
            expected_output = ("Catalog id: GT0002\n"
                               "name                  fr_b3.1_a0.4_oth.045_M77\n"
                               "m1                    0.5001\n"
                               "m2                    0.4999\n"
                               "m1_irr                0.4895\n"
                               "m2_irr                0.4894\n"
                               "q                     1.0002\n"
                               "eta                   0.2500\n"
                               "a1                    0.3999\n"
                               "a1x                   0.2828\n"
                               "a1y                   0.0000\n"
                               "a1z                   0.2828\n"
                               "a2                    0.4001\n"
                               "a2x                   -0.4001\n"
                               "a2y                   -0.0000\n"
                               "a2z                   0.0000\n"
                               "chi_eff               0.1414\n"
                               "chi_p                 0.4000\n"
                               "f_lower_at_1MSUN      3741.0305\n"
                               "separation            6.2000\n"
                               "eccentricity          0.0322\n"
                               "mean_anomaly          -1.0000\n"
                               "merge_time            191.2377\n"
                               "maya file size (GB)   0.0338\n"
                               "lvcnr file size (GB)  0.0014\n")
            TestCatalogUtils.catalog._print_parameters('GT0002')
            actual_output = stdout_mock.getvalue()
            self.assertEqual(expected_output, actual_output)

    def test_spin_magnitudes_for_simulation(self):
        actual_spin = 0.4
        expected_spin = TestCatalogUtils.catalog.spin_magnitudes_for_simulation("GT0001")
        self.assertTrue(np.isclose(actual_spin, expected_spin[0], atol=0.01))

        # more test cases such as a precessing one
        actual_spin = 0.6
        expected_spin = TestCatalogUtils.catalog.spin_magnitudes_for_simulation("GT0493")
        self.assertTrue(np.isclose(actual_spin, expected_spin[0], atol=0.01))

    def test_get_simulations_with_parameters(self):
        m1_result = TestCatalogUtils.catalog.get_simulations_with_parameters(params=[Parameter.MASS_1], values=[0.5],
                                                                             tol=[0.01])
        actual_value = 184
        self.assertEqual(len(m1_result), actual_value)

        m1_a1_result = TestCatalogUtils.catalog.get_simulations_with_parameters(
            params=[Parameter.MASS_1, Parameter.DIMENSIONLESS_SPIN_1], values=[0.5, 0.5], tol=[0.01, 0.1])
        actual_value = 24
        self.assertEqual(len(m1_a1_result), actual_value)

        print(TestCatalogUtils.catalog.get_parameters_for_simulation('MAYA1005'))

        q7_simulations = TestCatalogUtils.catalog.get_simulations_with_parameters(params=[Parameter.MASS_RATIO],
                                                                                  values=[7], tol=[0.1])
        expected_q7_simulations = ['GT0688', 'GT0740', 'GT0742', 'GT0818', 'GT0834', 'GT0888', 'MAYA1033', 'MAYA1034',
                                   'MAYA1035', 'MAYA1036', 'MAYA1037', 'MAYA1038', 'MAYA1039']
        self.assertEqual(set(expected_q7_simulations), set(q7_simulations))

        q7_simulations = TestCatalogUtils.catalog.get_simulations_with_parameters(
            params=[Parameter.MASS_RATIO, Parameter.DIMENSIONLESS_SPIN_2], values=[7, 0], tol=[0.1, 0.05])
        expected_q7_simulations = ['GT0688', 'GT0740', 'GT0742', 'GT0818', 'GT0834', 'MAYA1039']
        self.assertEqual(set(expected_q7_simulations), set(q7_simulations))

    def test_get_simulations_with_mass_ratio(self):
        # This method should be able to take q <1 and give the same result as 1/q
        expected_q2 = TestCatalogUtils.catalog.get_simulations_with_mass_ratio(2)
        q_one_half_result = TestCatalogUtils.catalog.get_simulations_with_mass_ratio(0.5)
        self.assertEqual(expected_q2, q_one_half_result)

        # The method should be able to accept negative tolerances and give the same result as positive tolerances.
        positive_tolerance_result = TestCatalogUtils.catalog.get_simulations_with_mass_ratio(2, tol=0.01)
        negative_tolerance_result = TestCatalogUtils.catalog.get_simulations_with_mass_ratio(2, tol=-0.01)
        self.assertEqual(positive_tolerance_result, negative_tolerance_result)

        # Compare to get_simulations_with_parameters
        get_simulations_with_mass_ratio_result = TestCatalogUtils.catalog.get_simulations_with_mass_ratio(2, tol=0.01)
        get_params_result = TestCatalogUtils.catalog.get_simulations_with_parameters([Parameter.MASS_RATIO], [2],
                                                                                     [0.01])
        self.assertEqual(get_simulations_with_mass_ratio_result, get_params_result)

        expected_q2 = 34
        self.assertEqual(expected_q2, len(TestCatalogUtils.catalog.get_simulations_with_mass_ratio(2, tol=0.00001)))

        expected_q10 = ['GT0568']
        self.assertEqual(set(expected_q10), set(TestCatalogUtils.catalog.get_simulations_with_mass_ratio(10, tol=0.1)))

        expected_q1_point_5 = ['GT0411', 'GT0455', 'GT0456', 'GT0457', 'GT0477', 'GT0479', 'GT0543', 'GT0558', 'GT0567',
                               'GT0764', 'GT0835', 'GT0836', 'GT0839', 'GT0847', 'GT0848', 'GT0858', 'GT0870', 'GT0871',
                               'GT0873', 'MAYA0907']
        self.assertEqual(set(expected_q1_point_5),
                         set(TestCatalogUtils.catalog.get_simulations_with_mass_ratio(1.5, tol=0.01)))

    def test_get_simulations_with_symmetric_mass_ratio(self):
        # Compare to get_simulations_with_parameters
        get_simulations_with_symmetric_mass_ratio_result = TestCatalogUtils.catalog.get_simulations_with_symmetric_mass_ratio(
            0.25, tol=0.01)
        get_params_result = TestCatalogUtils.catalog.get_simulations_with_parameters([Parameter.SYMMETRIC_MASS_RATIO],
                                                                                     [0.25], [0.01])
        self.assertEqual(get_simulations_with_symmetric_mass_ratio_result, get_params_result)

        # actually count up how many simulations you expect for eta=0.25 within given tolerance
        eta_result = 184
        get_simulations_with_symmetric_mass_ratio_result = TestCatalogUtils.catalog.get_simulations_with_symmetric_mass_ratio(
            0.25, tol=0.0001)
        self.assertEqual(len(get_simulations_with_symmetric_mass_ratio_result), eta_result)

        eta_result = 3
        get_simulations_with_symmetric_mass_ratio_result = TestCatalogUtils.catalog.get_simulations_with_symmetric_mass_ratio(
            0.22, tol=0.002)
        self.assertEqual(len(get_simulations_with_symmetric_mass_ratio_result), eta_result)

    def test_get_parameters_for_simulation(self):
        # test for a couple different simulations
        test_simulation = TestCatalogUtils.catalog.get_parameters_for_simulation("GT0001")
        self.assertEqual(test_simulation[Parameter.NAME], 'fr_b3.1_a0.4_oth.000_M77')
        self.assertAlmostEqual(test_simulation[Parameter.MASS_1], 0.5, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MASS_2], 0.4999, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.IRREDUCIBLE_MASS_1], 0.4895, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.IRREDUCIBLE_MASS_2], 0.4895, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MASS_RATIO], 1.0002, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.SYMMETRIC_MASS_RATIO], 0.25, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_1], 0.3999, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_X_1], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Y_1], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Z_1], 0.3999, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_1], 0.4, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_X_2], -0.4, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Y_2], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Z_2], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.CHI_EFF], 0.2, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.CHI_P], 0.4, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.F_LOWER_AT_1MSUN], 3689.612, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.ECCENTRICITY], 0.0349, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MEAN_ANOMALY], -1, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MERGE_TIME], 202.1975, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MAYA_SIZE_GB], 0.0338, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.LVCNR_SIZE_GB], 0.00139, places=3)


        test_simulation = TestCatalogUtils.catalog.get_parameters_for_simulation("GT0533")
        self.assertEqual(test_simulation[Parameter.NAME], 'SS_D6.2_a0.6_th210_M103')
        self.assertAlmostEqual(test_simulation[Parameter.MASS_1], 0.4993, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MASS_2], 0.4993, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.IRREDUCIBLE_MASS_1], 0.4736, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.IRREDUCIBLE_MASS_2], 0.4736, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MASS_RATIO], 0.9999, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.SYMMETRIC_MASS_RATIO], 0.25, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_1], 0.6012, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_X_1], -0.3006, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Y_1], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Z_1], -0.5207, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_1], 0.6012, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_X_2], 0.3006, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Y_2], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Z_2], -0.5207, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.CHI_EFF], -0.5206, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.CHI_P], 0.3006, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.F_LOWER_AT_1MSUN], 4990.0715, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.ECCENTRICITY], 0.0342, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MEAN_ANOMALY], -1, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MERGE_TIME], 115.2473, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MAYA_SIZE_GB], 0.1373, places=3)
        self.assertTrue(np.isnan(test_simulation[Parameter.LVCNR_SIZE_GB]))

        test_simulation = TestCatalogUtils.catalog.get_parameters_for_simulation("MAYA0978")
        self.assertEqual(test_simulation[Parameter.NAME], 'D11_q3_a1_0_0_0_a2_0_0_0_m240_e0.08')
        self.assertAlmostEqual(test_simulation[Parameter.MASS_1], 0.7499, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MASS_2], 0.2499, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.IRREDUCIBLE_MASS_1], 0.7499, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.IRREDUCIBLE_MASS_2], 0.2499, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MASS_RATIO], 3.0000, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.SYMMETRIC_MASS_RATIO], 0.1875, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_1], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_X_1], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Y_1], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Z_1], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_1], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_X_2], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Y_2], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.DIMENSIONLESS_SPIN_Z_2], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.CHI_EFF], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.CHI_P], 0, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.F_LOWER_AT_1MSUN], 1826.359, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.ECCENTRICITY], 0.1020, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MEAN_ANOMALY], 4.2408, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MERGE_TIME], 958.679, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.MAYA_SIZE_GB], 0.619, places=3)
        self.assertAlmostEqual(test_simulation[Parameter.LVCNR_SIZE_GB], 0.0051, places=3)

    @mock.patch("matplotlib.pyplot.show")
    def test_plot_catalog_parameters(self, mock_show):
        # provide a path
        TestCatalogUtils.catalog.plot_catalog_parameters(Parameter.MASS_RATIO, Parameter.DIMENSIONLESS_SPIN_2,
                                                         save_path=TestCatalogUtils.output_directory)
        self.assertTrue(os.path.exists(os.path.join(TestCatalogUtils.output_directory, "MAYA_q_a2_plot.png")))
        self.assertTrue(os.path.exists(os.path.join(TestCatalogUtils.output_directory, "MAYA_q_hist.png")))
        self.assertTrue(os.path.exists(os.path.join(TestCatalogUtils.output_directory, "MAYA_a2_hist.png")))
        os.remove(os.path.join(TestCatalogUtils.output_directory, "MAYA_q_a2_plot.png"))
        os.remove(os.path.join(TestCatalogUtils.output_directory, "MAYA_q_hist.png"))
        os.remove(os.path.join(TestCatalogUtils.output_directory, "MAYA_a2_hist.png"))
        self.assertEqual(0, mock_show.call_count)
        mock_show.reset_mock()

        # provide an invalid path
        TestCatalogUtils.catalog.plot_catalog_parameters(Parameter.MASS_RATIO, Parameter.DIMENSIONLESS_SPIN_2,
                                                         save_path=TestCatalogUtils.output_directory + 'invalid')
        self.assertFalse(
            os.path.exists(os.path.join(TestCatalogUtils.output_directory + 'invalid', "MAYA_q_a2_plot.png")))
        self.assertFalse(os.path.exists(os.path.join(TestCatalogUtils.output_directory + 'invalid', "MAYA_q_hist.png")))
        self.assertFalse(
            os.path.exists(os.path.join(TestCatalogUtils.output_directory + 'invalid', "MAYA_a2_hist.png")))
        self.assertEqual(3, mock_show.call_count)
        mock_show.reset_mock()

        # don't provide a path
        TestCatalogUtils.catalog.plot_catalog_parameters(Parameter.MASS_RATIO, Parameter.DIMENSIONLESS_SPIN_2)
        self.assertFalse(
            os.path.exists(os.path.join(TestCatalogUtils.output_directory + 'invalid', "MAYA_q_a2_plot.png")))
        self.assertFalse(os.path.exists(os.path.join(TestCatalogUtils.output_directory + 'invalid', "MAYA_q_hist.png")))
        self.assertFalse(
            os.path.exists(os.path.join(TestCatalogUtils.output_directory + 'invalid', "MAYA_a2_hist.png")))
        self.assertEqual(3, mock_show.call_count)
        mock_show.reset_mock()

    def test_download_waveforms(self):
        with mock.patch('wget.download') as download_mock:
            TestCatalogUtils.catalog.download_waveforms(["GT0001"], save_wf_path=TestCatalogUtils.output_directory)
            self.assertEqual(1, download_mock.call_count)
            download_mock.assert_any_call('https://cgpstorage.ph.utexas.edu/maya_format/GT0001.h5',
                                          out=TestCatalogUtils.output_directory)

        with mock.patch('wget.download') as download_mock:
            TestCatalogUtils.catalog.download_waveforms(["GT0001", "GT0002"],
                                                        save_wf_path=TestCatalogUtils.output_directory)
            self.assertEqual(2, download_mock.call_count)
            download_mock.assert_any_call('https://cgpstorage.ph.utexas.edu/maya_format/GT0001.h5',
                                          out=TestCatalogUtils.output_directory)
            download_mock.assert_any_call('https://cgpstorage.ph.utexas.edu/maya_format/GT0002.h5',
                                          out=TestCatalogUtils.output_directory)

        with mock.patch('wget.download') as download_mock:
            TestCatalogUtils.catalog.download_waveforms(["GT0001"], save_wf_path=TestCatalogUtils.output_directory,
                                                        lvcnr_format=True)
            self.assertEqual(1, download_mock.call_count)
            download_mock.assert_any_call('https://cgpstorage.ph.utexas.edu/lvcnr_format/GT0001.h5',
                                          out=TestCatalogUtils.output_directory)

        with mock.patch('wget.download') as download_mock:
            TestCatalogUtils.catalog.download_waveforms(["GT0001", "GT0002"],
                                                        save_wf_path=TestCatalogUtils.output_directory,
                                                        lvcnr_format=True)
            self.assertEqual(2, download_mock.call_count)
            download_mock.assert_any_call('https://cgpstorage.ph.utexas.edu/lvcnr_format/GT0001.h5',
                                          out=TestCatalogUtils.output_directory)
            download_mock.assert_any_call('https://cgpstorage.ph.utexas.edu/lvcnr_format/GT0002.h5',
                                          out=TestCatalogUtils.output_directory)

        # try passing in invalid waveforms
        with mock.patch('wget.download') as download_mock:
            TestCatalogUtils.catalog.download_waveforms(["GT001"],
                                                        save_wf_path=TestCatalogUtils.output_directory)
            download_mock.assert_not_called()

        # try passing in invalid waveforms
        with mock.patch('wget.download') as download_mock:
            TestCatalogUtils.catalog.download_waveforms(["GT001", "GT0001"],
                                                        save_wf_path=TestCatalogUtils.output_directory)
            self.assertEqual(1, download_mock.call_count)
            download_mock.assert_any_call('https://cgpstorage.ph.utexas.edu/maya_format/GT0001.h5',
                                          out=TestCatalogUtils.output_directory)

        # invalid path
        TestCatalogUtils.catalog.download_waveforms(["GT0001"],
                                                    save_wf_path=TestCatalogUtils.output_directory + 'invalid')
        self.assertFalse(os.path.exists(os.path.join(TestCatalogUtils.output_directory, 'GT0001.h5')))

        # test if it's greater than 10 GB and user says y
        waveforms = TestCatalogUtils.catalog.get_simulations_with_mass_ratio(2)
        with mock.patch('builtins.input', return_value='y'):
            with mock.patch('wget.download') as download_mock:
                # call download function
                TestCatalogUtils.catalog.download_waveforms(waveforms[:20],
                                                            save_wf_path=TestCatalogUtils.output_directory)
                self.assertEqual(20, download_mock.call_count)
                for waveform in waveforms[:20]:
                    download_mock.assert_any_call(f'https://cgpstorage.ph.utexas.edu/maya_format/{waveform}.h5',
                                                  out=TestCatalogUtils.output_directory)

        # test if it's greater than 10 GB and user says something other than 'y'
        with mock.patch('builtins.input', return_value='n'):
            with mock.patch('wget.download') as download_mock:
                # call download function
                TestCatalogUtils.catalog.download_waveforms(waveforms[20:40],
                                                            save_wf_path=TestCatalogUtils.output_directory)
                download_mock.assert_not_called()
