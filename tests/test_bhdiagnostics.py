import shutil
from unittest import TestCase
import numpy as np
import os


class TestBH_diagnostics(TestCase):
    CURR_DIR = os.path.dirname(__file__)
    simulation_name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
    output_directory = None

    @classmethod
    def setUpClass(cls) -> None:
        from mayawaves.utils import postprocessingutils
        postprocessingutils._TIMESTEP = None

        TestBH_diagnostics.output_directory = os.path.join(TestBH_diagnostics.CURR_DIR, "resources/test_output")
        if os.path.exists(TestBH_diagnostics.output_directory):
            shutil.rmtree(TestBH_diagnostics.output_directory)
        os.mkdir(TestBH_diagnostics.output_directory)

        h5_file_original = os.path.join(TestBH_diagnostics.CURR_DIR, "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        os.mkdir(os.path.join(TestBH_diagnostics.CURR_DIR, "resources/temp"))
        shutil.copy2(h5_file_original, os.path.join(TestBH_diagnostics.CURR_DIR, "resources/temp"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TestBH_diagnostics.output_directory)

        if os.path.exists(os.path.join(TestBH_diagnostics.CURR_DIR, "resources/temp.h5")):
            os.remove(os.path.join(TestBH_diagnostics.CURR_DIR, "resources/temp.h5"))

        if os.path.exists(os.path.join(TestBH_diagnostics.CURR_DIR, "resources/temp")):
            shutil.rmtree(os.path.join(TestBH_diagnostics.CURR_DIR, "resources/temp"))

    def test_store_compact_object_data_from_filetype(self):
        from mayawaves.utils.postprocessingutils import _BH_diagnostics

        filepaths = {
            "BH_diagnostics.ah1.gp":
                [
                    os.path.join(TestBH_diagnostics.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/BH_diagnostics.ah1.gp") % (
                        TestBH_diagnostics.simulation_name,
                        TestBH_diagnostics.simulation_name)
                ],
            "BH_diagnostics.ah2.gp":
                [
                    os.path.join(TestBH_diagnostics.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/BH_diagnostics.ah2.gp") % (
                        TestBH_diagnostics.simulation_name,
                        TestBH_diagnostics.simulation_name)
                ],
            "BH_diagnostics.ah3.gp":
                [
                    os.path.join(TestBH_diagnostics.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/BH_diagnostics.ah3.gp") % (
                        TestBH_diagnostics.simulation_name,
                        TestBH_diagnostics.simulation_name),
                    os.path.join(TestBH_diagnostics.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0001/%s/BH_diagnostics.ah3.gp") % (
                        TestBH_diagnostics.simulation_name,
                        TestBH_diagnostics.simulation_name)
                ]
        }
        compact_object_dict = {}
        parameter_file = os.path.join(TestBH_diagnostics.CURR_DIR,
                                      "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                             TestBH_diagnostics.simulation_name,
                             TestBH_diagnostics.simulation_name)

        actual_compact_object_dict, metadata_dict = _BH_diagnostics.store_compact_object_data_from_filetype(filepaths,
                                                                                                            compact_object_dict,
                                                                                                            parameter_file)

        bh_diagnostics_columns = _BH_diagnostics.column_list

        expected_compact_objects = [0, 1, 2]
        actual_compact_objects = sorted(actual_compact_object_dict.keys())
        self.assertEqual(expected_compact_objects, actual_compact_objects)
        self.assertEqual({}, metadata_dict)

        expected_bh_diagnostics_1_data = np.loadtxt(os.path.join(TestBH_diagnostics.CURR_DIR,
                                                                 "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                                    % TestBH_diagnostics.simulation_name)
        expected_bh_diagnostics_2_data = np.loadtxt(os.path.join(TestBH_diagnostics.CURR_DIR,
                                                                 "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                                    % TestBH_diagnostics.simulation_name)
        expected_bh_diagnostics_3_data = np.loadtxt(os.path.join(TestBH_diagnostics.CURR_DIR,
                                                                 "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah3.gp")
                                                    % TestBH_diagnostics.simulation_name)

        header_list = actual_compact_object_dict[0].columns.values.tolist()
        bh_diagnostics_column_idxs = [header_list.index(col) for col in bh_diagnostics_columns]

        raw_compact_object_0_data = actual_compact_object_dict[0].to_numpy()
        actual_bh_diagnostics_1_data = raw_compact_object_0_data[
                                           ~np.isnan(raw_compact_object_0_data[:, bh_diagnostics_column_idxs[-1]])][:,
                                       bh_diagnostics_column_idxs]
        raw_compact_object_1_data = actual_compact_object_dict[1].to_numpy()
        actual_bh_diagnostics_2_data = raw_compact_object_1_data[
                                           ~np.isnan(raw_compact_object_1_data[:, bh_diagnostics_column_idxs[-1]])][:,
                                       bh_diagnostics_column_idxs]
        raw_compact_object_2_data = actual_compact_object_dict[2].to_numpy()
        actual_bh_diagnostics_3_data = raw_compact_object_2_data[
                                           ~np.isnan(raw_compact_object_2_data[:, bh_diagnostics_column_idxs[-1]])][:,
                                       bh_diagnostics_column_idxs]

        # BH_diagnostics 1, should be stored as compact object 0
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_1_data[:, 5:], actual_bh_diagnostics_1_data[:, 5:], atol=0.001)))
        # check position data
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_1_data[:, 2:5], actual_bh_diagnostics_1_data[:, 2:5], atol=0.05)))
        # check iteration column
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_1_data[:, 0], actual_bh_diagnostics_1_data[:, 0], rtol=0.001)))
        # check time column
        # the time column is rounded off in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_1_data[:, 1], actual_bh_diagnostics_1_data[:, 1], atol=0.001)))

        # BH_diagnostics 2, should be stored as compact object 1
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_2_data[:, 5:], actual_bh_diagnostics_2_data[:, 5:], atol=0.001)))
        # check position data
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_2_data[:, 2:5], actual_bh_diagnostics_2_data[:, 2:5], atol=0.05)))
        # check iteration column
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_2_data[:, 0], actual_bh_diagnostics_2_data[:, 0], rtol=0.001)))
        # check time column
        # the time column is rounded off in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_2_data[:, 1], actual_bh_diagnostics_2_data[:, 1], atol=0.001)))

        # BH_diagnostics 3, should be stored as compact object 2
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_3_data[:, 5:], actual_bh_diagnostics_3_data[:, 5:], atol=0.001)))
        # check position data
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_3_data[:, 2:5], actual_bh_diagnostics_3_data[:, 2:5], atol=0.05)))
        # check iteration column
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_3_data[:, 0], actual_bh_diagnostics_3_data[:, 0], rtol=0.001)))
        # check time column
        # the time column is rounded off in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_bh_diagnostics_3_data[:, 1], actual_bh_diagnostics_3_data[:, 1], atol=0.001)))

    def test_export_compact_object_data_to_ascii(self):
        from mayawaves.coalescence import Coalescence
        from mayawaves.utils.postprocessingutils import _BH_diagnostics

        h5_filename = os.path.join(TestBH_diagnostics.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)

        compact_object_0_data = coalescence.compact_object_data_for_object(0)
        compact_object_1_data = coalescence.compact_object_data_for_object(1)
        compact_object_2_data = coalescence.compact_object_data_for_object(2)

        compact_object_dict = {
            0: compact_object_0_data,
            1: compact_object_1_data,
            2: compact_object_2_data
        }

        parfile_dict = coalescence.parameter_files

        _BH_diagnostics.export_compact_object_data_to_ascii(compact_object_dict=compact_object_dict,
                                                            coalescence_output_directory=TestBH_diagnostics.output_directory,
                                                            metadata_dict={}, parfile_dict=parfile_dict)

        # primary compact object
        generated_file_0 = os.path.join(TestBH_diagnostics.output_directory, 'BH_diagnostics.ah1.gp')
        self.assertTrue(os.path.isfile(generated_file_0))

        # check header
        expected_header = """apparent horizon 1/3
column  1 = cctk_iteration
column  2 = cctk_time
column  3 = centroid_x
column  4 = centroid_y
column  5 = centroid_z
column  6 = min radius
column  7 = max radius
column  8 = mean radius
column  9 = quadrupole_xx
column 10 = quadrupole_xy
column 11 = quadrupole_xz
column 12 = quadrupole_yy
column 13 = quadrupole_yz
column 14 = quadrupole_zz
column 15 = min x
column 16 = max x
column 17 = min y
column 18 = max y
column 19 = min z
column 20 = max z
column 21 = xy-plane circumference
column 22 = xz-plane circumference
column 23 = yz-plane circumference
column 24 = ratio of xz/xy-plane circumferences
column 25 = ratio of yz/xy-plane circumferences
column 26 = area
column 27 = m_irreducible
column 28 = areal radius
column 29 = expansion Theta_(l)
column 30 = inner expansion Theta_(n)
column 31 = product of the expansions
column 32 = mean curvature
column 33 = gradient of the areal radius
column 34 = gradient of the expansion Theta_(l)
column 35 = gradient of the inner expansion Theta_(n)
column 36 = gradient of the product of the expansions
column 37 = gradient of the mean curvature
column 38 = minimum  of the mean curvature
column 39 = maximum  of the mean curvature
column 40 = integral of the mean curvature"""
        actual_header = ""
        with open(generated_file_0, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    actual_header = actual_header + line[2:]
                else:
                    break
        actual_header = actual_header.rstrip()
        self.assertEqual(expected_header, actual_header)

        # check content
        bh_diagnostics_columns = _BH_diagnostics.column_list
        compact_object_header = list(compact_object_0_data.attrs['header'])
        column_idxs = [compact_object_header.index(col.header_text) for col in bh_diagnostics_columns]
        expected_data = compact_object_0_data[()][~np.isnan(compact_object_0_data[:, column_idxs[-1]])][:, column_idxs]

        actual_data = np.loadtxt(generated_file_0)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-3))

        os.remove(generated_file_0)

        # secondary compact object
        generated_file_1 = os.path.join(TestBH_diagnostics.output_directory, 'BH_diagnostics.ah2.gp')
        self.assertTrue(os.path.isfile(generated_file_1))

        # check header
        expected_header = """apparent horizon 2/3
column  1 = cctk_iteration
column  2 = cctk_time
column  3 = centroid_x
column  4 = centroid_y
column  5 = centroid_z
column  6 = min radius
column  7 = max radius
column  8 = mean radius
column  9 = quadrupole_xx
column 10 = quadrupole_xy
column 11 = quadrupole_xz
column 12 = quadrupole_yy
column 13 = quadrupole_yz
column 14 = quadrupole_zz
column 15 = min x
column 16 = max x
column 17 = min y
column 18 = max y
column 19 = min z
column 20 = max z
column 21 = xy-plane circumference
column 22 = xz-plane circumference
column 23 = yz-plane circumference
column 24 = ratio of xz/xy-plane circumferences
column 25 = ratio of yz/xy-plane circumferences
column 26 = area
column 27 = m_irreducible
column 28 = areal radius
column 29 = expansion Theta_(l)
column 30 = inner expansion Theta_(n)
column 31 = product of the expansions
column 32 = mean curvature
column 33 = gradient of the areal radius
column 34 = gradient of the expansion Theta_(l)
column 35 = gradient of the inner expansion Theta_(n)
column 36 = gradient of the product of the expansions
column 37 = gradient of the mean curvature
column 38 = minimum  of the mean curvature
column 39 = maximum  of the mean curvature
column 40 = integral of the mean curvature"""
        actual_header = ""
        with open(generated_file_1, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    actual_header = actual_header + line[2:]
                else:
                    break
        actual_header = actual_header.rstrip()
        self.assertEqual(expected_header, actual_header)

        # check content
        bh_diagnostics_columns = _BH_diagnostics.column_list
        compact_object_header = list(compact_object_1_data.attrs['header'])
        column_idxs = [compact_object_header.index(col.header_text) for col in bh_diagnostics_columns]
        expected_data = compact_object_1_data[()][~np.isnan(compact_object_1_data[:, column_idxs[-1]])][:, column_idxs]

        actual_data = np.loadtxt(generated_file_1)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-3))

        os.remove(generated_file_1)

    def test_get_header(self):
        from mayawaves.utils.postprocessingutils import _BH_diagnostics

        horizon_num = 0
        horizon_count = 3
        expected_header = """apparent horizon 1/3
column  1 = cctk_iteration
column  2 = cctk_time
column  3 = centroid_x
column  4 = centroid_y
column  5 = centroid_z
column  6 = min radius
column  7 = max radius
column  8 = mean radius
column  9 = quadrupole_xx
column 10 = quadrupole_xy
column 11 = quadrupole_xz
column 12 = quadrupole_yy
column 13 = quadrupole_yz
column 14 = quadrupole_zz
column 15 = min x
column 16 = max x
column 17 = min y
column 18 = max y
column 19 = min z
column 20 = max z
column 21 = xy-plane circumference
column 22 = xz-plane circumference
column 23 = yz-plane circumference
column 24 = ratio of xz/xy-plane circumferences
column 25 = ratio of yz/xy-plane circumferences
column 26 = area
column 27 = m_irreducible
column 28 = areal radius
column 29 = expansion Theta_(l)
column 30 = inner expansion Theta_(n)
column 31 = product of the expansions
column 32 = mean curvature
column 33 = gradient of the areal radius
column 34 = gradient of the expansion Theta_(l)
column 35 = gradient of the inner expansion Theta_(n)
column 36 = gradient of the product of the expansions
column 37 = gradient of the mean curvature
column 38 = minimum  of the mean curvature
column 39 = maximum  of the mean curvature
column 40 = integral of the mean curvature"""
        actual_header = _BH_diagnostics.get_header(horizon_num, horizon_count)
        self.assertEqual(expected_header, actual_header)
