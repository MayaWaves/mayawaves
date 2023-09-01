import shutil
from unittest import TestCase
import numpy as np
import os
import pandas as pd
from copy import deepcopy


class TestIhspin_hn(TestCase):
    CURR_DIR = os.path.dirname(__file__)
    simulation_name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"

    @classmethod
    def setUpClass(cls) -> None:
        from mayawaves.utils import postprocessingutils
        postprocessingutils._TIMESTEP = None

        TestIhspin_hn.output_directory = os.path.join(TestIhspin_hn.CURR_DIR, "resources/test_output")
        if os.path.exists(TestIhspin_hn.output_directory):
            shutil.rmtree(TestIhspin_hn.output_directory)
        os.mkdir(TestIhspin_hn.output_directory)

        h5_file_original = os.path.join(TestIhspin_hn.CURR_DIR, "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        os.mkdir(os.path.join(TestIhspin_hn.CURR_DIR, "resources/temp"))
        shutil.copy2(h5_file_original, os.path.join(TestIhspin_hn.CURR_DIR, "resources/temp"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TestIhspin_hn.output_directory)

        if os.path.exists(os.path.join(TestIhspin_hn.CURR_DIR, "resources/temp.h5")):
            os.remove(os.path.join(TestIhspin_hn.CURR_DIR, "resources/temp.h5"))

        if os.path.exists(os.path.join(TestIhspin_hn.CURR_DIR, "resources/temp")):
            shutil.rmtree(os.path.join(TestIhspin_hn.CURR_DIR, "resources/temp"))

    def test_store_compact_object_data_from_filetype(self):
        from mayawaves.utils.postprocessingutils import _Ihspin_hn
        from mayawaves.utils.postprocessingutils import _BH_diagnostics
        from mayawaves.compactobject import CompactObject
        from mayawaves.utils import postprocessingutils

        # starting with empty dict, it won't work because _TIMESTEP isn't set
        filepaths = {
            "ihspin_hn_0.asc":
                [
                    os.path.join(TestIhspin_hn.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_0.asc") % (
                        TestIhspin_hn.simulation_name,
                        TestIhspin_hn.simulation_name)
                ],
            "ihspin_hn_1.asc":
                [
                    os.path.join(TestIhspin_hn.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_1.asc") % (
                        TestIhspin_hn.simulation_name,
                        TestIhspin_hn.simulation_name)
                ],
            "ihspin_hn_2.asc":
                [
                    os.path.join(TestIhspin_hn.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_2.asc") % (
                        TestIhspin_hn.simulation_name,
                        TestIhspin_hn.simulation_name),
                    os.path.join(TestIhspin_hn.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0001/%s/ihspin_hn_2.asc") % (
                        TestIhspin_hn.simulation_name,
                        TestIhspin_hn.simulation_name)
                ],
        }
        compact_object_dict = {}
        parameter_file = os.path.join(TestIhspin_hn.CURR_DIR,
                                      "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                             TestIhspin_hn.simulation_name,
                             TestIhspin_hn.simulation_name)

        actual_compact_object_dict, metadata_dict = _Ihspin_hn.store_compact_object_data_from_filetype(filepaths,
                                                                                                       compact_object_dict,
                                                                                                       parameter_file)

        self.assertEqual({}, actual_compact_object_dict)
        self.assertEqual({}, metadata_dict)

        # if data already exists in compact_object_dict
        previous_compact_object_dict = {}

        bh_diagnostics_data_1 = np.loadtxt(os.path.join(TestIhspin_hn.CURR_DIR,
                                                        "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp") % (
                                               TestIhspin_hn.simulation_name))
        indices_0 = bh_diagnostics_data_1[:, 0]
        compact_object_0_dataframe = pd.DataFrame(data=bh_diagnostics_data_1, index=indices_0,
                                                  columns=_BH_diagnostics.column_list)

        bh_diagnostics_data_2 = np.loadtxt(os.path.join(TestIhspin_hn.CURR_DIR,
                                                        "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp") % (
                                               TestIhspin_hn.simulation_name))
        indices_1 = bh_diagnostics_data_2[:, 0]
        compact_object_1_dataframe = pd.DataFrame(data=bh_diagnostics_data_2, index=indices_1,
                                                  columns=_BH_diagnostics.column_list)

        bh_diagnostics_data_3 = np.loadtxt(os.path.join(TestIhspin_hn.CURR_DIR,
                                                        "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah3.gp") % (
                                               TestIhspin_hn.simulation_name))
        indices_2 = bh_diagnostics_data_3[:, 0]
        compact_object_2_dataframe = pd.DataFrame(data=bh_diagnostics_data_3, index=indices_2,
                                                  columns=_BH_diagnostics.column_list)

        previous_compact_object_dict[0] = compact_object_0_dataframe
        previous_compact_object_dict[1] = compact_object_1_dataframe
        previous_compact_object_dict[2] = compact_object_2_dataframe

        postprocessingutils._TIMESTEP = bh_diagnostics_data_1[-1][1] / bh_diagnostics_data_1[-1][0]

        filepaths = {
            "ihspin_hn_0.asc":
                [
                    os.path.join(TestIhspin_hn.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_0.asc") % (
                        TestIhspin_hn.simulation_name,
                        TestIhspin_hn.simulation_name)
                ],
            "ihspin_hn_1.asc":
                [
                    os.path.join(TestIhspin_hn.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_1.asc") % (
                        TestIhspin_hn.simulation_name,
                        TestIhspin_hn.simulation_name)
                ],
            "ihspin_hn_2.asc":
                [
                    os.path.join(TestIhspin_hn.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_2.asc") % (
                        TestIhspin_hn.simulation_name,
                        TestIhspin_hn.simulation_name),
                    os.path.join(TestIhspin_hn.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0001/%s/ihspin_hn_2.asc") % (
                        TestIhspin_hn.simulation_name,
                        TestIhspin_hn.simulation_name)
                ],
        }

        parameter_file = os.path.join(TestIhspin_hn.CURR_DIR,
                                      "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                             TestIhspin_hn.simulation_name,
                             TestIhspin_hn.simulation_name)

        actual_compact_object_dict, metadata_dict = _Ihspin_hn.store_compact_object_data_from_filetype(filepaths,
                                                                                                       deepcopy(
                                                                                                           previous_compact_object_dict),
                                                                                                       parameter_file)

        expected_compact_objects = [0, 1, 2]
        actual_compact_objects = sorted(actual_compact_object_dict.keys())
        self.assertEqual(expected_compact_objects, actual_compact_objects)
        self.assertEqual({}, metadata_dict)

        expected_ihspin_0_data = np.loadtxt(os.path.join(TestIhspin_hn.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ihspin_hn_0.asc")
                                            % TestIhspin_hn.simulation_name)
        expected_ihspin_1_data = np.loadtxt(os.path.join(TestIhspin_hn.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ihspin_hn_1.asc")
                                            % TestIhspin_hn.simulation_name)
        expected_ihspin_2_data = np.loadtxt(os.path.join(TestIhspin_hn.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ihspin_hn_2.asc")
                                            % TestIhspin_hn.simulation_name)

        dataframe_columns = actual_compact_object_dict[0].columns.values.tolist()
        ih_spin_column_idxs = [dataframe_columns.index(col) for col in _Ihspin_hn.column_list if col is not None]

        ih_spin_column_idxs.insert(0, dataframe_columns.index(CompactObject.Column.TIME))

        raw_compact_object_0_data = actual_compact_object_dict[0].to_numpy()
        actual_ihspin_0_data = raw_compact_object_0_data[
                                   ~np.isnan(raw_compact_object_0_data[:, ih_spin_column_idxs[-1]])][:,
                               ih_spin_column_idxs]

        raw_compact_object_1_data = actual_compact_object_dict[1].to_numpy()
        actual_ihspin_1_data = raw_compact_object_1_data[
                                   ~np.isnan(raw_compact_object_1_data[:, ih_spin_column_idxs[-1]])][:,
                               ih_spin_column_idxs]

        raw_compact_object_2_data = actual_compact_object_dict[2].to_numpy()
        actual_ihspin_2_data = raw_compact_object_2_data[
                                   ~np.isnan(raw_compact_object_2_data[:, ih_spin_column_idxs[-1]])][:,
                               ih_spin_column_idxs]

        self.assertTrue(np.allclose(expected_ihspin_0_data, actual_ihspin_0_data, atol=0.05))
        self.assertTrue(np.allclose(expected_ihspin_1_data, actual_ihspin_1_data, atol=0.05))
        self.assertTrue(np.allclose(expected_ihspin_2_data, actual_ihspin_2_data, atol=0.05))

        bh_diagnostics_columns = _BH_diagnostics.column_list

        recovered_previous_data_0 = actual_compact_object_dict[0].loc[indices_0][bh_diagnostics_columns]
        expected_previous_data_0 = previous_compact_object_dict[0][bh_diagnostics_columns]
        self.assertEqual(list(expected_previous_data_0.index.values), list(recovered_previous_data_0.index.values))
        self.assertTrue(len(expected_previous_data_0) == len(recovered_previous_data_0))
        self.assertTrue(
            np.allclose(expected_previous_data_0.to_numpy(), recovered_previous_data_0.to_numpy(), atol=0.05))

        recovered_previous_data_1 = actual_compact_object_dict[1].loc[indices_1][bh_diagnostics_columns]
        expected_previous_data_1 = previous_compact_object_dict[1][bh_diagnostics_columns]
        self.assertEqual(list(expected_previous_data_1.index.values), list(recovered_previous_data_1.index.values))
        self.assertTrue(len(expected_previous_data_1) == len(recovered_previous_data_1))
        self.assertTrue(
            np.allclose(expected_previous_data_1.to_numpy(), recovered_previous_data_1.to_numpy(), atol=0.05))

        recovered_previous_data_2 = actual_compact_object_dict[2].loc[indices_2][bh_diagnostics_columns]
        expected_previous_data_2 = previous_compact_object_dict[2][bh_diagnostics_columns]
        self.assertEqual(list(expected_previous_data_2.index.values), list(recovered_previous_data_2.index.values))
        self.assertTrue(len(expected_previous_data_2) == len(recovered_previous_data_2))
        self.assertTrue(
            np.allclose(expected_previous_data_2.to_numpy(), recovered_previous_data_2.to_numpy(), atol=0.05))

    def test_export_compact_object_data_to_ascii(self):
        from mayawaves.coalescence import Coalescence
        from mayawaves.compactobject import CompactObject
        from mayawaves.utils.postprocessingutils import _Ihspin_hn

        h5_filename = os.path.join(TestIhspin_hn.CURR_DIR,
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

        _Ihspin_hn.export_compact_object_data_to_ascii(compact_object_dict=compact_object_dict,
                                                       coalescence_output_directory=TestIhspin_hn.output_directory,
                                                       metadata_dict={}, parfile_dict=parfile_dict)

        # primary compact object
        generated_file_0 = os.path.join(TestIhspin_hn.output_directory, 'ihspin_hn_0.asc')
        self.assertTrue(os.path.isfile(generated_file_0))

        # check header
        expected_header = """IHSpin
horizon no.=0
gnuplot column index:
1:t 2:Sx 3:Sy 4:Sz 5:Px 6:Py 7:Pz"""
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
        ihspin_columns = _Ihspin_hn.column_list
        compact_object_header = list(compact_object_0_data.attrs['header'])
        column_idxs = [compact_object_header.index(col.header_text) for col in ihspin_columns if col is not None]
        column_idxs.insert(0, compact_object_header.index(CompactObject.Column.TIME.header_text))
        expected_data = compact_object_0_data[()][~np.isnan(compact_object_0_data[:, column_idxs[-1]])][:, column_idxs]

        actual_data = np.loadtxt(generated_file_0)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-4))

        os.remove(generated_file_0)

        # secondary compact object
        generated_file_1 = os.path.join(TestIhspin_hn.output_directory, 'ihspin_hn_1.asc')
        self.assertTrue(os.path.isfile(generated_file_1))

        # check header
        expected_header = """IHSpin
horizon no.=1
gnuplot column index:
1:t 2:Sx 3:Sy 4:Sz 5:Px 6:Py 7:Pz"""
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
        ihspin_columns = _Ihspin_hn.column_list
        compact_object_header = list(compact_object_1_data.attrs['header'])
        column_idxs = [compact_object_header.index(col.header_text) for col in ihspin_columns if col is not None]
        column_idxs.insert(0, compact_object_header.index(CompactObject.Column.TIME.header_text))
        expected_data = compact_object_1_data[()][~np.isnan(compact_object_1_data[:, column_idxs[-1]])][:, column_idxs]

        actual_data = np.loadtxt(generated_file_1)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-4))

        os.remove(generated_file_1)

    def test_get_header(self):
        from mayawaves.utils.postprocessingutils import _Ihspin_hn

        horizon_num = 0
        horizon_count = 3

        expected_header = """IHSpin
horizon no.=0
gnuplot column index:
1:t 2:Sx 3:Sy 4:Sz 5:Px 6:Py 7:Pz"""
        actual_header = _Ihspin_hn.get_header(horizon_num, horizon_count)
        self.assertEqual(expected_header, actual_header)

        horizon_num = 1
        expected_header = """IHSpin
horizon no.=1
gnuplot column index:
1:t 2:Sx 3:Sy 4:Sz 5:Px 6:Py 7:Pz"""
        actual_header = _Ihspin_hn.get_header(horizon_num, horizon_count)
        self.assertEqual(expected_header, actual_header)
