import shutil
from unittest import TestCase
import numpy as np
import os
import pandas as pd
from copy import deepcopy


class TestShifttracker(TestCase):
    CURR_DIR = os.path.dirname(__file__)
    simulation_name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"

    @classmethod
    def setUpClass(cls) -> None:
        from mayawaves.utils import postprocessingutils
        postprocessingutils._TIMESTEP = None

        TestShifttracker.output_directory = os.path.join(TestShifttracker.CURR_DIR, "resources/test_output")
        if os.path.exists(TestShifttracker.output_directory):
            shutil.rmtree(TestShifttracker.output_directory)
        os.mkdir(TestShifttracker.output_directory)

        h5_file_original = os.path.join(TestShifttracker.CURR_DIR, "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        os.mkdir(os.path.join(TestShifttracker.CURR_DIR, "resources/temp"))
        shutil.copy2(h5_file_original, os.path.join(TestShifttracker.CURR_DIR, "resources/temp"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TestShifttracker.output_directory)

        if os.path.exists(os.path.join(TestShifttracker.CURR_DIR, "resources/temp.h5")):
            os.remove(os.path.join(TestShifttracker.CURR_DIR, "resources/temp.h5"))

        if os.path.exists(os.path.join(TestShifttracker.CURR_DIR, "resources/temp")):
            shutil.rmtree(os.path.join(TestShifttracker.CURR_DIR, "resources/temp"))

    def test_store_compact_object_data_from_filetype(self):
        from mayawaves.utils.postprocessingutils import _Shifttracker
        from mayawaves.utils.postprocessingutils import _BH_diagnostics

        # starting with empty dict
        filepaths = {
            "ShiftTracker0.asc":
                [
                    os.path.join(TestShifttracker.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ShiftTracker0.asc") % (
                        TestShifttracker.simulation_name,
                        TestShifttracker.simulation_name),
                    os.path.join(TestShifttracker.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0001/%s/ShiftTracker0.asc") % (
                        TestShifttracker.simulation_name,
                        TestShifttracker.simulation_name)
                ],
            "ShiftTracker1.asc":
                [
                    os.path.join(TestShifttracker.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ShiftTracker1.asc") % (
                        TestShifttracker.simulation_name,
                        TestShifttracker.simulation_name),
                    os.path.join(TestShifttracker.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0001/%s/ShiftTracker1.asc") % (
                        TestShifttracker.simulation_name,
                        TestShifttracker.simulation_name)
                ]
        }
        compact_object_dict = {}
        parameter_file = os.path.join(TestShifttracker.CURR_DIR,
                                      "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                             TestShifttracker.simulation_name,
                             TestShifttracker.simulation_name)

        actual_compact_object_dict, metadata_dict = _Shifttracker.store_compact_object_data_from_filetype(filepaths,
                                                                                                          compact_object_dict,
                                                                                                          parameter_file)

        expected_compact_objects = [0, 1]
        actual_compact_objects = sorted(actual_compact_object_dict.keys())
        self.assertEqual(expected_compact_objects, actual_compact_objects)
        self.assertEqual({}, metadata_dict)

        expected_shifttracker_0_data = np.loadtxt(os.path.join(TestShifttracker.CURR_DIR,
                                                               "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc")
                                                  % TestShifttracker.simulation_name)
        expected_shifttracker_1_data = np.loadtxt(os.path.join(TestShifttracker.CURR_DIR,
                                                               "resources/main_test_simulation/stitched/%s/ShiftTracker1.asc")
                                                  % TestShifttracker.simulation_name)

        shifttracker_columns = _Shifttracker.column_list
        header_list = actual_compact_object_dict[0].columns.values.tolist()
        shifttracker_column_idxs = [header_list.index(col) for col in shifttracker_columns if col is not None]

        raw_compact_object_0_data = actual_compact_object_dict[0].to_numpy()
        actual_shifttracker_0_data = raw_compact_object_0_data[
                                         ~np.isnan(raw_compact_object_0_data[:, shifttracker_column_idxs[-1]])][:,
                                     shifttracker_column_idxs]

        raw_compact_object_1_data = actual_compact_object_dict[1].to_numpy()
        actual_shifttracker_1_data = raw_compact_object_1_data[
                                         ~np.isnan(raw_compact_object_1_data[:, shifttracker_column_idxs[-1]])][:,
                                     shifttracker_column_idxs]

        self.assertTrue(np.all(expected_shifttracker_0_data == actual_shifttracker_0_data))
        self.assertTrue(np.all(expected_shifttracker_1_data == actual_shifttracker_1_data))

        # if data already exists in compact_object_dict
        filepaths = {
            "ShiftTracker0.asc":
                [
                    os.path.join(TestShifttracker.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ShiftTracker0.asc") % (
                        TestShifttracker.simulation_name,
                        TestShifttracker.simulation_name),
                    os.path.join(TestShifttracker.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0001/%s/ShiftTracker0.asc") % (
                        TestShifttracker.simulation_name,
                        TestShifttracker.simulation_name)
                ],
            "ShiftTracker1.asc":
                [
                    os.path.join(TestShifttracker.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0000/%s/ShiftTracker1.asc") % (
                        TestShifttracker.simulation_name,
                        TestShifttracker.simulation_name),
                    os.path.join(TestShifttracker.CURR_DIR,
                                 "resources/main_test_simulation/%s/output-0001/%s/ShiftTracker1.asc") % (
                        TestShifttracker.simulation_name,
                        TestShifttracker.simulation_name)
                ]
        }

        parameter_file = os.path.join(TestShifttracker.CURR_DIR,
                                      "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                             TestShifttracker.simulation_name,
                             TestShifttracker.simulation_name)

        previous_compact_object_dict = {}
        bh_diagnostics_data_1 = np.loadtxt(os.path.join(TestShifttracker.CURR_DIR,
                                                        "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                           % TestShifttracker.simulation_name)
        indices_0 = bh_diagnostics_data_1[:, 0]
        compact_object_0_dataframe = pd.DataFrame(data=bh_diagnostics_data_1, index=indices_0,
                                                  columns=_BH_diagnostics.column_list)

        bh_diagnostics_data_2 = np.loadtxt(os.path.join(TestShifttracker.CURR_DIR,
                                                        "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                           % TestShifttracker.simulation_name)
        indices_1 = bh_diagnostics_data_2[:, 0]
        compact_object_1_dataframe = pd.DataFrame(data=bh_diagnostics_data_2, index=indices_1,
                                                  columns=_BH_diagnostics.column_list)

        bh_diagnostics_data_3 = np.loadtxt(os.path.join(TestShifttracker.CURR_DIR,
                                                        "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah3.gp")
                                           % TestShifttracker.simulation_name)
        indices_2 = bh_diagnostics_data_3[:, 0]
        compact_object_2_dataframe = pd.DataFrame(data=bh_diagnostics_data_3, index=indices_2,
                                                  columns=_BH_diagnostics.column_list)

        previous_compact_object_dict[0] = compact_object_0_dataframe
        previous_compact_object_dict[1] = compact_object_1_dataframe
        previous_compact_object_dict[2] = compact_object_2_dataframe

        actual_compact_object_dict, metadata_dict = _Shifttracker.store_compact_object_data_from_filetype(filepaths,
                                                                                                          deepcopy(
                                                                                                              previous_compact_object_dict),
                                                                                                          parameter_file)

        expected_compact_objects = [0, 1, 2]
        actual_compact_objects = sorted(actual_compact_object_dict.keys())
        self.assertEqual(expected_compact_objects, actual_compact_objects)
        self.assertEqual({}, metadata_dict)

        expected_shifttracker_0_data = np.loadtxt(os.path.join(TestShifttracker.CURR_DIR,
                                                               "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc")
                                                  % TestShifttracker.simulation_name)
        expected_shifttracker_1_data = np.loadtxt(os.path.join(TestShifttracker.CURR_DIR,
                                                               "resources/main_test_simulation/stitched/%s/ShiftTracker1.asc")
                                                  % TestShifttracker.simulation_name)

        header_list = actual_compact_object_dict[0].columns.values.tolist()
        shifttracker_column_idxs = [header_list.index(col) for col in shifttracker_columns if col is not None]

        raw_compact_object_0_data = actual_compact_object_dict[0].to_numpy()
        actual_shifttracker_0_data = raw_compact_object_0_data[
                                         ~np.isnan(raw_compact_object_0_data[:, shifttracker_column_idxs[-1]])][:,
                                     shifttracker_column_idxs]

        raw_compact_object_1_data = actual_compact_object_dict[1].to_numpy()
        actual_shifttracker_1_data = raw_compact_object_1_data[
                                         ~np.isnan(raw_compact_object_1_data[:, shifttracker_column_idxs[-1]])][:,
                                     shifttracker_column_idxs]

        self.assertTrue(np.all(expected_shifttracker_0_data == actual_shifttracker_0_data))
        self.assertTrue(np.all(expected_shifttracker_1_data == actual_shifttracker_1_data))

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
        from mayawaves.utils.postprocessingutils import _Shifttracker

        h5_filename = os.path.join(TestShifttracker.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)

        compact_object_0_data = coalescence.compact_object_data_for_object(0)
        compact_object_1_data = coalescence.compact_object_data_for_object(1)

        compact_object_dict = {
            0: compact_object_0_data,
            1: compact_object_1_data
        }

        parfile_dict = coalescence.parameter_files

        _Shifttracker.export_compact_object_data_to_ascii(compact_object_dict=compact_object_dict,
                                                          coalescence_output_directory=TestShifttracker.output_directory,
                                                          metadata_dict={}, parfile_dict=parfile_dict)

        # primary compact object
        generated_file_0 = os.path.join(TestShifttracker.output_directory, 'ShiftTracker0.asc')
        self.assertTrue(os.path.isfile(generated_file_0))

        # check header
        expected_header = """ShiftTracker0.asc:
itt   time    x       y       z       vx      vy      vz      ax      ay      az
======================================================================="""
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
        shifttracker_columns = _Shifttracker.column_list
        compact_object_header = list(compact_object_0_data.attrs['header'])
        column_idxs = [compact_object_header.index(col.header_text) for col in shifttracker_columns]
        expected_data = compact_object_0_data[()][~np.isnan(compact_object_0_data[:, column_idxs[-1]])][:, column_idxs]

        actual_data = np.loadtxt(generated_file_0)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-4))

        os.remove(generated_file_0)

        # secondary compact object
        generated_file_1 = os.path.join(TestShifttracker.output_directory, 'ShiftTracker1.asc')
        self.assertTrue(os.path.isfile(generated_file_1))

        # check header
        expected_header = """ShiftTracker1.asc:
itt   time    x       y       z       vx      vy      vz      ax      ay      az
======================================================================="""
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
        shifttracker_columns = _Shifttracker.column_list
        compact_object_header = list(compact_object_1_data.attrs['header'])
        column_idxs = [compact_object_header.index(col.header_text) for col in shifttracker_columns]
        expected_data = compact_object_1_data[()][~np.isnan(compact_object_1_data[:, column_idxs[-1]])][:, column_idxs]

        actual_data = np.loadtxt(generated_file_1)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-4))

        os.remove(generated_file_1)

    def test_get_header(self):
        from mayawaves.utils.postprocessingutils import _Shifttracker

        horizon_num = 0
        horizon_count = 3

        expected_header = """ShiftTracker0.asc:
itt   time    x       y       z       vx      vy      vz      ax      ay      az
======================================================================="""
        actual_header = _Shifttracker.get_header(horizon_num, horizon_count)
        self.assertEqual(expected_header, actual_header)

        horizon_num = 1
        expected_header = """ShiftTracker1.asc:
itt   time    x       y       z       vx      vy      vz      ax      ay      az
======================================================================="""
        actual_header = _Shifttracker.get_header(horizon_num, horizon_count)
        self.assertEqual(expected_header, actual_header)
