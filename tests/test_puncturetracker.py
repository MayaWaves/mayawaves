import shutil
from unittest import TestCase
import numpy as np
import os
import pandas as pd
from copy import deepcopy


class TestPuncturetracker(TestCase):
    CURR_DIR = os.path.dirname(__file__)
    simulation_name = "GW150914"
    output_directory = None

    @classmethod
    def setUpClass(cls) -> None:
        from mayawaves.utils import postprocessingutils
        postprocessingutils._TIMESTEP = None

        TestPuncturetracker.output_directory = os.path.join(TestPuncturetracker.CURR_DIR, "resources/test_output")
        if os.path.exists(TestPuncturetracker.output_directory):
            shutil.rmtree(TestPuncturetracker.output_directory)
        os.mkdir(TestPuncturetracker.output_directory)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(TestPuncturetracker.output_directory):
            shutil.rmtree(TestPuncturetracker.output_directory)

    def test_store_compact_object_data_from_filetype(self):
        from mayawaves.utils.postprocessingutils import _Puncturetracker
        from mayawaves.utils.postprocessingutils import _BH_diagnostics

        # starting with empty dict
        filepaths = {
            "puncturetracker-pt_loc..asc":
                [
                    os.path.join(TestPuncturetracker.CURR_DIR,
                                 "resources/sample_etk_simulations/%s/output-0000/%s/puncturetracker-pt_loc..asc") % (
                        TestPuncturetracker.simulation_name,
                        TestPuncturetracker.simulation_name)
                ]
        }

        compact_object_dict = {}
        parameter_file = os.path.join(TestPuncturetracker.CURR_DIR,
                                      "resources/sample_etk_simulations/%s/output-0000/%s.rpar") % (
                             TestPuncturetracker.simulation_name, TestPuncturetracker.simulation_name)

        actual_compact_object_dict, metadata_dict = _Puncturetracker.store_compact_object_data_from_filetype(filepaths,
                                                                                                             compact_object_dict,
                                                                                                             parameter_file)

        expected_compact_objects = [0, 1]
        actual_compact_objects = sorted(actual_compact_object_dict.keys())
        self.assertEqual(expected_compact_objects, actual_compact_objects)

        expected_metadata_dict = {
            'Puncturetracker_header_info': '''# 0D ASCII output created by CarpetIOASCII
# created on c486-061.stampede2.tacc.utexas.edu by dferg on May 07 2022 at 11:16:18-0500
# parameter filename: "/scratch/05765/dferg/simulations/GW150914/output-0000/GW150914.par"
# Build ID: build-bbh-etk-login4.stampede2.tacc.utexas.edu-dferg-2022.05.04-19.53.59-95126
# Simulation ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
# Run ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
''',
            'Puncturetracker_surface_count': 10
        }
        self.assertEqual(expected_metadata_dict, metadata_dict)

        expected_puncturetracker_0_data = np.loadtxt(os.path.join(TestPuncturetracker.CURR_DIR,
                                                                  "resources/sample_etk_simulations/%s/output-0000/%s/puncturetracker-pt_loc..asc") % (
                                                         TestPuncturetracker.simulation_name,
                                                         TestPuncturetracker.simulation_name),
                                                     usecols=(0, 12, 22, 32, 42))

        raw_compact_object_0_data = actual_compact_object_dict[0].to_numpy()
        actual_puncturetracker_0_data = raw_compact_object_0_data[~np.isnan(raw_compact_object_0_data[:, 4])][:, :5]

        self.assertTrue(np.all(expected_puncturetracker_0_data == actual_puncturetracker_0_data))

        expected_puncturetracker_1_data = np.loadtxt(os.path.join(TestPuncturetracker.CURR_DIR,
                                                                  "resources/sample_etk_simulations/%s/output-0000/%s/puncturetracker-pt_loc..asc") % (
                                                         TestPuncturetracker.simulation_name,
                                                         TestPuncturetracker.simulation_name),
                                                     usecols=(0, 13, 23, 33, 43))

        raw_compact_object_1_data = actual_compact_object_dict[1].to_numpy()
        actual_puncturetracker_1_data = raw_compact_object_1_data[~np.isnan(raw_compact_object_1_data[:, 4])][:, :5]

        self.assertTrue(np.all(expected_puncturetracker_1_data == actual_puncturetracker_1_data))

        # if data already exists in compact_object_dict
        previous_compact_object_dict = {}

        bh_diagnostics_data_1 = np.loadtxt(os.path.join(TestPuncturetracker.CURR_DIR,
                                                        "resources/sample_etk_simulations/%s/stitched/BH_diagnostics.ah1.gp") % (
                                               TestPuncturetracker.simulation_name))
        indices_0 = bh_diagnostics_data_1[:, 0]
        compact_object_0_dataframe = pd.DataFrame(data=bh_diagnostics_data_1, index=indices_0,
                                                  columns=_BH_diagnostics.column_list)

        bh_diagnostics_data_2 = np.loadtxt(os.path.join(TestPuncturetracker.CURR_DIR,
                                                        "resources/sample_etk_simulations/%s/stitched/BH_diagnostics.ah2.gp") % (
                                               TestPuncturetracker.simulation_name))
        indices_1 = bh_diagnostics_data_2[:, 0]
        compact_object_1_dataframe = pd.DataFrame(data=bh_diagnostics_data_2, index=indices_1,
                                                  columns=_BH_diagnostics.column_list)

        bh_diagnostics_data_3 = np.loadtxt(os.path.join(TestPuncturetracker.CURR_DIR,
                                                        "resources/sample_etk_simulations/%s/stitched/BH_diagnostics.ah3.gp") % (
                                               TestPuncturetracker.simulation_name))
        indices_2 = bh_diagnostics_data_3[:, 0]
        compact_object_2_dataframe = pd.DataFrame(data=bh_diagnostics_data_3, index=indices_2,
                                                  columns=_BH_diagnostics.column_list)

        previous_compact_object_dict[0] = compact_object_0_dataframe
        previous_compact_object_dict[1] = compact_object_1_dataframe
        previous_compact_object_dict[2] = compact_object_2_dataframe

        filepaths = {
            "puncturetracker-pt_loc..asc":
                [
                    os.path.join(TestPuncturetracker.CURR_DIR,
                                 "resources/sample_etk_simulations/%s/output-0000/%s/puncturetracker-pt_loc..asc") % (
                        TestPuncturetracker.simulation_name,
                        TestPuncturetracker.simulation_name)
                ]
        }

        compact_object_dict = {}
        parameter_file = os.path.join(TestPuncturetracker.CURR_DIR,
                                      "resources/sample_etk_simulations/%s/output-0000/%s.rpar") % (
                             TestPuncturetracker.simulation_name, TestPuncturetracker.simulation_name)

        actual_compact_object_dict, metadata_dict = _Puncturetracker.store_compact_object_data_from_filetype(filepaths,
                                                                                                             deepcopy(
                                                                                                                 previous_compact_object_dict),
                                                                                                             parameter_file)

        expected_compact_objects = [0, 1, 2]
        actual_compact_objects = sorted(actual_compact_object_dict.keys())
        self.assertEqual(expected_compact_objects, actual_compact_objects)

        expected_metadata_dict = {
            'Puncturetracker_header_info': '''# 0D ASCII output created by CarpetIOASCII
# created on c486-061.stampede2.tacc.utexas.edu by dferg on May 07 2022 at 11:16:18-0500
# parameter filename: "/scratch/05765/dferg/simulations/GW150914/output-0000/GW150914.par"
# Build ID: build-bbh-etk-login4.stampede2.tacc.utexas.edu-dferg-2022.05.04-19.53.59-95126
# Simulation ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
# Run ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
''',
            'Puncturetracker_surface_count': 10
        }
        self.assertEqual(expected_metadata_dict, metadata_dict)

        expected_puncturetracker_0_data = np.loadtxt(os.path.join(TestPuncturetracker.CURR_DIR,
                                                                  "resources/sample_etk_simulations/%s/output-0000/%s/puncturetracker-pt_loc..asc") % (
                                                         TestPuncturetracker.simulation_name,
                                                         TestPuncturetracker.simulation_name),
                                                     usecols=(0, 12, 22, 32, 42))

        puncture_tracker_columns = _Puncturetracker.column_list_primary_object
        header_list = actual_compact_object_dict[0].columns.values.tolist()
        puncture_tracker_column_idxs = [header_list.index(col) for col in puncture_tracker_columns if col is not None]

        raw_compact_object_0_data = actual_compact_object_dict[0].to_numpy()
        actual_puncturetracker_0_data = raw_compact_object_0_data[
                                            ~np.isnan(raw_compact_object_0_data[:, puncture_tracker_column_idxs[-1]])][
                                        :, puncture_tracker_column_idxs]

        common_iterations, indices_generated, indices_expected = np.intersect1d(actual_puncturetracker_0_data[:, 0],
                                                                                expected_puncturetracker_0_data[:, 0],
                                                                                assume_unique=True,
                                                                                return_indices=True)

        self.assertTrue(np.all(
            expected_puncturetracker_0_data[indices_expected] == actual_puncturetracker_0_data[indices_generated]))

        expected_puncturetracker_1_data = np.loadtxt(os.path.join(TestPuncturetracker.CURR_DIR,
                                                                  "resources/sample_etk_simulations/%s/output-0000/%s/puncturetracker-pt_loc..asc") % (
                                                         TestPuncturetracker.simulation_name,
                                                         TestPuncturetracker.simulation_name),
                                                     usecols=(0, 13, 23, 33, 43))

        raw_compact_object_1_data = actual_compact_object_dict[1].to_numpy()
        actual_puncturetracker_1_data = raw_compact_object_1_data[
                                            ~np.isnan(raw_compact_object_1_data[:, puncture_tracker_column_idxs[-1]])][
                                        :, puncture_tracker_column_idxs]

        common_iterations, indices_generated, indices_expected = np.intersect1d(actual_puncturetracker_1_data[:, 0],
                                                                                expected_puncturetracker_1_data[:, 0],
                                                                                assume_unique=True,
                                                                                return_indices=True)

        self.assertTrue(np.all(
            expected_puncturetracker_1_data[indices_expected] == actual_puncturetracker_1_data[indices_generated]))

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
        from mayawaves.utils.postprocessingutils import _Puncturetracker

        h5_filename = os.path.join(TestPuncturetracker.CURR_DIR,
                                   "resources/sample_etk_simulations/GW150914.h5")
        coalescence = Coalescence(h5_filename)

        compact_object_0_data = coalescence.compact_object_data_for_object(0)
        compact_object_1_data = coalescence.compact_object_data_for_object(1)

        compact_object_dict = {
            0: compact_object_0_data,
            1: compact_object_1_data
        }

        metadata_dict = coalescence.compact_object_metadata_dict()

        parfile_dict = coalescence.parameter_files

        _Puncturetracker.export_compact_object_data_to_ascii(compact_object_dict=compact_object_dict,
                                                             coalescence_output_directory=TestPuncturetracker.output_directory,
                                                             metadata_dict=metadata_dict, parfile_dict=parfile_dict)

        generated_file = os.path.join(TestPuncturetracker.output_directory, 'puncturetracker-pt_loc..asc')
        self.assertTrue(os.path.isfile(generated_file))

        # check header
        expected_header = """# 0D ASCII output created by CarpetIOASCII
# created on c486-061.stampede2.tacc.utexas.edu by dferg on May 07 2022 at 11:16:18-0500
# parameter filename: "/scratch/05765/dferg/simulations/GW150914/output-0000/GW150914.par"
# Build ID: build-bbh-etk-login4.stampede2.tacc.utexas.edu-dferg-2022.05.04-19.53.59-95126
# Simulation ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
# Run ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
#
# PUNCTURETRACKER::PT_LOC (puncturetracker-pt_loc)
#
# iteration 0   time 0.000000
# time level 0
# refinement level 0   multigrid level 0   map 0   component 0
# column format: 1:it	2:tl	3:rl 4:c 5:ml	6:ix 7:iy 8:iz	9:time	10:x 11:y 12:z	13:data
# data columns: 13:pt_loc_t[0] 14:pt_loc_t[1] 15:pt_loc_t[2] 16:pt_loc_t[3] 17:pt_loc_t[4] 18:pt_loc_t[5] 19:pt_loc_t[6] 20:pt_loc_t[7] 21:pt_loc_t[8] 22:pt_loc_t[9] 23:pt_loc_x[0] 24:pt_loc_x[1] 25:pt_loc_x[2] 26:pt_loc_x[3] 27:pt_loc_x[4] 28:pt_loc_x[5] 29:pt_loc_x[6] 30:pt_loc_x[7] 31:pt_loc_x[8] 32:pt_loc_x[9] 33:pt_loc_y[0] 34:pt_loc_y[1] 35:pt_loc_y[2] 36:pt_loc_y[3] 37:pt_loc_y[4] 38:pt_loc_y[5] 39:pt_loc_y[6] 40:pt_loc_y[7] 41:pt_loc_y[8] 42:pt_loc_y[9] 43:pt_loc_z[0] 44:pt_loc_z[1] 45:pt_loc_z[2] 46:pt_loc_z[3] 47:pt_loc_z[4] 48:pt_loc_z[5] 49:pt_loc_z[6] 50:pt_loc_z[7] 51:pt_loc_z[8] 52:pt_loc_z[9]"""
        actual_header = ""
        with open(generated_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    actual_header = actual_header + line
                else:
                    break
        actual_header = actual_header.rstrip()
        self.assertEqual(expected_header, actual_header)

        # check content
        expected_file = os.path.join(TestPuncturetracker.CURR_DIR,
                                     "resources/sample_etk_simulations/GW150914/stitched/puncturetracker-pt_loc..asc")
        expected_data = np.loadtxt(expected_file)

        actual_data = np.loadtxt(generated_file)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-2))

        os.remove(generated_file)
