import shutil
from unittest import TestCase
import h5py
import numpy as np
import os


class TestQuasiLocalMeasures(TestCase):
    CURR_DIR = os.path.dirname(__file__)
    simulation_name = "GW150914"
    output_directory = None

    @classmethod
    def setUpClass(cls) -> None:
        from mayawaves.utils import postprocessingutils
        postprocessingutils._TIMESTEP = None

        TestQuasiLocalMeasures.output_directory = os.path.join(TestQuasiLocalMeasures.CURR_DIR, "resources/test_output")
        if os.path.exists(TestQuasiLocalMeasures.output_directory):
            shutil.rmtree(TestQuasiLocalMeasures.output_directory)
        os.mkdir(TestQuasiLocalMeasures.output_directory)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(TestQuasiLocalMeasures.output_directory):
            shutil.rmtree(TestQuasiLocalMeasures.output_directory)

    def test_store_compact_object_data_from_filetype(self):
        from mayawaves.utils.postprocessingutils import _QuasiLocalMeasures

        quasilocalmeasures_columns = _QuasiLocalMeasures.column_list[1:]

        # only one output, 6 tracked surfaces
        simulation_name = "qc0-mclachlan"
        filepaths = {
            "quasilocalmeasures-qlm_scalars..asc":
                [
                    os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                 "resources/sample_etk_simulations/%s/output-0000/%s/quasilocalmeasures-qlm_scalars..asc") % (
                    simulation_name, simulation_name)
                ]
        }
        parameter_file_path = os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                           "resources/sample_etk_simulations/%s/output-0000/%s.par") % (
                              simulation_name, simulation_name)
        temp_h5_file = h5py.File(os.path.join(TestQuasiLocalMeasures.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parameter_file_path) as f:
            parfile_group.attrs['par_content'] = f.read()
        compact_object_dict = {}
        actual_compact_object_dict, metadata_dict = _QuasiLocalMeasures.store_compact_object_data_from_filetype(
            filepaths, compact_object_dict, parfile_group)

        expected_compact_objects = [0, 1, 2, 3, 4, 5]
        actual_compact_objects = sorted(actual_compact_object_dict.keys())
        self.assertEqual(expected_compact_objects, actual_compact_objects)

        expected_metadata_dict = {
            'QuasiLocalMeasures_header_info': '''# 0D ASCII output created by CarpetIOASCII
# created on c478-094.stampede2.tacc.utexas.edu by dferg on May 05 2022 at 12:57:29-0500
# parameter filename: "/scratch/05765/dferg/simulations/qc0-mclachlan/output-0000/qc0-mclachlan.par"
# Build ID: build-bbh-etk-login4.stampede2.tacc.utexas.edu-dferg-2022.05.04-19.53.59-95126
# Simulation ID: run-qc0-mclachlan-c478-094.stampede2.tacc.utexas.edu-dferg-2022.05.05-17.56.37-180456
# Run ID: run-qc0-mclachlan-c478-094.stampede2.tacc.utexas.edu-dferg-2022.05.05-17.56.37-180456
''',
            'QuasiLocalMeasures_surface_count': 6
        }
        self.assertEqual(expected_metadata_dict, metadata_dict)

        raw_expected_data_0 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                      "resources/sample_etk_simulations/%s/output-0000/%s/quasilocalmeasures-qlm_scalars..asc") % (
                                             simulation_name, simulation_name),
                                         usecols=range(12, 210, 6))  # [12, 18, 24])
        expected_data_0 = raw_expected_data_0[~np.isnan(raw_expected_data_0[:, 0])]
        _, i = np.unique(expected_data_0[:, 0], return_index=True)
        expected_data_0 = expected_data_0[i]
        expected_data_0 = expected_data_0[expected_data_0[:, 1] != 0]
        header_list = actual_compact_object_dict[0].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_0_data = actual_compact_object_dict[0].to_numpy()
        actual_data_0 = raw_compact_object_0_data[
                            ~np.isnan(raw_compact_object_0_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_0 == actual_data_0))

        raw_expected_data_1 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                      "resources/sample_etk_simulations/%s/output-0000/%s/quasilocalmeasures-qlm_scalars..asc") % (
                                             simulation_name, simulation_name), usecols=range(13, 210, 6))
        expected_data_1 = raw_expected_data_1[~np.isnan(raw_expected_data_1[:, 0])]
        _, i = np.unique(expected_data_1[:, 0], return_index=True)
        expected_data_1 = expected_data_1[i]
        expected_data_1 = expected_data_1[expected_data_1[:, 1] != 0]
        header_list = actual_compact_object_dict[1].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_1_data = actual_compact_object_dict[1].to_numpy()
        actual_data_1 = raw_compact_object_1_data[
                            ~np.isnan(raw_compact_object_1_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_1 == actual_data_1))

        raw_expected_spin_data_2 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                           "resources/sample_etk_simulations/%s/output-0000/%s/quasilocalmeasures-qlm_scalars..asc") % (
                                                  simulation_name, simulation_name), usecols=range(14, 210, 6))
        expected_data_2 = raw_expected_spin_data_2[~np.isnan(raw_expected_spin_data_2[:, 0])]
        _, i = np.unique(expected_data_2[:, 0], return_index=True)
        expected_data_2 = expected_data_2[i]
        expected_data_2 = expected_data_2[expected_data_2[:, 1] != 0]
        header_list = actual_compact_object_dict[2].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_2_data = actual_compact_object_dict[2].to_numpy()
        actual_data_2 = raw_compact_object_2_data[
                            ~np.isnan(raw_compact_object_2_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_2 == actual_data_2))

        raw_expected_data_3 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                      "resources/sample_etk_simulations/%s/output-0000/%s/quasilocalmeasures-qlm_scalars..asc") % (
                                             simulation_name, simulation_name), usecols=range(15, 210, 6))
        expected_data_3 = raw_expected_data_3[~np.isnan(raw_expected_data_3[:, 0])]
        _, i = np.unique(expected_data_3[:, 0], return_index=True)
        expected_data_3 = expected_data_3[i]
        expected_data_3 = expected_data_3[expected_data_3[:, 1] != 0]
        header_list = actual_compact_object_dict[3].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_3_data = actual_compact_object_dict[3].to_numpy()
        actual_data_3 = raw_compact_object_3_data[
                            ~np.isnan(raw_compact_object_3_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_3 == actual_data_3))

        raw_expected_data_4 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                      "resources/sample_etk_simulations/%s/output-0000/%s/quasilocalmeasures-qlm_scalars..asc") % (
                                             simulation_name, simulation_name), usecols=range(16, 210, 6))
        expected_data_4 = raw_expected_data_4[~np.isnan(raw_expected_data_4[:, 0])]
        _, i = np.unique(expected_data_4[:, 0], return_index=True)
        expected_data_4 = expected_data_4[i]
        expected_data_4 = expected_data_4[expected_data_4[:, 1] != 0]
        header_list = actual_compact_object_dict[4].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_4_data = actual_compact_object_dict[4].to_numpy()
        actual_data_4 = raw_compact_object_4_data[
                            ~np.isnan(raw_compact_object_4_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_4 == actual_data_4))

        raw_expected_data_5 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                      "resources/sample_etk_simulations/%s/output-0000/%s/quasilocalmeasures-qlm_scalars..asc") % (
                                             simulation_name, simulation_name), usecols=range(17, 210, 6))
        expected_data_5 = raw_expected_data_5[~np.isnan(raw_expected_data_5[:, 0])]
        expected_data_5 = expected_data_5[expected_data_5[:, 1] != 0]
        header_list = actual_compact_object_dict[5].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_5_data = actual_compact_object_dict[5].to_numpy()
        actual_data_5 = raw_compact_object_5_data[
                            ~np.isnan(raw_compact_object_5_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_5 == actual_data_5))

        temp_h5_file.close()
        os.remove(os.path.join(TestQuasiLocalMeasures.CURR_DIR, "resources/temp.h5"))

        # multiple outputs, 3 tracked surfaces
        simulation_name = "GW150914"
        filepaths = {
            "quasilocalmeasures-qlm_scalars..asc":
                [
                    os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                 "resources/sample_etk_simulations/%s/output-0000/%s/quasilocalmeasures-qlm_scalars..asc") % (
                        simulation_name, simulation_name),
                    os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                 "resources/sample_etk_simulations/%s/output-0001/%s/quasilocalmeasures-qlm_scalars..asc") % (
                        simulation_name, simulation_name),
                    os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                 "resources/sample_etk_simulations/%s/output-0002/%s/quasilocalmeasures-qlm_scalars..asc") % (
                        simulation_name, simulation_name)
                ]
        }
        parameter_file_path = os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                           "resources/sample_etk_simulations/%s/output-0000/%s.par") % (
                                  simulation_name, simulation_name)
        temp_h5_file = h5py.File(os.path.join(TestQuasiLocalMeasures.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parameter_file_path) as f:
            parfile_group.attrs['par_content'] = f.read()
        compact_object_dict = {}
        actual_compact_object_dict, metadata_dict = _QuasiLocalMeasures.store_compact_object_data_from_filetype(
            filepaths,
            compact_object_dict,
            parfile_group)

        expected_compact_objects = [0, 1, 2]
        actual_compact_objects = sorted(actual_compact_object_dict.keys())
        self.assertEqual(expected_compact_objects, actual_compact_objects)

        expected_metadata_dict = {
            'QuasiLocalMeasures_header_info': '''# 0D ASCII output created by CarpetIOASCII
# created on c486-061.stampede2.tacc.utexas.edu by dferg on May 07 2022 at 11:16:17-0500
# parameter filename: "/scratch/05765/dferg/simulations/GW150914/output-0000/GW150914.par"
# Build ID: build-bbh-etk-login4.stampede2.tacc.utexas.edu-dferg-2022.05.04-19.53.59-95126
# Simulation ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
# Run ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
''',
            'QuasiLocalMeasures_surface_count': 3
        }
        self.assertEqual(expected_metadata_dict, metadata_dict)

        raw_expected_data_0 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                      "resources/sample_etk_simulations/%s/stitched/quasilocalmeasures-qlm_scalars..asc") % (
                                             simulation_name), usecols=range(12, 111, 3))
        expected_data_0 = raw_expected_data_0[~np.isnan(raw_expected_data_0[:, 0])]
        _, i = np.unique(expected_data_0[:, 0], return_index=True)
        expected_data_0 = expected_data_0[i]
        expected_data_0 = expected_data_0[expected_data_0[:, 1] != 0]
        header_list = actual_compact_object_dict[0].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_0_data = actual_compact_object_dict[0].to_numpy()
        actual_data_0 = raw_compact_object_0_data[
                            ~np.isnan(raw_compact_object_0_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_0 == actual_data_0))

        raw_expected_data_1 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                      "resources/sample_etk_simulations/%s/stitched/quasilocalmeasures-qlm_scalars..asc") % (
                                             simulation_name), usecols=range(13, 111, 3))
        expected_data_1 = raw_expected_data_1[~np.isnan(raw_expected_data_1[:, 0])]
        _, i = np.unique(expected_data_1[:, 0], return_index=True)
        expected_data_1 = expected_data_1[i]
        expected_data_1 = expected_data_1[expected_data_1[:, 1] != 0]
        header_list = actual_compact_object_dict[1].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_1_data = actual_compact_object_dict[1].to_numpy()
        actual_data_1 = raw_compact_object_1_data[
                            ~np.isnan(raw_compact_object_1_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_1 == actual_data_1))

        raw_expected_spin_data_2 = np.loadtxt(os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                                           "resources/sample_etk_simulations/%s/stitched/quasilocalmeasures-qlm_scalars..asc") % (
                                                  simulation_name), usecols=range(14, 111, 3))
        expected_data_2 = raw_expected_spin_data_2[~np.isnan(raw_expected_spin_data_2[:, 0])]
        _, i = np.unique(expected_data_2[:, 0], return_index=True)
        expected_data_2 = expected_data_2[i]
        expected_data_2 = expected_data_2[expected_data_2[:, 1] != 0]
        header_list = actual_compact_object_dict[2].columns.values.tolist()
        quasilocalmeasures_column_idxs = [header_list.index(col) for col in quasilocalmeasures_columns]
        raw_compact_object_2_data = actual_compact_object_dict[2].to_numpy()
        actual_data_2 = raw_compact_object_2_data[
                            ~np.isnan(raw_compact_object_2_data[:, quasilocalmeasures_column_idxs[-1]])][:,
                        quasilocalmeasures_column_idxs]
        self.assertTrue(np.all(expected_data_2 == actual_data_2))

        temp_h5_file.close()
        os.remove(os.path.join(TestQuasiLocalMeasures.CURR_DIR, "resources/temp.h5"))

    def test_export_compact_object_data_to_ascii(self):
        from mayawaves.coalescence import Coalescence
        from mayawaves.utils.postprocessingutils import _QuasiLocalMeasures

        h5_filename = os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                   "resources/sample_etk_simulations/GW150914.h5")
        coalescence = Coalescence(h5_filename)

        compact_object_0_data = coalescence.compact_object_data_for_object(0)
        compact_object_1_data = coalescence.compact_object_data_for_object(1)
        compact_object_2_data = coalescence.compact_object_data_for_object(2)

        compact_object_dict = {
            0: compact_object_0_data,
            1: compact_object_1_data,
            2: compact_object_2_data
        }

        metadata_dict = coalescence.compact_object_metadata_dict()

        parfile_dict = coalescence.parameter_files

        _QuasiLocalMeasures.export_compact_object_data_to_ascii(compact_object_dict=compact_object_dict,
                                                                coalescence_output_directory=TestQuasiLocalMeasures.output_directory,
                                                                metadata_dict=metadata_dict, parfile_dict=parfile_dict)

        generated_file = os.path.join(TestQuasiLocalMeasures.output_directory, 'quasilocalmeasures-qlm_scalars..asc')
        self.assertTrue(os.path.isfile(generated_file))

        # check header
        expected_header = """# Export from Mayawaves. Some values (time, irreducible mass, area, radius) saved using BHDiagnostics data rather than the raw QLM data.
# 0D ASCII output created by CarpetIOASCII
# created on c486-061.stampede2.tacc.utexas.edu by dferg on May 07 2022 at 11:16:17-0500
# parameter filename: "/scratch/05765/dferg/simulations/GW150914/output-0000/GW150914.par"
# Build ID: build-bbh-etk-login4.stampede2.tacc.utexas.edu-dferg-2022.05.04-19.53.59-95126
# Simulation ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
# Run ID: run-GW150914-c486-061.stampede2.tacc.utexas.edu-dferg-2022.05.07-16.11.55-102406
#
# QUASILOCALMEASURES::QLM_SCALARS (quasilocalmeasures-qlm_scalars)
#
# iteration 0   time 0.000000
# time level 0
# refinement level 0   multigrid level 0   map 0   component 0
# column format: 1:it	2:tl	3:rl 4:c 5:ml	6:ix 7:iy 8:iz	9:time	10:x 11:y 12:z	13:data
# data columns: 13:qlm_time[0] 14:qlm_time[1] 15:qlm_time[2] 16:qlm_equatorial_circumference[0] 17:qlm_equatorial_circumference[1] 18:qlm_equatorial_circumference[2] 19:qlm_polar_circumference_0[0] 20:qlm_polar_circumference_0[1] 21:qlm_polar_circumference_0[2] 22:qlm_polar_circumference_pi_2[0] 23:qlm_polar_circumference_pi_2[1] 24:qlm_polar_circumference_pi_2[2] 25:qlm_area[0] 26:qlm_area[1] 27:qlm_area[2] 28:qlm_irreducible_mass[0] 29:qlm_irreducible_mass[1] 30:qlm_irreducible_mass[2] 31:qlm_radius[0] 32:qlm_radius[1] 33:qlm_radius[2] 34:qlm_spin_guess[0] 35:qlm_spin_guess[1] 36:qlm_spin_guess[2] 37:qlm_mass_guess[0] 38:qlm_mass_guess[1] 39:qlm_mass_guess[2] 40:qlm_killing_eigenvalue_re[0] 41:qlm_killing_eigenvalue_re[1] 42:qlm_killing_eigenvalue_re[2] 43:qlm_killing_eigenvalue_im[0] 44:qlm_killing_eigenvalue_im[1] 45:qlm_killing_eigenvalue_im[2] 46:qlm_spin[0] 47:qlm_spin[1] 48:qlm_spin[2] 49:qlm_npspin[0] 50:qlm_npspin[1] 51:qlm_npspin[2] 52:qlm_wsspin[0] 53:qlm_wsspin[1] 54:qlm_wsspin[2] 55:qlm_cvspin[0] 56:qlm_cvspin[1] 57:qlm_cvspin[2] 58:qlm_coordspinx[0] 59:qlm_coordspinx[1] 60:qlm_coordspinx[2] 61:qlm_coordspiny[0] 62:qlm_coordspiny[1] 63:qlm_coordspiny[2] 64:qlm_coordspinz[0] 65:qlm_coordspinz[1] 66:qlm_coordspinz[2] 67:qlm_mass[0] 68:qlm_mass[1] 69:qlm_mass[2] 70:qlm_adm_energy[0] 71:qlm_adm_energy[1] 72:qlm_adm_energy[2] 73:qlm_adm_momentum_x[0] 74:qlm_adm_momentum_x[1] 75:qlm_adm_momentum_x[2] 76:qlm_adm_momentum_y[0] 77:qlm_adm_momentum_y[1] 78:qlm_adm_momentum_y[2] 79:qlm_adm_momentum_z[0] 80:qlm_adm_momentum_z[1] 81:qlm_adm_momentum_z[2] 82:qlm_adm_angular_momentum_x[0] 83:qlm_adm_angular_momentum_x[1] 84:qlm_adm_angular_momentum_x[2] 85:qlm_adm_angular_momentum_y[0] 86:qlm_adm_angular_momentum_y[1] 87:qlm_adm_angular_momentum_y[2] 88:qlm_adm_angular_momentum_z[0] 89:qlm_adm_angular_momentum_z[1] 90:qlm_adm_angular_momentum_z[2] 91:qlm_w_energy[0] 92:qlm_w_energy[1] 93:qlm_w_energy[2] 94:qlm_w_momentum_x[0] 95:qlm_w_momentum_x[1] 96:qlm_w_momentum_x[2] 97:qlm_w_momentum_y[0] 98:qlm_w_momentum_y[1] 99:qlm_w_momentum_y[2] 100:qlm_w_momentum_z[0] 101:qlm_w_momentum_z[1] 102:qlm_w_momentum_z[2] 103:qlm_w_angular_momentum_x[0] 104:qlm_w_angular_momentum_x[1] 105:qlm_w_angular_momentum_x[2] 106:qlm_w_angular_momentum_y[0] 107:qlm_w_angular_momentum_y[1] 108:qlm_w_angular_momentum_y[2] 109:qlm_w_angular_momentum_z[0] 110:qlm_w_angular_momentum_z[1] 111:qlm_w_angular_momentum_z[2]"""
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
        expected_file = os.path.join(TestQuasiLocalMeasures.CURR_DIR,
                                     "resources/sample_etk_simulations/GW150914/stitched/quasilocalmeasures-qlm_scalars..asc")
        expected_data = np.loadtxt(expected_file)

        actual_data = np.loadtxt(generated_file)

        overwritten_data_columns = [8, 12, 13, 14, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        qlm_alone_columns = list(set(range(0, actual_data.shape[1])) - set(overwritten_data_columns))

        actual_data = np.nan_to_num(actual_data)
        expected_data = np.nan_to_num(expected_data)

        self.assertTrue(
            np.allclose(expected_data[:, overwritten_data_columns], actual_data[:, overwritten_data_columns],
                        rtol=2e-2))
        self.assertTrue(np.allclose(expected_data[:, qlm_alone_columns], actual_data[:, qlm_alone_columns], atol=1e-4))

        os.remove(generated_file)
