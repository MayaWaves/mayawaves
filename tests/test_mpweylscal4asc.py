import shutil
from unittest import TestCase
import os
import numpy as np
import h5py


class TestMP_WeylScal4_ASC(TestCase):
    CURR_DIR = os.path.dirname(__file__)

    @classmethod
    def setUpClass(cls) -> None:
        from mayawaves.utils import postprocessingutils
        postprocessingutils._TIMESTEP = None

        TestMP_WeylScal4_ASC.output_directory = os.path.join(TestMP_WeylScal4_ASC.CURR_DIR, "resources/test_output")
        if os.path.exists(TestMP_WeylScal4_ASC.output_directory):
            shutil.rmtree(TestMP_WeylScal4_ASC.output_directory)
        os.mkdir(TestMP_WeylScal4_ASC.output_directory)

        h5_file_original = os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                                        "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        os.mkdir(os.path.join(TestMP_WeylScal4_ASC.CURR_DIR, "resources/temp"))
        shutil.copy2(h5_file_original, os.path.join(TestMP_WeylScal4_ASC.CURR_DIR, "resources/temp"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TestMP_WeylScal4_ASC.output_directory)

        if os.path.exists(os.path.join(TestMP_WeylScal4_ASC.CURR_DIR, "resources/temp.h5")):
            os.remove(os.path.join(TestMP_WeylScal4_ASC.CURR_DIR, "resources/temp.h5"))

        if os.path.exists(os.path.join(TestMP_WeylScal4_ASC.CURR_DIR, "resources/temp")):
            shutil.rmtree(os.path.join(TestMP_WeylScal4_ASC.CURR_DIR, "resources/temp"))

    def test_store_radiative_data_from_filetype(self):
        from mayawaves.utils.postprocessingutils import _MP_WeylScal4_ASC
        from mayawaves.utils.postprocessingutils import _RadiativeFilenames

        temp_h5_file = h5py.File(os.path.join(TestMP_WeylScal4_ASC.CURR_DIR, "resources/temp.h5"), 'w')
        psi4_group = temp_h5_file.create_group('psi4')

        filepaths = {
            'radiative': {
                _RadiativeFilenames.MP_WEYLSCAL4_ASC: {
                    "mp_WeylScal4::Psi4i_l2_m2_r70.00.asc": [
                        os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                                     "resources/psi4_formats/mp_WeylScal4::Psi4i_l2_m2_r70.00.asc")
                    ],
                    "mp_WeylScal4::Psi4i_l2_m1_r70.00.asc": [
                        os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                                     "resources/psi4_formats/mp_WeylScal4::Psi4i_l2_m1_r70.00.asc")
                    ],
                    "mp_WeylScal4::Psi4i_l2_m2_r80.00.asc": [
                        os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                                     "resources/psi4_formats/mp_WeylScal4::Psi4i_l2_m2_r80.00.asc")
                    ],
                    "mp_WeylScal4::Psi4i_l3_m3_r80.00.asc": [
                        os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                                     "resources/psi4_formats/mp_WeylScal4::Psi4i_l3_m3_r80.00.asc")
                    ],
                }
            }
        }

        expected_time_70_2_2, expected_real_70_2_2, expected_imag_70_2_2 = np.loadtxt(
            os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                         "resources/psi4_formats/mp_WeylScal4::Psi4i_l2_m2_r70.00.asc"),
            unpack=True)
        expected_time_70_2_1, expected_real_70_2_1, expected_imag_70_2_1 = np.loadtxt(
            os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                         "resources/psi4_formats/mp_WeylScal4::Psi4i_l2_m1_r70.00.asc"),
            unpack=True)
        expected_time_80_2_2, expected_real_80_2_2, expected_imag_80_2_2 = np.loadtxt(
            os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                         "resources/psi4_formats/mp_WeylScal4::Psi4i_l2_m2_r80.00.asc"),
            unpack=True)
        expected_time_80_3_3, expected_real_80_3_3, expected_imag_80_3_3 = np.loadtxt(
            os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                         "resources/psi4_formats/mp_WeylScal4::Psi4i_l3_m3_r80.00.asc"),
            unpack=True)

        expected_70_time, indices_2_2, indices_2_1 = np.intersect1d(expected_time_70_2_2, expected_time_70_2_1,
                                                                    assume_unique=True, return_indices=True)
        expected_time_70_2_2 = expected_time_70_2_2[indices_2_2]
        expected_real_70_2_2 = expected_real_70_2_2[indices_2_2]
        expected_imag_70_2_2 = expected_imag_70_2_2[indices_2_2]
        expected_time_70_2_1 = expected_time_70_2_1[indices_2_1]
        expected_real_70_2_1 = expected_real_70_2_1[indices_2_1]
        expected_imag_70_2_1 = expected_imag_70_2_1[indices_2_1]

        expected_80_time, indices_3_3, indices_2_2 = np.intersect1d(expected_time_80_3_3, expected_time_80_2_2,
                                                                    assume_unique=True, return_indices=True)
        expected_time_80_3_3 = expected_time_80_3_3[indices_3_3]
        expected_real_80_3_3 = expected_real_80_3_3[indices_3_3]
        expected_imag_80_3_3 = expected_imag_80_3_3[indices_3_3]
        expected_time_80_2_2 = expected_time_80_2_2[indices_2_2]
        expected_real_80_2_2 = expected_real_80_2_2[indices_2_2]
        expected_imag_80_2_2 = expected_imag_80_2_2[indices_2_2]

        _MP_WeylScal4_ASC.store_radiative_data_from_filetype(filepaths, psi4_group)

        expected_radius_keys = sorted(["radius=70.00", "radius=80.00"])
        actual_radius_keys = sorted(list(psi4_group.keys()))
        self.assertEqual(expected_radius_keys, actual_radius_keys)

        r70_group = psi4_group["radius=70.00"]
        expected_keys = sorted(["modes", "time"])
        actual_keys = sorted(list(r70_group.keys()))
        self.assertEqual(expected_keys, actual_keys)

        modes_group = r70_group["modes"]
        expected_l_keys = ["l=2"]
        actual_l_keys = sorted(list(modes_group.keys()))
        self.assertEqual(expected_l_keys, actual_l_keys)

        l2_group = modes_group["l=2"]
        expected_m_keys = sorted(["m=1", "m=2"])
        actual_m_keys = sorted(list(l2_group.keys()))
        self.assertEqual(expected_m_keys, actual_m_keys)

        l2_m2_group = l2_group["m=2"]
        expected_keys = sorted(["real", "imaginary"])
        actual_keys = sorted(list(l2_m2_group.keys()))
        self.assertEqual(expected_keys, actual_keys)
        actual_real_array = l2_m2_group["real"][()]
        actual_imag_array = l2_m2_group["imaginary"][()]
        self.assertTrue(np.all(expected_real_70_2_2 == actual_real_array))
        self.assertTrue(np.all(expected_imag_70_2_2 == actual_imag_array))

        l2_m1_group = l2_group["m=1"]
        expected_keys = sorted(["real", "imaginary"])
        actual_keys = sorted(list(l2_m1_group.keys()))
        self.assertEqual(expected_keys, actual_keys)
        actual_real_array = l2_m1_group["real"][()]
        actual_imag_array = l2_m1_group["imaginary"][()]
        self.assertTrue(np.all(expected_real_70_2_1 == actual_real_array))
        self.assertTrue(np.all(expected_imag_70_2_1 == actual_imag_array))

        actual_time_array = r70_group["time"][()]
        self.assertTrue(np.all(expected_70_time == actual_time_array))

        r80_group = psi4_group["radius=80.00"]
        expected_keys = sorted(["modes", "time"])
        actual_keys = sorted(list(r80_group.keys()))
        self.assertEqual(expected_keys, actual_keys)

        modes_group = r80_group["modes"]
        expected_l_keys = sorted(["l=2", "l=3"])
        actual_l_keys = sorted(list(modes_group.keys()))
        self.assertEqual(expected_l_keys, actual_l_keys)

        l2_group = modes_group["l=2"]
        expected_m_keys = sorted(["m=2"])
        actual_m_keys = sorted(list(l2_group.keys()))
        self.assertEqual(expected_m_keys, actual_m_keys)

        l2_m2_group = l2_group["m=2"]
        expected_keys = sorted(["real", "imaginary"])
        actual_keys = sorted(list(l2_m2_group.keys()))
        self.assertEqual(expected_keys, actual_keys)
        actual_real_array = l2_m2_group["real"][()]
        actual_imag_array = l2_m2_group["imaginary"][()]
        self.assertTrue(np.all(expected_real_80_2_2 == actual_real_array))
        self.assertTrue(np.all(expected_imag_80_2_2 == actual_imag_array))

        actual_time_array = r70_group["time"][()]
        self.assertTrue(np.all(expected_70_time == actual_time_array))

        l3_group = modes_group["l=3"]
        expected_m_keys = sorted(["m=3"])
        actual_m_keys = sorted(list(l3_group.keys()))
        self.assertEqual(expected_m_keys, actual_m_keys)

        l3_m3_group = l3_group["m=3"]
        expected_keys = sorted(["real", "imaginary"])
        actual_keys = sorted(list(l3_m3_group.keys()))
        self.assertEqual(expected_keys, actual_keys)
        actual_real_array = l3_m3_group["real"][()]
        actual_imag_array = l3_m3_group["imaginary"][()]
        self.assertTrue(np.all(expected_real_80_3_3 == actual_real_array))
        self.assertTrue(np.all(expected_imag_80_3_3 == actual_imag_array))

        actual_time_array = r80_group["time"][()]
        self.assertTrue(np.all(expected_80_time == actual_time_array))

    def test_get_header(self):
        from mayawaves.utils.postprocessingutils import _MP_WeylScal4_ASC

        l_value = 3
        m_value = -2
        radius = 120.00
        actual_header = _MP_WeylScal4_ASC.get_header(radius=radius, l_value=l_value, m_value=m_value)
        self.assertIsNone(actual_header)

        l_value = 3
        m_value = -2
        radius = 120
        actual_header = _MP_WeylScal4_ASC.get_header(radius=radius, l_value=l_value, m_value=m_value)
        self.assertIsNone(actual_header)

        l_value = 3
        m_value = -2
        radius = 120
        com_corrected = True
        expected_header = "Corrected for center of mass drift"
        actual_header = _MP_WeylScal4_ASC.get_header(radius=radius, l_value=l_value, m_value=m_value,
                                                     com_correction=com_corrected)
        self.assertEqual(expected_header, actual_header)

    def test_export_radiative_data_to_ascii(self):
        from mayawaves.utils.postprocessingutils import _MP_WeylScal4_ASC

        psi4_data_dict = {}
        # l=2, m=2, r=70
        expected_file = os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                                     'resources/psi4_formats/mp_WeylScal4::Psi4i_l2_m2_r70.00.asc')
        time, psi4_real, psi4_imag = np.loadtxt(expected_file, unpack=True)
        psi4_data_dict[70] = {
            (2, 2): np.column_stack([time, psi4_real, psi4_imag])
        }

        # l=3, m=3, r=80
        expected_file = os.path.join(TestMP_WeylScal4_ASC.CURR_DIR,
                                     'resources/psi4_formats/mp_WeylScal4::Psi4i_l3_m3_r80.00.asc')
        time, psi4_real, psi4_imag = np.loadtxt(expected_file, unpack=True)
        psi4_data_dict[80] = {
            (3, 3): np.column_stack([time, psi4_real, psi4_imag])
        }

        _MP_WeylScal4_ASC.export_radiative_data_to_ascii(psi4_data_dict=psi4_data_dict,
                                                         coalescence_output_directory=TestMP_WeylScal4_ASC.output_directory)

        # l=2, m=2, r=70
        generated_file = os.path.join(TestMP_WeylScal4_ASC.output_directory, 'mp_WeylScal4::Psi4i_l2_m2_r70.00.asc')
        self.assertTrue(os.path.isfile(generated_file))

        # header should be empty
        with open(generated_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    self.fail()
                else:
                    break

        # check content
        expected_data = psi4_data_dict[70][(2, 2)]

        actual_data = np.loadtxt(generated_file)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-6))

        os.remove(generated_file)

        # l=3, m=3, r=80
        generated_file = os.path.join(TestMP_WeylScal4_ASC.output_directory, 'mp_WeylScal4::Psi4i_l3_m3_r80.00.asc')
        self.assertTrue(os.path.isfile(generated_file))

        # header should be empty
        with open(generated_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    self.fail()
                else:
                    break

        # check content
        expected_data = psi4_data_dict[80][(3, 3)]

        actual_data = np.loadtxt(generated_file)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-6))

        os.remove(generated_file)

        # com correction
        _MP_WeylScal4_ASC.export_radiative_data_to_ascii(psi4_data_dict=psi4_data_dict,
                                                         coalescence_output_directory=TestMP_WeylScal4_ASC.output_directory,
                                                         com_correction=True)

        # l=2, m=2, r=70
        generated_file = os.path.join(TestMP_WeylScal4_ASC.output_directory, 'mp_WeylScal4::Psi4i_l2_m2_r70.00.asc')
        self.assertTrue(os.path.isfile(generated_file))

        expected_header = "Corrected for center of mass drift"
        actual_header = ""
        with open(generated_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    actual_header = actual_header + line[2:]
                else:
                    break
        actual_header = actual_header.rstrip()
        self.assertEqual(expected_header, actual_header)

        # check content
        expected_data = psi4_data_dict[70][(2, 2)]

        actual_data = np.loadtxt(generated_file)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-6))

        os.remove(generated_file)

        # l=3, m=3, r=80, com corrected
        generated_file = os.path.join(TestMP_WeylScal4_ASC.output_directory, 'mp_WeylScal4::Psi4i_l3_m3_r80.00.asc')
        self.assertTrue(os.path.isfile(generated_file))

        expected_header = "Corrected for center of mass drift"
        actual_header = ""
        with open(generated_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    actual_header = actual_header + line[2:]
                else:
                    break
        actual_header = actual_header.rstrip()
        self.assertEqual(expected_header, actual_header)

        # check content
        expected_data = psi4_data_dict[80][(3, 3)]

        actual_data = np.loadtxt(generated_file)

        self.assertTrue(np.allclose(expected_data, actual_data, atol=1e-6))

        os.remove(generated_file)
