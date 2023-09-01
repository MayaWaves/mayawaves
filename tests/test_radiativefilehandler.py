from unittest import TestCase
import h5py
import os
import numpy as np


class TestRadiativeFileHandler(TestCase):
    CURR_DIR = os.path.dirname(__file__)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(os.path.join(TestRadiativeFileHandler.CURR_DIR, "resources/temp.h5")):
            os.remove(os.path.join(TestRadiativeFileHandler.CURR_DIR, "resources/temp.h5"))

    def test_store_stitched_psi4_data_from_dict(self):
        from mayawaves.utils.postprocessingutils import _RadiativeFileHandler

        temp_h5_file = h5py.File(os.path.join(TestRadiativeFileHandler.CURR_DIR, "resources/temp.h5"), 'w')
        psi4_group = temp_h5_file.create_group('psi4')

        time_75 = [1, 2, 3, 4, 5]
        real_2_2_75 = [10, 20, 30, 40, 50]
        imag_2_2_75 = [-10, -20, -30, -40, -50]
        real_3_2_75 = [2, 4, 6, 8, 10]
        imag_3_2_75 = [-2, -4, -6, -8, -10]

        time_80 = [1, 2, 3, 4, 5, 6]
        real_2_2_80 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        imag_2_2_80 = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        real_2_1_80 = [0.2, 0.4, 0.6, 0.8, 1, 1.2]
        imag_2_1_80 = [-0.2, -0.4, -0.6, -0.8, -1, -1.2]

        psi4_data = {
            '75': {
                (2, 2): np.column_stack((time_75, real_2_2_75, imag_2_2_75)),
                (3, 2): np.column_stack((time_75, real_3_2_75, imag_3_2_75))
            },
            '80.00': {
                (2, 2): np.column_stack((time_80, real_2_2_80, imag_2_2_80)),
                (2, 1): np.column_stack((time_80, real_2_1_80, imag_2_1_80))
            }
        }

        _RadiativeFileHandler.store_stitched_psi4_data_from_dict(psi4_data, psi4_group)

        expected_radius_keys = sorted(['radius=75', 'radius=80.00'])
        actual_radius_keys = sorted(list(psi4_group.keys()))
        self.assertEqual(expected_radius_keys, actual_radius_keys)

        # r=75
        r75_group = psi4_group["radius=75"]
        expected_keys = sorted(["modes", "time"])
        actual_keys = sorted(list(r75_group.keys()))
        self.assertEqual(expected_keys, actual_keys)

        modes_group = r75_group["modes"]
        expected_l_keys = sorted(["l=2", "l=3"])
        actual_l_keys = sorted(list(modes_group.keys()))
        self.assertEqual(expected_l_keys, actual_l_keys)

        l2_group = modes_group["l=2"]
        expected_m_keys = ["m=2"]
        actual_m_keys = sorted(list(l2_group.keys()))
        self.assertEqual(expected_m_keys, actual_m_keys)

        l2_m2_group = l2_group["m=2"]
        expected_keys = sorted(["real", "imaginary"])
        actual_keys = sorted(list(l2_m2_group.keys()))
        self.assertEqual(expected_keys, actual_keys)
        actual_real_array = l2_m2_group["real"][()]
        actual_imag_array = l2_m2_group["imaginary"][()]
        self.assertTrue(np.all(real_2_2_75 == actual_real_array))
        self.assertTrue(np.all(imag_2_2_75 == actual_imag_array))

        l3_group = modes_group["l=3"]
        expected_m_keys = ["m=2"]
        actual_m_keys = sorted(list(l3_group.keys()))
        self.assertEqual(expected_m_keys, actual_m_keys)

        l3_m2_group = l3_group["m=2"]
        expected_keys = sorted(["real", "imaginary"])
        actual_keys = sorted(list(l3_m2_group.keys()))
        self.assertEqual(expected_keys, actual_keys)
        actual_real_array = l3_m2_group["real"][()]
        actual_imag_array = l3_m2_group["imaginary"][()]
        self.assertTrue(np.all(real_3_2_75 == actual_real_array))
        self.assertTrue(np.all(imag_3_2_75 == actual_imag_array))

        actual_time_array = r75_group["time"][()]
        self.assertTrue(np.all(time_75 == actual_time_array))

        # r=80
        r80_group = psi4_group["radius=80.00"]
        expected_keys = sorted(["modes", "time"])
        actual_keys = sorted(list(r80_group.keys()))
        self.assertEqual(expected_keys, actual_keys)

        modes_group = r80_group["modes"]
        expected_l_keys = sorted(["l=2"])
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
        self.assertTrue(np.all(real_2_2_80 == actual_real_array))
        self.assertTrue(np.all(imag_2_2_80 == actual_imag_array))

        l2_m1_group = l2_group["m=1"]
        expected_keys = sorted(["real", "imaginary"])
        actual_keys = sorted(list(l2_m1_group.keys()))
        self.assertEqual(expected_keys, actual_keys)
        actual_real_array = l2_m1_group["real"][()]
        actual_imag_array = l2_m1_group["imaginary"][()]
        self.assertTrue(np.all(real_2_1_80 == actual_real_array))
        self.assertTrue(np.all(imag_2_1_80 == actual_imag_array))

        actual_time_array = r80_group["time"][()]
        self.assertTrue(np.all(time_80 == actual_time_array))
