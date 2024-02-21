import shutil
from unittest import TestCase
import os
import h5py
import numpy as np


class TestMP_Psi4_H5(TestCase):
    CURR_DIR = os.path.dirname(__file__)

    @classmethod
    def setUpClass(cls) -> None:
        from mayawaves.utils import postprocessingutils
        postprocessingutils._TIMESTEP = None

        TestMP_Psi4_H5.output_directory = os.path.join(TestMP_Psi4_H5.CURR_DIR, "resources/test_output")
        if os.path.exists(TestMP_Psi4_H5.output_directory):
            shutil.rmtree(TestMP_Psi4_H5.output_directory)
        os.mkdir(TestMP_Psi4_H5.output_directory)

        h5_file_original = os.path.join(TestMP_Psi4_H5.CURR_DIR,
                                        "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        os.mkdir(os.path.join(TestMP_Psi4_H5.CURR_DIR, "resources/temp"))
        shutil.copy2(h5_file_original, os.path.join(TestMP_Psi4_H5.CURR_DIR, "resources/temp"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(TestMP_Psi4_H5.output_directory)

        if os.path.exists(os.path.join(TestMP_Psi4_H5.CURR_DIR, "resources/temp.h5")):
            os.remove(os.path.join(TestMP_Psi4_H5.CURR_DIR, "resources/temp.h5"))

        if os.path.exists(os.path.join(TestMP_Psi4_H5.CURR_DIR, "resources/temp")):
            shutil.rmtree(os.path.join(TestMP_Psi4_H5.CURR_DIR, "resources/temp"))

    def test_store_radiative_data_from_filetype(self):
        from mayawaves.utils.postprocessingutils import _MP_Psi4_H5
        from mayawaves.utils.postprocessingutils import _RadiativeFilenames

        temp_h5_file = h5py.File(os.path.join(TestMP_Psi4_H5.CURR_DIR, "resources/temp.h5"), 'w')
        psi4_group = temp_h5_file.create_group('psi4')

        filepaths = {
            'radiative': {
                _RadiativeFilenames.MP_PSI4_H5: {
                    "mp_psi4.h5": [
                        os.path.join(TestMP_Psi4_H5.CURR_DIR,
                                     "resources/sample_etk_simulations/GW150914/output-0000/GW150914/mp_psi4.h5"),
                        os.path.join(TestMP_Psi4_H5.CURR_DIR,
                                     "resources/sample_etk_simulations/GW150914/output-0001/GW150914/mp_psi4.h5"),
                        os.path.join(TestMP_Psi4_H5.CURR_DIR,
                                     "resources/sample_etk_simulations/GW150914/output-0002/GW150914/mp_psi4.h5")
                    ]
                }
            }
        }

        _MP_Psi4_H5.store_radiative_data_from_filetype(filepaths, psi4_group)

        expected_radius_keys = sorted(['radius=100.00', 'radius=300.00', 'radius=500.00'])
        actual_radius_keys = sorted(list(psi4_group.keys()))
        self.assertEqual(expected_radius_keys, actual_radius_keys)

        r100_group = psi4_group["radius=100.00"]
        expected_keys = sorted(["modes", "time"])
        actual_keys = sorted(list(r100_group.keys()))
        self.assertEqual(expected_keys, actual_keys)

        for radius_key in expected_radius_keys:
            r_group = psi4_group[radius_key]
            modes_group = r_group["modes"]
            expected_l_keys = sorted(["l=0", "l=1", "l=2", "l=3"])
            actual_l_keys = sorted(list(modes_group.keys()))
            self.assertEqual(expected_l_keys, actual_l_keys)

            for l in range(0, 4):
                expected_m_keys = sorted([f"m={m}" for m in range(-l, l + 1)])
                actual_m_keys = sorted(list(modes_group[f"l={l}"]))
                self.assertEqual(expected_m_keys, actual_m_keys)

                for m in range(-l, l + 1):
                    l_m_group = modes_group[f"l={l}"][f"m={m}"]
                    expected_keys = sorted(["real", "imaginary"])
                    actual_keys = sorted(list(l_m_group.keys()))

                    psi4_0_file = h5py.File(os.path.join(TestMP_Psi4_H5.CURR_DIR,
                                                         "resources/sample_etk_simulations/GW150914/output-0000/GW150914/mp_psi4.h5"), )
                    psi4_1_file = h5py.File(os.path.join(TestMP_Psi4_H5.CURR_DIR,
                                                         "resources/sample_etk_simulations/GW150914/output-0001/GW150914/mp_psi4.h5"), )
                    psi4_2_file = h5py.File(os.path.join(TestMP_Psi4_H5.CURR_DIR,
                                                         "resources/sample_etk_simulations/GW150914/output-0002/GW150914/mp_psi4.h5"), )
                    psi4_data_0 = psi4_0_file[f"l{l}_m{m}_r{radius_key[7:]}"]
                    psi4_data_1 = psi4_1_file[f"l{l}_m{m}_r{radius_key[7:]}"]
                    psi4_data_2 = psi4_2_file[f"l{l}_m{m}_r{radius_key[7:]}"]

                    psi4_data = np.concatenate((psi4_data_0, psi4_data_1, psi4_data_2), axis=0)

                    expected_time = psi4_data[:, 0]
                    expected_real = psi4_data[:, 1]
                    expected_imag = psi4_data[:, 2]

                    actual_time = r_group["time"][()]
                    actual_real = l_m_group["real"][()]
                    actual_imag = l_m_group["imaginary"][()]

                    self.assertTrue(np.all(expected_time == actual_time))
                    self.assertTrue(np.all(expected_real == actual_real))
                    self.assertTrue(np.all(expected_imag == actual_imag))

    def test_export_radiative_data_to_ascii(self):
        from mayawaves.coalescence import Coalescence
        from mayawaves.utils.postprocessingutils import _MP_Psi4_H5
        test_coalescence = Coalescence(
            os.path.join(TestMP_Psi4_H5.CURR_DIR, 'resources/sample_etk_simulations/GW150914.h5'))

        psi4_data_dict = {}
        for radius in test_coalescence.included_extraction_radii:
            psi4_data_dict[radius] = {}
            for mode in test_coalescence.included_modes:
                time, real, imag = test_coalescence.psi4_real_imag_for_mode(l=mode[0], m=mode[1],
                                                                            extraction_radius=radius)
                psi4_data_dict[radius][mode] = np.column_stack([time, real, imag])

        _MP_Psi4_H5.export_radiative_data_to_ascii(psi4_data_dict=psi4_data_dict,
                                                   coalescence_output_directory=TestMP_Psi4_H5.output_directory)

        generated_filename = os.path.join(TestMP_Psi4_H5.output_directory, 'mp_psi4.h5')
        self.assertTrue(os.path.isfile(generated_filename))

        generated_file = h5py.File(generated_filename)

        for radius in test_coalescence.included_extraction_radii:
            for mode in test_coalescence.included_modes:
                expected_time, expected_real, expected_imag = test_coalescence.psi4_real_imag_for_mode(l=mode[0],
                                                                                                       m=mode[1],
                                                                                                       extraction_radius=radius)
                dataset_name = f"l{mode[0]}_m{mode[1]}_r{radius:.2f}"
                generated_data = generated_file[dataset_name][()]
                self.assertTrue(np.all(expected_time == generated_data[:, 0]))
                self.assertTrue(np.all(expected_real == generated_data[:, 1]))
                self.assertTrue(np.all(expected_imag == generated_data[:, 2]))

        self.assertTrue('frame' not in generated_file.attrs)

        generated_file.close()
        os.remove(generated_filename)

        _MP_Psi4_H5.export_radiative_data_to_ascii(psi4_data_dict=psi4_data_dict,
                                                   coalescence_output_directory=TestMP_Psi4_H5.output_directory,
                                                   com_correction=True)
        generated_filename = os.path.join(TestMP_Psi4_H5.output_directory, 'mp_psi4.h5')
        self.assertTrue(os.path.isfile(generated_filename))

        generated_file = h5py.File(generated_filename)

        self.assertTrue('frame' in list(generated_file.attrs))
        self.assertEqual('Corrected for center of mass drift', generated_file.attrs['frame'])

        generated_file.close()
