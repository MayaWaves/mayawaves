import os
from unittest import TestCase
from unittest.mock import patch, PropertyMock
import h5py
import scipy.optimize
from mayawaves.coalescence import Coalescence
from mayawaves.radiation import RadiationSphere
import numpy as np
from mayawaves.radiation import RadiationMode


class TestRadiationSphere(TestCase):
    # define some helper values
    time = np.array([1, 2, 3, 4, 5])
    real22 = np.array([20, 21, 22, 23, 24])
    imag22 = np.array([25, 26, 27, 28, 29])
    real33 = np.array([30, 31, 32, 33, 34])
    imag33 = np.array([35, 36, 37, 38, 39])
    real31 = np.array([300, 310, 320, 330, 340])
    imag31 = np.array([350, 360, 370, 380, 390])

    def setUp(self) -> None:
        modes = {}
        time = np.array([])
        radius = 0
        self.radiation_sphere = RadiationSphere(mode_dict=modes, time=time, radius=radius)

        TestRadiationSphere.CURR_DIR = os.path.dirname(__file__)

    def test_create_radiation_sphere(self):
        temp_dir = os.path.join(TestRadiationSphere.CURR_DIR, "resources/temp")
        os.mkdir(temp_dir)
        temp_h5_filename = os.path.join(TestRadiationSphere.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        radius_group = temp_h5_file.create_group('radius')
        radius_group.create_group('modes')

        new_radiation_sphere = RadiationSphere.create_radiation_sphere(radius_group=radius_group, radius=0)
        self.assertTrue(new_radiation_sphere is None)

        os.remove(temp_h5_filename)
        os.rmdir(temp_dir)

        temp_dir = os.path.join(TestRadiationSphere.CURR_DIR, "resources/temp")
        os.mkdir(temp_dir)
        temp_h5_filename = os.path.join(TestRadiationSphere.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        radius_group = temp_h5_file.create_group('radius')
        radius_group.create_dataset('time', data=TestRadiationSphere.time)

        new_radiation_sphere = RadiationSphere.create_radiation_sphere(radius_group=radius_group, radius=0)
        self.assertTrue(np.all(new_radiation_sphere.time == TestRadiationSphere.time))

        os.remove(temp_h5_filename)
        os.rmdir(temp_dir)

        temp_dir = os.path.join(TestRadiationSphere.CURR_DIR, "resources/temp")
        os.mkdir(temp_dir)
        temp_h5_filename = os.path.join(TestRadiationSphere.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        radius_group = temp_h5_file.create_group('radius')
        radius_group.create_dataset('time', data=TestRadiationSphere.time)
        modes_group = radius_group.create_group('modes')

        l2_group = modes_group.create_group('l=2')
        m2_group = l2_group.create_group('m=2')
        m2_group.create_dataset('real', data=TestRadiationSphere.real22)
        m2_group.create_dataset('imaginary', data=TestRadiationSphere.imag22)

        l3_group = modes_group.create_group('l=3')
        m3_group = l3_group.create_group('m=3')
        m3_group.create_dataset('real', data=TestRadiationSphere.real33)
        m3_group.create_dataset('imaginary', data=TestRadiationSphere.imag33)
        m1_group = l3_group.create_group('m=1')
        m1_group.create_dataset('real', data=TestRadiationSphere.real31)
        m1_group.create_dataset('imaginary', data=TestRadiationSphere.imag31)

        new_radiation_sphere = RadiationSphere.create_radiation_sphere(radius_group=radius_group, radius=0)
        expected_modes = [(2, 2), (3, 1), (3, 3)]
        generated_modes = sorted(list(new_radiation_sphere._RadiationSphere__raw_modes.keys()))
        self.assertTrue(expected_modes == generated_modes)
        self.assertTrue(isinstance(new_radiation_sphere._RadiationSphere__raw_modes[(2, 2)], RadiationMode))
        self.assertTrue(np.all(TestRadiationSphere.real22 == new_radiation_sphere._RadiationSphere__raw_modes[
            (2, 2)].psi4_real))

        os.remove(temp_h5_filename)
        os.rmdir(temp_dir)

    def test_frame(self):
        from mayawaves.radiation import Frame

        self.radiation_sphere._RadiationSphere__frame = Frame.RAW
        self.assertEqual(Frame.RAW, self.radiation_sphere.frame)

        self.radiation_sphere._RadiationSphere__frame = Frame.COM_CORRECTED
        self.assertEqual(Frame.COM_CORRECTED, self.radiation_sphere.frame)

    def test_set_frame(self):
        from mayawaves.radiation import Frame

        # invalid frame
        self.radiation_sphere.set_frame('coprecessing')
        self.assertEqual(Frame.RAW, self.radiation_sphere._RadiationSphere__frame)

        self.radiation_sphere._RadiationSphere__frame = Frame.COM_CORRECTED
        self.radiation_sphere.set_frame('coprecessing')
        self.assertEqual(Frame.COM_CORRECTED, self.radiation_sphere._RadiationSphere__frame)

        # raw
        self.radiation_sphere.set_frame(Frame.RAW)
        self.assertEqual(Frame.RAW, self.radiation_sphere._RadiationSphere__frame)

        # com
        self.radiation_sphere.set_frame(Frame.COM_CORRECTED)
        self.assertEqual(Frame.COM_CORRECTED, self.radiation_sphere._RadiationSphere__frame)

        # test with time, com
        self.radiation_sphere.set_frame(Frame.RAW)
        time = np.array([1, 2, 3, 4])
        com = np.array([5, 6, 7, 8])
        with patch.object(RadiationSphere, '_set_alpha_beta_for_com_transformation') as mock_set_alpha_beta:
            self.radiation_sphere.set_frame(Frame.COM_CORRECTED, time=time, center_of_mass=com)
            mock_set_alpha_beta.assert_called_once_with(time, com)

        # test setting alpha, beta directly
        self.radiation_sphere.set_frame(Frame.RAW)
        alpha = np.array([1, 2, 3])
        beta = np.array([5, 6, 7])
        with patch.object(RadiationSphere, '_set_alpha_beta_for_com_transformation') as mock_set_alpha_beta:
            self.radiation_sphere.set_frame(Frame.COM_CORRECTED, alpha=alpha, beta=beta)
            mock_set_alpha_beta.assert_not_called()
            self.assertTrue(np.all(alpha == self.radiation_sphere._RadiationSphere__alpha))
            self.assertTrue(np.all(beta == self.radiation_sphere._RadiationSphere__beta))

    def test_modes(self):
        example_radiation_mode = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                               psi4_real=TestRadiationSphere.real22,
                                               psi4_imaginary=TestRadiationSphere.imag22)
        example_com_corrected_radiation_mode = RadiationMode(l=3, m=2, rad=70, time=TestRadiationSphere.time,
                                                             psi4_real=TestRadiationSphere.real22,
                                                             psi4_imaginary=TestRadiationSphere.imag22)
        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_radiation_mode
        }
        self.radiation_sphere._RadiationSphere__com_corrected_modes = {
            (3, 2): example_com_corrected_radiation_mode
        }

        self.assertTrue(self.radiation_sphere._RadiationSphere__raw_modes == self.radiation_sphere.modes)

        # test that it returns correctly based on frame
        from mayawaves.radiation import Frame
        self.radiation_sphere._RadiationSphere__frame = Frame.COM_CORRECTED
        self.assertTrue(self.radiation_sphere._RadiationSphere__com_corrected_modes == self.radiation_sphere.modes)

    def test_raw_modes(self):
        example_radiation_mode = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                               psi4_real=TestRadiationSphere.real22,
                                               psi4_imaginary=TestRadiationSphere.imag22)
        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_radiation_mode
        }
        self.assertTrue(self.radiation_sphere._RadiationSphere__raw_modes == self.radiation_sphere.raw_modes)

    def test_included_modes(self):
        example_radiation_mode = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                               psi4_real=TestRadiationSphere.real22,
                                               psi4_imaginary=TestRadiationSphere.imag22)
        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_radiation_mode,
            (2, 1): example_radiation_mode,
            (3, 2): example_radiation_mode
        }
        self.assertTrue(np.all([(2, 1), (2, 2), (3, 2)] == self.radiation_sphere.included_modes))

    def test_l_max(self):
        example_radiation_mode = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                               psi4_real=TestRadiationSphere.real22,
                                               psi4_imaginary=TestRadiationSphere.imag22)
        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_radiation_mode,
            (2, 1): example_radiation_mode,
            (3, 2): example_radiation_mode
        }
        self.assertTrue(3 == self.radiation_sphere.l_max)

    def test_time(self):
        self.radiation_sphere._RadiationSphere__time = TestRadiationSphere.time
        com_time = [1, 2, 3, 12, 12, 23]
        self.radiation_sphere._RadiationSphere__com_corrected_time = com_time

        self.assertTrue(np.all(self.radiation_sphere._RadiationSphere__time == self.radiation_sphere.time))

        # check when using other frames
        from mayawaves.radiation import Frame
        self.radiation_sphere._RadiationSphere__frame = Frame.COM_CORRECTED
        self.assertTrue(np.all(com_time == self.radiation_sphere.time))

    def test_radius(self):
        self.radiation_sphere._RadiationSphere__radius = 80
        self.assertTrue(np.all(self.radiation_sphere._RadiationSphere__radius == self.radiation_sphere.radius))

    def test_extrapolated(self):
        self.radiation_sphere._RadiationSphere__extrapolated = False
        self.assertTrue(
            np.all(self.radiation_sphere._RadiationSphere__extrapolated == self.radiation_sphere.extrapolated))

        self.radiation_sphere._RadiationSphere__extrapolated = True
        self.assertTrue(
            np.all(self.radiation_sphere._RadiationSphere__extrapolated == self.radiation_sphere.extrapolated))

    def test_get_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        # mode is not in modes
        mode = self.radiation_sphere.get_mode(l=3, m=2)
        self.assertTrue(mode is None)

        # mode is in modes
        mode = self.radiation_sphere.get_mode(l=2, m=2)
        self.assertTrue(example_mode_22 == mode)

        mode = self.radiation_sphere.get_mode(l=3, m=1)
        self.assertTrue(example_mode_31 == mode)

        mode = self.radiation_sphere.get_mode(l=3, m=3)
        self.assertTrue(example_mode_33 == mode)

    def test_get_psi4_real_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        with patch.object(RadiationMode, 'psi4_real', new_callable=PropertyMock,
                          return_value=TestRadiationSphere.real33) as mock_psi4_real:
            # mode is not in modes
            psi4_real = self.radiation_sphere.get_psi4_real_for_mode(l=3, m=2)
            self.assertTrue(psi4_real is None)
            mock_psi4_real.assert_not_called()

        with patch.object(RadiationMode, 'psi4_real', new_callable=PropertyMock,
                          return_value=TestRadiationSphere.real33) as mock_psi4_real:
            # mode is in modes
            psi4_real = self.radiation_sphere.get_psi4_real_for_mode(l=3, m=3)
            mock_psi4_real.assert_called_once()
            self.assertTrue(np.all(TestRadiationSphere.real33 == psi4_real))

    def test_get_psi4_imaginary_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        with patch.object(RadiationMode, 'psi4_imaginary', new_callable=PropertyMock,
                          return_value=TestRadiationSphere.imag33) as mock_psi4_imaginary:
            # mode is not in modes
            psi4_imaginary = self.radiation_sphere.get_psi4_imaginary_for_mode(l=3, m=2)
            self.assertTrue(psi4_imaginary is None)
            mock_psi4_imaginary.assert_not_called()

        with patch.object(RadiationMode, 'psi4_imaginary', new_callable=PropertyMock,
                          return_value=TestRadiationSphere.imag33) as mock_psi4_imaginary:
            # mode is in modes
            psi4_imaginary = self.radiation_sphere.get_psi4_imaginary_for_mode(l=3, m=3)
            mock_psi4_imaginary.assert_called_once()
            self.assertTrue(np.all(TestRadiationSphere.imag33 == psi4_imaginary))

    def test_get_psi4_amplitude_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        amplitude = np.array([5, 4, 3, 2, 1])

        with patch.object(RadiationMode, 'psi4_amplitude', new_callable=PropertyMock,
                          return_value=amplitude) as mock_psi4_amplitude:
            # mode is not in modes
            psi4_amplitude = self.radiation_sphere.get_psi4_amplitude_for_mode(l=3, m=2)
            self.assertTrue(psi4_amplitude is None)
            mock_psi4_amplitude.assert_not_called()

        with patch.object(RadiationMode, 'psi4_amplitude', new_callable=PropertyMock,
                          return_value=amplitude) as mock_psi4_amplitude:
            # mode is in modes
            psi4_amplitude = self.radiation_sphere.get_psi4_amplitude_for_mode(l=3, m=3)
            mock_psi4_amplitude.assert_called_once()
            self.assertTrue(np.all(amplitude == psi4_amplitude))

    def test_get_psi4_phase_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        phase = np.array([5, 4, 3, 2, 1])

        with patch.object(RadiationMode, 'psi4_phase', new_callable=PropertyMock,
                          return_value=phase) as mock_psi4_phase:
            # mode is not in modes
            psi4_phase = self.radiation_sphere.get_psi4_phase_for_mode(l=3, m=2)
            self.assertTrue(psi4_phase is None)
            mock_psi4_phase.assert_not_called()

        with patch.object(RadiationMode, 'psi4_phase', new_callable=PropertyMock,
                          return_value=phase) as mock_psi4_phase:
            # mode is in modes
            psi4_phase = self.radiation_sphere.get_psi4_phase_for_mode(l=3, m=3)
            mock_psi4_phase.assert_called_once()
            self.assertTrue(np.all(phase == psi4_phase))

    def test_get_strain_plus_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        plus = np.array([5, 4, 3, 2, 1])

        with patch.object(RadiationMode, 'strain_plus', new_callable=PropertyMock,
                          return_value=plus) as mock_strain_plus:
            # mode is not in modes
            generated_strain_plus = self.radiation_sphere.get_strain_plus_for_mode(l=3, m=2)
            self.assertTrue(generated_strain_plus is None)
            mock_strain_plus.assert_not_called()

        with patch.object(RadiationMode, 'strain_plus', new_callable=PropertyMock,
                          return_value=plus) as mock_strain_plus:
            # mode is in modes
            generated_strain_plus = self.radiation_sphere.get_strain_plus_for_mode(l=3, m=3)
            mock_strain_plus.assert_called_once()
            self.assertTrue(np.all(plus == generated_strain_plus))

    def test_get_strain_recomposed_at_sky_location(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        h_plus_22 = TestRadiationSphere.real22
        h_plus_31 = TestRadiationSphere.real31
        h_plus_33 = TestRadiationSphere.real33

        h_cross_22 = TestRadiationSphere.imag22
        h_cross_31 = TestRadiationSphere.imag31
        h_cross_33 = TestRadiationSphere.imag33

        h_22 = h_plus_22 - 1j * h_cross_22
        h_31 = h_plus_31 - 1j * h_cross_31
        h_33 = h_plus_33 - 1j * h_cross_33

        self.radiation_sphere.modes[2, 2]._RadiationMode__strain_plus = h_plus_22
        self.radiation_sphere.modes[3, 1]._RadiationMode__strain_plus = h_plus_31
        self.radiation_sphere.modes[3, 3]._RadiationMode__strain_plus = h_plus_33

        self.radiation_sphere.modes[2, 2]._RadiationMode__strain_cross = h_cross_22
        self.radiation_sphere.modes[3, 1]._RadiationMode__strain_cross = h_cross_31
        self.radiation_sphere.modes[3, 3]._RadiationMode__strain_cross = h_cross_33

        y_22 = RadiationMode.ylm(2, 2, np.pi / 2, np.pi / 2)
        y_31 = RadiationMode.ylm(3, 1, np.pi / 2, np.pi / 2)
        y_33 = RadiationMode.ylm(3, 3, np.pi / 2, np.pi / 2)

        h_t = h_22 * y_22 + h_33 * y_33 + h_31 * y_31

        h_plus_returned, h_cross_returned = self.radiation_sphere.get_strain_recomposed_at_sky_location(np.pi / 2,
                                                                                                        np.pi / 2)

        self.assertTrue(np.allclose(h_plus_returned, np.real(h_t), rtol=1e-6))
        self.assertTrue(np.allclose(h_cross_returned, - np.imag(h_t), rtol=1e-6))

        y_22 = RadiationMode.ylm(2, 2, 0, 0)
        y_31 = RadiationMode.ylm(3, 1, 0, 0)
        y_33 = RadiationMode.ylm(3, 3, 0, 0)

        h_t = h_22 * y_22 + h_33 * y_33 + h_31 * y_31

        h_plus_returned, h_cross_returned = self.radiation_sphere.get_strain_recomposed_at_sky_location(0, 0)

        self.assertTrue(np.allclose(h_plus_returned, np.real(h_t), rtol=1e-6))
        self.assertTrue(np.allclose(h_cross_returned, - np.imag(h_t), rtol=1e-6))

        plus = np.array([5, 4, 3, 2, 1])
        cross = np.array([-5, -4, -3, -2, -1])
        with patch.object(RadiationMode, 'strain_plus', new_callable=PropertyMock,
                          return_value=plus) as mock_strain_plus:
            with patch.object(RadiationMode, 'strain_cross', new_callable=PropertyMock,
                              return_value=cross) as mock_strain_cross:
                self.radiation_sphere.get_strain_recomposed_at_sky_location(np.pi / 2, np.pi / 2)
                self.assertEqual(4, mock_strain_plus.call_count)
                self.assertEqual(3, mock_strain_cross.call_count)

    def test_get_strain_cross_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        cross = np.array([5, 4, 3, 2, 1])

        with patch.object(RadiationMode, 'strain_cross', new_callable=PropertyMock,
                          return_value=cross) as mock_strain_cross:
            # mode is not in modes
            generated_strain_cross = self.radiation_sphere.get_strain_cross_for_mode(l=3, m=2)
            self.assertTrue(generated_strain_cross is None)
            mock_strain_cross.assert_not_called()

        with patch.object(RadiationMode, 'strain_cross', new_callable=PropertyMock,
                          return_value=cross) as mock_strain_cross:
            # mode is in modes
            generated_strain_cross = self.radiation_sphere.get_strain_cross_for_mode(l=3, m=3)
            mock_strain_cross.assert_called_once()
            self.assertTrue(np.all(cross == generated_strain_cross))

    def test_get_strain_amplitude_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        amplitude = np.array([5, 4, 3, 2, 1])

        with patch.object(RadiationMode, 'strain_amplitude', new_callable=PropertyMock,
                          return_value=amplitude) as mock_strain_amplitude:
            # mode is not in modes
            generated_strain_amplitude = self.radiation_sphere.get_strain_amplitude_for_mode(l=3, m=2)
            self.assertTrue(generated_strain_amplitude is None)
            mock_strain_amplitude.assert_not_called()

        with patch.object(RadiationMode, 'strain_amplitude', new_callable=PropertyMock,
                          return_value=amplitude) as mock_strain_amplitude:
            # mode is in modes
            generated_strain_amplitude = self.radiation_sphere.get_strain_amplitude_for_mode(l=3, m=3)
            mock_strain_amplitude.assert_called_once()
            self.assertTrue(np.all(amplitude == generated_strain_amplitude))

    def test_get_strain_phase_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        phase = np.array([5, 4, 3, 2, 1])

        with patch.object(RadiationMode, 'strain_phase', new_callable=PropertyMock,
                          return_value=phase) as mock_strain_phase:
            # mode is not in modes
            generated_strain_phase = self.radiation_sphere.get_strain_phase_for_mode(l=3, m=2)
            self.assertTrue(generated_strain_phase is None)
            mock_strain_phase.assert_not_called()

        with patch.object(RadiationMode, 'strain_phase', new_callable=PropertyMock,
                          return_value=phase) as mock_strain_phase:
            # mode is in modes
            generated_strain_phase = self.radiation_sphere.get_strain_phase_for_mode(l=3, m=3)
            mock_strain_phase.assert_called_once()
            self.assertTrue(np.all(phase == generated_strain_phase))

    def test_get_psi4_max_time_for_mode(self):
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        psi4_max_time = 50

        # mode is not in modes
        with patch.object(RadiationMode, 'psi4_max_time', new_callable=PropertyMock,
                          return_value=psi4_max_time) as mock_psi4_max_time:
            generated_psi4_max_time = self.radiation_sphere.get_psi4_max_time_for_mode(l=3, m=2)
            self.assertTrue(generated_psi4_max_time is None)
            mock_psi4_max_time.assert_not_called()

        # mode is in modes
        with patch.object(RadiationMode, 'psi4_max_time', new_callable=PropertyMock,
                          return_value=psi4_max_time) as mock_psi4_max_time:
            generated_psi4_max_time = self.radiation_sphere.get_psi4_max_time_for_mode(l=3, m=3)
            mock_psi4_max_time.assert_called_once()
            self.assertTrue(np.all(psi4_max_time == generated_psi4_max_time))

    def test_get_dEnergy_dt_radiated_for_mode(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(TestRadiationSphere.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle
        radiation_sphere = radiation_bundle.radiation_spheres[75]

        # compare with psi4analysis
        generated_time, generated_denergy_dt_radiated = radiation_sphere.get_dEnergy_dt_radiated(lmin=2, lmax=8)

        psi4_analysis_data = np.loadtxt(os.path.join(TestRadiationSphere.CURR_DIR,
                                                     'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
                                        usecols=5)

        tolerance = 1e-2 * max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_denergy_dt_radiated, atol=tolerance))

        # check that if you limit the modes, you get a smaller amount of energy radiated
        generated_time, generated_denergy_dt_radiated_lmax4 = radiation_sphere.get_dEnergy_dt_radiated(lmin=2, lmax=4)
        self.assertTrue(np.all(generated_denergy_dt_radiated >= generated_denergy_dt_radiated_lmax4))
        self.assertFalse(np.all(generated_denergy_dt_radiated == generated_denergy_dt_radiated_lmax4))

        generated_time, generated_denergy_dt_radiated_lmin3 = radiation_sphere.get_dEnergy_dt_radiated(lmin=3, lmax=8)
        self.assertTrue(np.all(generated_denergy_dt_radiated >= generated_denergy_dt_radiated_lmin3))
        self.assertFalse(np.all(generated_denergy_dt_radiated == generated_denergy_dt_radiated_lmin3))

        placeholder_energy_radiated = np.zeros(generated_time.shape)

        # test calls to radiationmodes based on changing lmin and lmax
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_dEnergy_dt_radiated:
            radiation_sphere.get_dEnergy_dt_radiated(lmin=2, lmax=4)
            self.assertEqual(21, mock_dEnergy_dt_radiated.call_count)

        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_dEnergy_dt_radiated:
            radiation_sphere.get_dEnergy_dt_radiated(lmin=3, lmax=4)
            self.assertEqual(16, mock_dEnergy_dt_radiated.call_count)

        # check that l=2 includes l=2, m=-2, -1, 0, 1, 2 modes
        generated_time, generated_denergy_dt_radiated_l2 = radiation_sphere.get_dEnergy_dt_radiated(lmin=2, lmax=2)
        expected_denergy_dt = np.zeros(generated_time.shape)
        expected_denergy_dt += radiation_sphere.modes[(2, -2)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(2, -1)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(2, 0)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(2, 1)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(2, 2)].dEnergy_dt_radiated
        self.assertTrue(np.all(expected_denergy_dt == generated_denergy_dt_radiated_l2))

        # check that lmax=3 includes l=2, m=-2, -1, 0, 1, 2 modes and l=3, m=-3, -2, -1, 0, 1, 2, 3
        generated_time, generated_denergy_dt_radiated_lmax3 = radiation_sphere.get_dEnergy_dt_radiated(lmin=2, lmax=3)
        expected_denergy_dt = np.zeros(generated_time.shape)
        expected_denergy_dt += radiation_sphere.modes[(2, -2)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(2, -1)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(2, 0)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(2, 1)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(2, 2)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(3, -3)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(3, -2)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(3, -1)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(3, 0)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(3, 1)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(3, 2)].dEnergy_dt_radiated
        expected_denergy_dt += radiation_sphere.modes[(3, 3)].dEnergy_dt_radiated
        self.assertTrue(np.all(expected_denergy_dt == generated_denergy_dt_radiated_lmax3))

        # check all combinations of kwargs
        placeholder_dEnergy_radiated = np.zeros((len(generated_time)))
        # nothing
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dEnergy_radiated) as mock_dEnergy_dt_radiated:
            radiation_sphere.get_dEnergy_dt_radiated()
            self.assertEqual(77, mock_dEnergy_dt_radiated.call_count)

        # lmin only
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dEnergy_radiated) as mock_dEnergy_dt_radiated:
            radiation_sphere.get_dEnergy_dt_radiated(lmin=8)
            self.assertEqual(17, mock_dEnergy_dt_radiated.call_count)

        # lmax only
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dEnergy_radiated) as mock_dEnergy_dt_radiated:
            radiation_sphere.get_dEnergy_dt_radiated(lmax=3)
            self.assertEqual(12, mock_dEnergy_dt_radiated.call_count)

        # lmin and lmax
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dEnergy_radiated) as mock_dEnergy_dt_radiated:
            radiation_sphere.get_dEnergy_dt_radiated(lmin=3, lmax=4)
            self.assertEqual(16, mock_dEnergy_dt_radiated.call_count)

        # l and m
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dEnergy_radiated) as mock_dEnergy_dt_radiated:
            radiation_sphere.get_dEnergy_dt_radiated(l=4, m=2)
            self.assertEqual(1, mock_dEnergy_dt_radiated.call_count)

        # l only
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dEnergy_radiated) as mock_dEnergy_dt_radiated:
            radiation_sphere.get_dEnergy_dt_radiated(l=3)
            self.assertEqual(7, mock_dEnergy_dt_radiated.call_count)

        # m only
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dEnergy_radiated) as mock_dEnergy_dt_radiated:
            result = radiation_sphere.get_dEnergy_dt_radiated(m=2)
            self.assertEqual(0, mock_dEnergy_dt_radiated.call_count)
            self.assertIsNone(result)

        # l and lmin
        with patch.object(RadiationMode, 'dEnergy_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dEnergy_radiated) as mock_dEnergy_dt_radiated:
            result = radiation_sphere.get_dEnergy_dt_radiated(l=2, lmin=8)
            self.assertEqual(0, mock_dEnergy_dt_radiated.call_count)
            self.assertIsNone(result)

    def test_get_energy_radiated_for_mode(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(TestRadiationSphere.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle
        radiation_sphere = radiation_bundle.radiation_spheres[75]

        # compare with psi4analysis
        generated_time, generated_energy_radiated = radiation_sphere.get_energy_radiated(lmin=2, lmax=8)

        psi4_analysis_data = np.loadtxt(os.path.join(TestRadiationSphere.CURR_DIR,
                                                     'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
                                        usecols=1)

        tolerance = 1e-2 * max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_energy_radiated, atol=tolerance))

        # check that if you limit the modes, you get a smaller amount of energy radiated
        generated_time, generated_energy_radiated_lmax4 = radiation_sphere.get_energy_radiated(lmin=2, lmax=4)
        self.assertTrue(np.all(generated_energy_radiated >= generated_energy_radiated_lmax4))
        self.assertFalse(np.all(generated_energy_radiated == generated_energy_radiated_lmax4))

        generated_time, generated_energy_radiated_lmin3 = radiation_sphere.get_energy_radiated(lmin=3, lmax=8)
        self.assertTrue(np.all(generated_energy_radiated >= generated_energy_radiated_lmin3))
        self.assertFalse(np.all(generated_energy_radiated == generated_energy_radiated_lmin3))

        placeholder_energy_radiated = np.zeros(generated_time.shape)

        # test calls to radiationmodes based on changing lmin and lmax
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            radiation_sphere.get_energy_radiated(lmin=2, lmax=4)
            self.assertEqual(21, mock_energy_radiated.call_count)

        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            radiation_sphere.get_energy_radiated(lmin=3, lmax=4)
            self.assertEqual(16, mock_energy_radiated.call_count)

        # check that l=2 includes l=2, m=-2, -1, 0, 1, 2 modes
        generated_time, generated_energy_radiated_l2 = radiation_sphere.get_energy_radiated(lmin=2, lmax=2)
        expected_energy = np.zeros(generated_time.shape)
        expected_energy += radiation_sphere.modes[(2, -2)].energy_radiated
        expected_energy += radiation_sphere.modes[(2, -1)].energy_radiated
        expected_energy += radiation_sphere.modes[(2, 0)].energy_radiated
        expected_energy += radiation_sphere.modes[(2, 1)].energy_radiated
        expected_energy += radiation_sphere.modes[(2, 2)].energy_radiated
        self.assertTrue(np.all(expected_energy == generated_energy_radiated_l2))

        # check that lmax=3 includes l=2, m=-2, -1, 0, 1, 2 modes and l=3, m=-3, -2, -1, 0, 1, 2, 3
        generated_time, generated_energy_radiated_lmax3 = radiation_sphere.get_energy_radiated(lmin=2, lmax=3)
        expected_energy = np.zeros(generated_time.shape)
        expected_energy += radiation_sphere.modes[(2, -2)].energy_radiated
        expected_energy += radiation_sphere.modes[(2, -1)].energy_radiated
        expected_energy += radiation_sphere.modes[(2, 0)].energy_radiated
        expected_energy += radiation_sphere.modes[(2, 1)].energy_radiated
        expected_energy += radiation_sphere.modes[(2, 2)].energy_radiated
        expected_energy += radiation_sphere.modes[(3, -3)].energy_radiated
        expected_energy += radiation_sphere.modes[(3, -2)].energy_radiated
        expected_energy += radiation_sphere.modes[(3, -1)].energy_radiated
        expected_energy += radiation_sphere.modes[(3, 0)].energy_radiated
        expected_energy += radiation_sphere.modes[(3, 1)].energy_radiated
        expected_energy += radiation_sphere.modes[(3, 2)].energy_radiated
        expected_energy += radiation_sphere.modes[(3, 3)].energy_radiated
        self.assertTrue(np.all(expected_energy == generated_energy_radiated_lmax3))

        # check all combinations of kwargs
        placeholder_energy_radiated = np.zeros((len(generated_time)))
        # nothing
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            radiation_sphere.get_energy_radiated()
            self.assertEqual(77, mock_energy_radiated.call_count)

        # lmin only
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            radiation_sphere.get_energy_radiated(lmin=8)
            self.assertEqual(17, mock_energy_radiated.call_count)

        # lmax only
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            radiation_sphere.get_energy_radiated(lmax=3)
            self.assertEqual(12, mock_energy_radiated.call_count)

        # lmin and lmax
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            radiation_sphere.get_energy_radiated(lmin=3, lmax=4)
            self.assertEqual(16, mock_energy_radiated.call_count)

        # l and m
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            radiation_sphere.get_energy_radiated(l=4, m=2)
            self.assertEqual(1, mock_energy_radiated.call_count)

        # l only
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            radiation_sphere.get_energy_radiated(l=3)
            self.assertEqual(7, mock_energy_radiated.call_count)

        # m only
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            result = radiation_sphere.get_energy_radiated(m=2)
            self.assertEqual(0, mock_energy_radiated.call_count)
            self.assertIsNone(result[0])
            self.assertIsNone(result[1])

        # l and lmin
        with patch.object(RadiationMode, 'energy_radiated', new_callable=PropertyMock,
                          return_value=placeholder_energy_radiated) as mock_energy_radiated:
            result = radiation_sphere.get_energy_radiated(l=2, lmin=8)
            self.assertEqual(0, mock_energy_radiated.call_count)
            self.assertIsNone(result[0])
            self.assertIsNone(result[1])

    def test_get_dP_dt_radiated(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(TestRadiationSphere.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle
        radiation_sphere = radiation_bundle.radiation_spheres[75]

        # compare with psi4analysis
        generated_time, generated_dP_dt_radiated = radiation_sphere.get_dP_dt_radiated(lmin=2, lmax=8)

        psi4_analysis_data = np.loadtxt(os.path.join(TestRadiationSphere.CURR_DIR,
                                                     'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
                                        usecols=(6, 7, 8))

        tolerance = 5e-2 * np.max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_dP_dt_radiated, atol=tolerance))

        placeholder_dP_radiated = np.zeros((len(generated_time), 3))

        # test calls to radiationmodes based on changing lmin and lmax
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            radiation_sphere.get_dP_dt_radiated(lmin=2, lmax=4)
            self.assertEqual(21, mock_dP_dt_radiated.call_count)

        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            radiation_sphere.get_dP_dt_radiated(lmin=3, lmax=4)
            self.assertEqual(16, mock_dP_dt_radiated.call_count)

        # check that l=2 includes l=2, m=-2, -1, 0, 1, 2 modes
        generated_time, generated_dP_dt_radiated_l2 = radiation_sphere.get_dP_dt_radiated(lmin=2, lmax=2)
        expected_dP_dt = np.zeros((len(generated_time), 3))
        expected_dP_dt += radiation_sphere.modes[(2, -2)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(2, -1)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(2, 0)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(2, 1)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(2, 2)].dP_dt_radiated
        self.assertTrue(np.all(expected_dP_dt == generated_dP_dt_radiated_l2))

        # check that lmax=3 includes l=2, m=-2, -1, 0, 1, 2 modes and l=3, m=-3, -2, -1, 0, 1, 2, 3
        generated_time, generated_dP_dt_radiated_lmax3 = radiation_sphere.get_dP_dt_radiated(lmin=2, lmax=3)
        expected_dP_dt = np.zeros((len(generated_time), 3))
        expected_dP_dt += radiation_sphere.modes[(2, -2)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(2, -1)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(2, 0)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(2, 1)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(2, 2)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(3, -3)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(3, -2)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(3, -1)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(3, 0)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(3, 1)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(3, 2)].dP_dt_radiated
        expected_dP_dt += radiation_sphere.modes[(3, 3)].dP_dt_radiated
        self.assertTrue(np.all(expected_dP_dt == generated_dP_dt_radiated_lmax3))

        # check all combinations of kwargs
        placeholder_dP_radiated = np.zeros((len(generated_time), 3))
        # nothing
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            radiation_sphere.get_dP_dt_radiated()
            self.assertEqual(77, mock_dP_dt_radiated.call_count)

        # lmin only
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            radiation_sphere.get_dP_dt_radiated(lmin=8)
            self.assertEqual(17, mock_dP_dt_radiated.call_count)

        # lmax only
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            radiation_sphere.get_dP_dt_radiated(lmax=3)
            self.assertEqual(12, mock_dP_dt_radiated.call_count)

        # lmin and lmax
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            radiation_sphere.get_dP_dt_radiated(lmin=3, lmax=4)
            self.assertEqual(16, mock_dP_dt_radiated.call_count)

        # l and m
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            radiation_sphere.get_dP_dt_radiated(l=4, m=2)
            self.assertEqual(1, mock_dP_dt_radiated.call_count)

        # l only
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            radiation_sphere.get_dP_dt_radiated(l=3)
            self.assertEqual(7, mock_dP_dt_radiated.call_count)

        # m only
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            result = radiation_sphere.get_dP_dt_radiated(m=2)
            self.assertEqual(0, mock_dP_dt_radiated.call_count)
            self.assertIsNone(result)

        # l and lmin
        with patch.object(RadiationMode, 'dP_dt_radiated', new_callable=PropertyMock,
                          return_value=placeholder_dP_radiated) as mock_dP_dt_radiated:
            result = radiation_sphere.get_dP_dt_radiated(l=2, lmin=8)
            self.assertEqual(0, mock_dP_dt_radiated.call_count)
            self.assertIsNone(result)

    def test_get_linear_momentum_radiated(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(TestRadiationSphere.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle
        radiation_sphere = radiation_bundle.radiation_spheres[75]

        # compare with psi4analysis
        generated_time, generated_linear_momentum_radiated = radiation_sphere.get_linear_momentum_radiated(lmin=2,
                                                                                                           lmax=8)

        psi4_analysis_data = np.loadtxt(os.path.join(TestRadiationSphere.CURR_DIR,
                                                     'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
                                        usecols=(2, 3, 4))

        tolerance = 5e-2 * np.max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_linear_momentum_radiated, atol=tolerance))

        placeholder_linear_momentum_radiated = np.zeros((len(generated_time), 3))

        # test calls to radiationmodes based on changing lmin and lmax
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            radiation_sphere.get_linear_momentum_radiated(lmin=2, lmax=4)
            self.assertEqual(21, mock_linear_momentum_radiated.call_count)

        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            radiation_sphere.get_linear_momentum_radiated(lmin=3, lmax=4)
            self.assertEqual(16, mock_linear_momentum_radiated.call_count)

        # check that l=2 includes l=2, m=-2, -1, 0, 1, 2 modes
        generated_time, generated_linear_momentum_radiated_l2 = radiation_sphere.get_linear_momentum_radiated(lmin=2,
                                                                                                              lmax=2)
        expected_linear_momentum = np.zeros((len(generated_time), 3))
        expected_linear_momentum += radiation_sphere.modes[(2, -2)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(2, -1)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(2, 0)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(2, 1)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(2, 2)].linear_momentum_radiated
        self.assertTrue(np.all(expected_linear_momentum == generated_linear_momentum_radiated_l2))

        # check that lmax=3 includes l=2, m=-2, -1, 0, 1, 2 modes and l=3, m=-3, -2, -1, 0, 1, 2, 3
        generated_time, generated_linear_momentum_radiated_lmax3 = radiation_sphere.get_linear_momentum_radiated(lmin=2,
                                                                                                                 lmax=3)
        expected_linear_momentum = np.zeros((len(generated_time), 3))
        expected_linear_momentum += radiation_sphere.modes[(2, -2)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(2, -1)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(2, 0)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(2, 1)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(2, 2)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(3, -3)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(3, -2)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(3, -1)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(3, 0)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(3, 1)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(3, 2)].linear_momentum_radiated
        expected_linear_momentum += radiation_sphere.modes[(3, 3)].linear_momentum_radiated
        self.assertTrue(np.all(expected_linear_momentum == generated_linear_momentum_radiated_lmax3))

        # check all combinations of kwargs
        placeholder_linear_momentum_radiated = np.zeros((len(generated_time), 3))
        # nothing
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            radiation_sphere.get_linear_momentum_radiated()
            self.assertEqual(77, mock_linear_momentum_radiated.call_count)

        # lmin only
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            radiation_sphere.get_linear_momentum_radiated(lmin=8)
            self.assertEqual(17, mock_linear_momentum_radiated.call_count)

        # lmax only
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            radiation_sphere.get_linear_momentum_radiated(lmax=3)
            self.assertEqual(12, mock_linear_momentum_radiated.call_count)

        # lmin and lmax
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            radiation_sphere.get_linear_momentum_radiated(lmin=3, lmax=4)
            self.assertEqual(16, mock_linear_momentum_radiated.call_count)

        # l and m
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            radiation_sphere.get_linear_momentum_radiated(l=4, m=2)
            self.assertEqual(1, mock_linear_momentum_radiated.call_count)

        # l only
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            radiation_sphere.get_linear_momentum_radiated(l=3)
            self.assertEqual(7, mock_linear_momentum_radiated.call_count)

        # m only
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            result = radiation_sphere.get_linear_momentum_radiated(m=2)
            self.assertEqual(0, mock_linear_momentum_radiated.call_count)
            self.assertIsNone(result)

        # l and lmin
        with patch.object(RadiationMode, 'linear_momentum_radiated', new_callable=PropertyMock,
                          return_value=placeholder_linear_momentum_radiated) as mock_linear_momentum_radiated:
            result = radiation_sphere.get_linear_momentum_radiated(l=2, lmin=8)
            self.assertEqual(0, mock_linear_momentum_radiated.call_count)
            self.assertIsNone(result)

    def test_get_extrapolated_sphere(self):
        from mayawaves.radiation import Frame
        example_mode_22 = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real22,
                                        psi4_imaginary=TestRadiationSphere.imag22)
        example_mode_31 = RadiationMode(l=3, m=1, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real31,
                                        psi4_imaginary=TestRadiationSphere.imag31)
        example_mode_33 = RadiationMode(l=3, m=3, rad=75, time=TestRadiationSphere.time,
                                        psi4_real=TestRadiationSphere.real33,
                                        psi4_imaginary=TestRadiationSphere.imag33)
        extrapolated_example_mode = RadiationMode(l=2, m=2, rad=75, time=TestRadiationSphere.time,
                                                  psi4_real=TestRadiationSphere.real22,
                                                  psi4_imaginary=TestRadiationSphere.imag22, extrapolated=True)

        self.radiation_sphere._RadiationSphere__raw_modes = {
            (2, 2): example_mode_22,
            (3, 1): example_mode_31,
            (3, 3): example_mode_33
        }

        # if already extrapolated, return self
        self.radiation_sphere._RadiationSphere__extrapolated = True
        extrapolated_radiation_sphere = self.radiation_sphere.get_extrapolated_sphere()
        self.assertTrue(self.radiation_sphere == extrapolated_radiation_sphere)

        # if not already extrapolated
        self.radiation_sphere._RadiationSphere__extrapolated = False
        with patch.object(RadiationMode, 'get_mode_with_extrapolated_radius',
                          return_value=extrapolated_example_mode) as mock_get_mode_with_extrapolated_radius:
            extrapolated_radiation_sphere = self.radiation_sphere.get_extrapolated_sphere()
            self.assertTrue(isinstance(extrapolated_radiation_sphere, RadiationSphere))
            self.assertTrue(extrapolated_radiation_sphere._RadiationSphere__extrapolated)

            expected_included_modes = [(2, 2), (3, 1), (3, 3)]
            generated_included_modes = sorted(list(extrapolated_radiation_sphere._RadiationSphere__raw_modes.keys()))
            self.assertTrue(expected_included_modes == generated_included_modes)
            for mode in generated_included_modes:
                self.assertTrue(
                    isinstance(extrapolated_radiation_sphere._RadiationSphere__raw_modes[mode], RadiationMode))
                self.assertTrue(
                    extrapolated_radiation_sphere._RadiationSphere__raw_modes[mode]._RadiationMode__extrapolated)
            self.assertEqual(Frame.RAW, extrapolated_radiation_sphere.frame)

        # test frame
        self.radiation_sphere._RadiationSphere__extrapolated = False
        self.radiation_sphere.set_frame(Frame.COM_CORRECTED, alpha=np.array([0.1, 0.1, 0.1]),
                                        beta=np.array([0.2, 0.2, 0.2]))
        with patch.object(RadiationMode, 'get_mode_with_extrapolated_radius',
                          return_value=extrapolated_example_mode) as mock_get_mode_with_extrapolated_radius:
            extrapolated_radiation_sphere = self.radiation_sphere.get_extrapolated_sphere()
            self.assertTrue(isinstance(extrapolated_radiation_sphere, RadiationSphere))
            self.assertTrue(extrapolated_radiation_sphere._RadiationSphere__extrapolated)

            expected_included_modes = [(2, 2), (3, 1), (3, 3)]
            generated_included_modes = sorted(list(extrapolated_radiation_sphere._RadiationSphere__raw_modes.keys()))
            self.assertTrue(expected_included_modes == generated_included_modes)
            for mode in generated_included_modes:
                self.assertTrue(
                    isinstance(extrapolated_radiation_sphere._RadiationSphere__raw_modes[mode], RadiationMode))
                self.assertTrue(
                    extrapolated_radiation_sphere._RadiationSphere__raw_modes[mode]._RadiationMode__extrapolated)
            self.assertEqual(Frame.COM_CORRECTED, extrapolated_radiation_sphere.frame)
            self.assertTrue(np.all(np.array([0.1, 0.1, 0.1]) == extrapolated_radiation_sphere._RadiationSphere__alpha))
            self.assertTrue(np.all(np.array([0.2, 0.2, 0.2]) == extrapolated_radiation_sphere._RadiationSphere__beta))

    def test__scri_waveform_modes_object(self):
        from spherical_functions import LM_index
        coalescence = Coalescence(os.path.join(TestRadiationSphere.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))
        test_radiation_sphere = coalescence.radiationbundle.radiation_spheres[75]

        scri_waveform_modes_object = test_radiation_sphere._scri_waveform_modes_object()

        # go through each mode and see if it has the same data
        for l in range(2, coalescence.l_max + 1):
            for m in range(-l, l + 1):
                data_lm = scri_waveform_modes_object.data[:, LM_index(l, m, scri_waveform_modes_object.ell_min)]
                scri_time = scri_waveform_modes_object.t
                scri_strain_plus = np.real(data_lm)
                scri_strain_cross = -1 * np.imag(data_lm)
                raw_time, raw_strain_plus, raw_strain_cross = coalescence.strain_for_mode(l=l, m=m,
                                                                                          extraction_radius=75)
                cut_index = np.argmax(raw_time > 150)
                maya_time = raw_time[cut_index:]
                maya_strain_plus = raw_strain_plus[cut_index:]
                maya_strain_cross = raw_strain_cross[cut_index:]
                self.assertTrue(np.allclose(maya_time, scri_time, rtol=1e-6))
                self.assertTrue(np.allclose(maya_strain_plus, scri_strain_plus, rtol=1e-6))
                self.assertTrue(np.allclose(maya_strain_cross, scri_strain_cross, rtol=1e-6))

    def test__set_alpha_beta_for_com_transformation(self):
        # check if alpha and beta minimize eq 5 of https://arxiv.org/abs/1904.04842
        coalescence = Coalescence(os.path.join(TestRadiationSphere.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))
        time, com = coalescence.center_of_mass
        coalescence.radiationbundle.radiation_spheres[75]._set_alpha_beta_for_com_transformation(time, com)
        computed_alpha = coalescence.radiationbundle.radiation_spheres[75]._RadiationSphere__alpha
        computed_beta = coalescence.radiationbundle.radiation_spheres[75]._RadiationSphere__beta

        time = time + 75
        t_max = np.max(time)
        ti = 0.1 * t_max
        tf = 0.9 * t_max
        ti_iter = np.argmax(time > ti)
        tf_iter = np.argmax(time > tf)
        time = time[ti_iter: tf_iter]
        com = com[ti_iter: tf_iter]

        def func_of_alpha_beta(x):
            ax, ay, az, bx, by, bz = x
            alpha = np.array([ax, ay, az])
            beta = np.array([bx, by, bz])
            alpha_plus_beta_t = alpha.reshape(1, 3) + beta.reshape(1, 3) * time.reshape(len(time), 1)
            alpha_plus_beta_t = alpha_plus_beta_t.reshape(len(alpha_plus_beta_t), 3)
            integrand = np.linalg.norm(com - alpha_plus_beta_t, axis=1) ** 2
            return np.trapz(integrand, time)

        result = scipy.optimize.minimize(func_of_alpha_beta, x0=(0, 0, 0, 0, 0, 0), method='Nelder-Mead')
        minimized_alpha = result.x[:3]
        minimized_beta = result.x[3:]
        self.assertTrue(np.allclose(minimized_alpha, computed_alpha, atol=5e-3))
        self.assertTrue(np.allclose(minimized_beta, computed_beta, atol=1e-3))

    def test__generate_com_corrected_modes(self):
        coalescence = Coalescence(os.path.join(TestRadiationSphere.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))
        incl_modes = coalescence.included_modes
        test_radiation_sphere = coalescence.radiationbundle.radiation_spheres[75]
        time, center_of_mass = coalescence.center_of_mass
        test_radiation_sphere._set_alpha_beta_for_com_transformation(time, center_of_mass)
        test_radiation_sphere._generate_com_corrected_modes()

        # com corrected time set and starts close to but greater than 75
        generated_com_corrected_time = test_radiation_sphere._RadiationSphere__com_corrected_time
        self.assertIsNotNone(generated_com_corrected_time)
        self.assertTrue(generated_com_corrected_time[0] >= 150)
        self.assertTrue(np.isclose(generated_com_corrected_time[0], 150, atol=1))

        # com corrected modes set
        com_corrected_modes = test_radiation_sphere._RadiationSphere__com_corrected_modes
        self.assertIsNotNone(com_corrected_modes)

        # all the modes from raw are represented
        self.assertEqual(sorted(incl_modes), sorted(list(com_corrected_modes.keys())))

        max_22_amp = max(test_radiation_sphere._RadiationSphere__raw_modes[(2, 2)].strain_amplitude)

        # strains are within ~10 percent of each other but not identical
        for mode_key in incl_modes:
            raw_mode = test_radiation_sphere._RadiationSphere__raw_modes[mode_key]
            com_corrected_mode = com_corrected_modes[mode_key]
            raw_time = raw_mode.time
            raw_strain_plus = raw_mode.strain_plus
            com_time = com_corrected_mode.time
            com_strain_plus = com_corrected_mode.strain_plus

            time_array = np.linspace(max(raw_time[0], com_time[0]), min(raw_time[-1], com_time[-1]), 1000)

            interpolated_raw_strain_plus = np.interp(time_array, raw_time, raw_strain_plus)
            interpolated_com_strain_plus = np.interp(time_array, com_time, com_strain_plus)

            self.assertFalse(np.all(interpolated_raw_strain_plus == interpolated_com_strain_plus))

            max_amp = max(raw_mode.strain_amplitude)
            if max_amp > 1e-2 * max_22_amp:
                tol = 0.1 * max_amp
                self.assertTrue(np.allclose(interpolated_raw_strain_plus, interpolated_com_strain_plus, atol=tol))
