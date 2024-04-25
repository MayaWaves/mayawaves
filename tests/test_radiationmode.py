from unittest import TestCase
from unittest.mock import patch, PropertyMock
import h5py
import numpy as np
import os
import scipy.integrate
from mayawaves.radiation import RadiationMode, RadiationSphere


class TestRadiationMode(TestCase):

    def setUp(self) -> None:
        self.time = np.array([1, 2, 3, 4, 5])
        self.psi4_real = np.array([6, 7, 8, 9, 10])
        self.psi4_imaginary = np.array([11, 12, 13, 14, 15])
        self.radiation_mode = RadiationMode(l=3, m=1, rad=75, time=self.time, psi4_real=self.psi4_real,
                                            psi4_imaginary=self.psi4_imaginary)
        self.strain_plus = np.array([21, 22, 23, 24, 25])
        self.strain_cross = np.array([31, 32, 33, 34, 35])
        self.psi4_phase = np.array([41, 42, 43, 44, 45])
        self.psi4_amplitude = np.array([51, 52, 53, 54, 55])

        self.CURR_DIR = os.path.dirname(__file__)

    def test_create_from_dict(self):
        # if real and imaginary and both in the dictionary
        temp_dir = os.path.join(self.CURR_DIR, "resources/temp")
        os.mkdir(temp_dir)
        temp_h5_filename = os.path.join(self.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        psi4_group = temp_h5_file.create_group('psi4')
        psi4_group.create_dataset('real', data=self.psi4_real)
        psi4_group.create_dataset('imaginary', data=self.psi4_imaginary)

        new_radiation_mode = RadiationMode.create_radiation_mode(psi4_group=psi4_group, l=2, m=2, rad=75,
                                                                 time=self.time)
        self.assertTrue(isinstance(new_radiation_mode, RadiationMode))
        self.assertTrue(np.all(self.time == new_radiation_mode._RadiationMode__time))
        self.assertTrue(np.all(self.psi4_real == self.psi4_real))
        self.assertTrue(np.all(self.psi4_imaginary == self.psi4_imaginary))
        self.assertTrue(2 == new_radiation_mode._RadiationMode__l_value)
        self.assertTrue(2 == new_radiation_mode._RadiationMode__m_value)
        self.assertTrue(75 == new_radiation_mode._RadiationMode__radius)
        self.assertFalse(new_radiation_mode._RadiationMode__extrapolated)

        os.remove(temp_h5_filename)
        os.rmdir(temp_dir)

    def test_l_value(self):
        self.assertTrue(self.radiation_mode._RadiationMode__l_value == self.radiation_mode.l_value)
        self.radiation_mode._RadiationMode__l_value = 2
        self.assertTrue(self.radiation_mode._RadiationMode__l_value == self.radiation_mode.l_value)

    def test_m_value(self):
        self.assertTrue(self.radiation_mode._RadiationMode__m_value == self.radiation_mode.m_value)
        self.radiation_mode._RadiationMode__m_value = 2
        self.assertTrue(self.radiation_mode._RadiationMode__m_value == self.radiation_mode.m_value)

    def test_radius(self):
        self.assertTrue(self.radiation_mode._RadiationMode__radius == self.radiation_mode.radius)
        self.radiation_mode._RadiationMode__radius = 70
        self.assertTrue(self.radiation_mode._RadiationMode__radius == self.radiation_mode.radius)

    def test_time(self):
        self.assertTrue(np.all(self.radiation_mode._RadiationMode__time == self.radiation_mode.time))

    def test_extrapolated(self):
        self.assertTrue(self.radiation_mode._RadiationMode__extrapolated == self.radiation_mode.extrapolated)
        self.radiation_mode._RadiationMode__extrapolated = True
        self.assertTrue(self.radiation_mode._RadiationMode__extrapolated == self.radiation_mode.extrapolated)

    def test_psi4_real(self):
        self.assertTrue(np.all(self.radiation_mode._RadiationMode__psi4_real == self.radiation_mode.psi4_real))

        # test when not set and derivative has to be taken
        self.radiation_mode._RadiationMode__psi4_real = None
        example_strain_plus = [1, 2, 4, 7, 11]
        example_time = [1, 2, 4, 5, 6]
        self.radiation_mode._RadiationMode__strain_plus = example_strain_plus
        with patch.object(RadiationMode, 'time', new_callable=PropertyMock, return_value=example_time) as mock_time:
            with patch.object(RadiationMode, 'radius', new_callable=PropertyMock, return_value=10) as mock_radius:
                expected_psi4_real = np.gradient(np.gradient(example_strain_plus, example_time), example_time) / 10
                self.assertTrue(np.all(expected_psi4_real == self.radiation_mode.psi4_real))

    def test_psi4_imaginary(self):
        self.assertTrue(
            np.all(self.radiation_mode._RadiationMode__psi4_imaginary == self.radiation_mode.psi4_imaginary))

        # test when not set and derivative has to be taken
        self.radiation_mode._RadiationMode__psi4_imaginary = None
        example_strain_cross = [1, 2, 4, 7, 11]
        example_time = [1, 2, 4, 5, 6]
        self.radiation_mode._RadiationMode__strain_cross = example_strain_cross
        with patch.object(RadiationMode, 'time', new_callable=PropertyMock, return_value=example_time) as mock_time:
            with patch.object(RadiationMode, 'radius', new_callable=PropertyMock, return_value=10) as mock_radius:
                expected_psi4_imaginary = -1 * np.gradient(np.gradient(example_strain_cross, example_time),
                                                           example_time) / 10
                self.assertTrue(np.all(expected_psi4_imaginary == self.radiation_mode.psi4_imaginary))

    def test_psi4_amplitude(self):
        # amplitude is None
        with patch.object(RadiationMode, 'psi4_real', new_callable=PropertyMock,
                          return_value=self.psi4_real) as mock_psi4_real:
            with patch.object(RadiationMode, 'psi4_imaginary', new_callable=PropertyMock,
                              return_value=self.psi4_imaginary) as mock_psi4_imaginary:
                expected_amplitude = np.sqrt(np.power(self.psi4_real, 2) + np.power(self.psi4_imaginary, 2))
                generated_amplitude = self.radiation_mode.psi4_amplitude
                self.assertTrue(np.all(expected_amplitude == generated_amplitude))
                mock_psi4_real.assert_called_once()
                mock_psi4_imaginary.assert_called_once()

        # amplitude is set
        self.assertTrue(
            np.all(self.radiation_mode._RadiationMode__psi4_amplitude == self.radiation_mode.psi4_amplitude))

    def test_psi4_phase(self):
        # phase is None
        with patch.object(RadiationMode, 'psi4_real', new_callable=PropertyMock,
                          return_value=self.psi4_real) as mock_psi4_real:
            with patch.object(RadiationMode, 'psi4_imaginary', new_callable=PropertyMock,
                              return_value=self.psi4_imaginary) as mock_psi4_imaginary:
                expected_phase = -1 * np.unwrap(np.arctan2(self.psi4_imaginary, self.psi4_real))
                generated_phase = self.radiation_mode.psi4_phase
                self.assertTrue(np.all(expected_phase == generated_phase))
                mock_psi4_real.assert_called_once()
                mock_psi4_imaginary.assert_called_once()

        # phase is set
        self.assertTrue(
            np.all(self.radiation_mode._RadiationMode__psi4_phase == self.radiation_mode.psi4_phase))

    def test_strain_plus(self):
        # strain is none
        def compute_and_store_strain_side_effect():
            self.radiation_mode._RadiationMode__strain_plus = self.strain_plus

        with patch.object(RadiationMode, 'compute_and_store_strain',
                          side_effect=compute_and_store_strain_side_effect) as mock_compute_and_store_strain:
            self.assertTrue(np.all(self.strain_plus == self.radiation_mode.strain_plus))
            mock_compute_and_store_strain.assert_called_once()

        # strain is set
        self.assertTrue(np.all(self.radiation_mode._RadiationMode__strain_plus == self.radiation_mode.strain_plus))

    def test_strain_cross(self):
        # strain is none
        def compute_and_store_strain_side_effect():
            self.radiation_mode._RadiationMode__strain_cross = self.strain_cross

        with patch.object(RadiationMode, 'compute_and_store_strain',
                          side_effect=compute_and_store_strain_side_effect) as mock_compute_and_store_strain:
            self.assertTrue(np.all(self.strain_cross == self.radiation_mode.strain_cross))
            mock_compute_and_store_strain.assert_called_once()

        # strain is set
        self.assertTrue(
            np.all(self.radiation_mode._RadiationMode__strain_cross == self.radiation_mode.strain_cross))

    # TODO test of actual amplitude
    def test_strain_amplitude(self):
        # amplitude is None
        with patch.object(RadiationMode, 'strain_plus', new_callable=PropertyMock,
                          return_value=self.strain_plus) as mock_strain_plus:
            with patch.object(RadiationMode, 'strain_cross', new_callable=PropertyMock,
                              return_value=self.strain_cross) as mock_strain_cross:
                expected_amplitude = np.sqrt(np.power(self.strain_plus, 2) + np.power(self.strain_cross, 2))
                generated_amplitude = self.radiation_mode.strain_amplitude
                self.assertTrue(np.all(expected_amplitude == generated_amplitude))
                mock_strain_plus.assert_called_once()
                mock_strain_cross.assert_called_once()

        # amplitude is set
        self.assertTrue(
            np.all(self.radiation_mode._RadiationMode__strain_amplitude == self.radiation_mode.strain_amplitude))

    def test_strain_phase(self):
        # phase is None
        with patch.object(RadiationMode, 'strain_plus', new_callable=PropertyMock,
                          return_value=self.strain_plus) as mock_strain_plus:
            with patch.object(RadiationMode, 'strain_cross', new_callable=PropertyMock,
                              return_value=self.strain_cross) as mock_strain_cross:
                expected_phase = -1 * np.unwrap(np.arctan2(self.strain_cross, self.strain_plus))
                generated_phase = self.radiation_mode.strain_phase
                self.assertTrue(np.all(expected_phase == generated_phase))
                mock_strain_plus.assert_called_once()
                mock_strain_cross.assert_called_once()

        # phase is set
        self.assertTrue(
            np.all(self.radiation_mode._RadiationMode__strain_phase == self.radiation_mode.strain_phase))

    def test_radiation_sphere(self):
        # getting the extraction sphere when not set
        self.assertTrue(self.radiation_mode.radiation_sphere is None)

        # setting the extraction sphere
        radiation_sphere = RadiationSphere(mode_dict={}, time=np.array([]), radius=0)
        self.radiation_mode.radiation_sphere = radiation_sphere
        self.assertTrue(radiation_sphere == self.radiation_mode._RadiationMode__radiation_sphere)

        # getting the extraction sphere
        self.assertTrue(self.radiation_mode._RadiationMode__radiation_sphere == self.radiation_mode.radiation_sphere)

    def test_psi4_omega(self):
        with patch.object(RadiationMode, 'psi4_phase', new_callable=PropertyMock,
                          return_value=self.psi4_phase) as mock_psi4_phase:
            with patch.object(RadiationMode, 'time', new_callable=PropertyMock,
                              return_value=self.time) as mock_time:
                expected_omega = np.gradient(self.psi4_phase, self.time)
                generated_omega = self.radiation_mode.psi4_omega
                self.assertTrue(np.all(expected_omega == generated_omega))
                mock_psi4_phase.assert_called_once()
                mock_time.assert_called_once()

    def test_omega_start(self):
        # if the time data goes past radius + 75
        time = np.arange(0, 1000, 10)
        psi4_omega = 0.1 + 0.0001 * np.power(time, 2)
        expected_start_omega = psi4_omega[np.argmax(time > 150)]
        with patch.object(RadiationMode, 'psi4_omega', new_callable=PropertyMock,
                          return_value=psi4_omega) as mock_psi4_omega:
            with patch.object(RadiationMode, 'time', new_callable=PropertyMock,
                              return_value=time) as mock_time:
                with patch.object(RadiationMode, 'radius', new_callable=PropertyMock,
                                  return_value=75) as mock_radius:
                    generated_start_omega = self.radiation_mode.omega_start
                    self.assertTrue(np.isclose(expected_start_omega, generated_start_omega, rtol=0.1))
                    mock_psi4_omega.assert_called_once()
                    self.assertEqual(5, mock_time.call_count)
                    mock_radius.assert_called_once()

        # if the time data does not go past radius + 75
        time = np.arange(0, 100, 1)
        psi4_omega = 0.1 + 0.0001 * np.power(time, 2)
        expected_start_omega = psi4_omega[0]
        with patch.object(RadiationMode, 'psi4_omega', new_callable=PropertyMock,
                          return_value=psi4_omega) as mock_psi4_omega:
            with patch.object(RadiationMode, 'time', new_callable=PropertyMock,
                              return_value=time) as mock_time:
                with patch.object(RadiationMode, 'radius', new_callable=PropertyMock,
                                  return_value=75) as mock_radius:
                    generated_start_omega = self.radiation_mode.omega_start
                    self.assertTrue(np.isclose(expected_start_omega, generated_start_omega, rtol=0.1))
                    mock_psi4_omega.assert_called_once()
                    self.assertTrue(4, mock_time.call_count)
                    mock_radius.assert_called_once()

    def test_psi4_max_time(self):
        time = np.arange(0, 1000, 10)
        psi4_amplitude = 1000 - 0.001 * np.power(time - 400, 2)
        with patch.object(RadiationMode, 'time', new_callable=PropertyMock, return_value=time) as mock_time:
            with patch.object(RadiationMode, 'psi4_amplitude', new_callable=PropertyMock,
                              return_value=psi4_amplitude) as mock_psi4_amplitude:
                expected_max_time = 400
                self.assertTrue(expected_max_time == self.radiation_mode.psi4_max_time)
                mock_time.assert_called_once()
                mock_psi4_amplitude.assert_called_once()

    def test_h_plus_dot(self):
        time = np.arange(0, 1000, 10)
        strain_plus = np.sin(0.01 * time)
        expected_hdot = 0.01 * np.cos(0.01 * time)

        with patch.object(RadiationMode, 'time', new_callable=PropertyMock, return_value=time) as mock_time:
            with patch.object(RadiationMode, 'strain_plus', new_callable=PropertyMock,
                              return_value=strain_plus) as mock_strain_plus:
                generated_h_plus_dot = self.radiation_mode.h_plus_dot
                self.assertTrue(np.all(np.isclose(expected_hdot, generated_h_plus_dot, atol=1e-3)))
                mock_time.assert_called_once()
                mock_strain_plus.assert_called_once()

    def test_h_cross_dot(self):
        time = np.arange(0, 1000, 10)
        strain_cross = np.sin(0.01 * time)
        expected_hdot = 0.01 * np.cos(0.01 * time)

        with patch.object(RadiationMode, 'time', new_callable=PropertyMock, return_value=time) as mock_time:
            with patch.object(RadiationMode, 'strain_cross', new_callable=PropertyMock,
                              return_value=strain_cross) as mock_strain_cross:
                generated_h_cross_dot = self.radiation_mode.h_cross_dot
                self.assertTrue(np.all(np.isclose(expected_hdot, generated_h_cross_dot, atol=1e-3)))
                mock_time.assert_called_once()
                mock_strain_cross.assert_called_once()

    def test_dEnergy_dt_radiated(self):
        # compare with what it should be
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(self.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle
        radiation_sphere = radiation_bundle.radiation_spheres[75]
        radiation_mode = radiation_sphere.modes[(2, 2)]
        generated_denergy_dt = radiation_mode.dEnergy_dt_radiated
        expected_denergy_dt = (radiation_mode.h_plus_dot ** 2 + radiation_mode.h_cross_dot ** 2) / (16 * np.pi)
        self.assertTrue(np.all(expected_denergy_dt == generated_denergy_dt))

    def test_energy_radiated(self):
        # check with what it should be
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(self.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle
        radiation_sphere = radiation_bundle.radiation_spheres[75]
        radiation_mode = radiation_sphere.modes[(2, 2)]
        generated_energy_radiated = radiation_mode.energy_radiated

        denergy_dt_radiated = radiation_mode.dEnergy_dt_radiated
        time = radiation_mode.time
        expected_energy_radiated = scipy.integrate.cumulative_trapezoid(denergy_dt_radiated, time, initial=0)

        self.assertTrue(np.all(expected_energy_radiated == generated_energy_radiated))

        # check that 22 mode is greater than 33 mode
        energy_radiated_22 = radiation_sphere.modes[(2, 2)].energy_radiated
        energy_radiated_33 = radiation_sphere.modes[(3, 3)].energy_radiated
        self.assertTrue(energy_radiated_22[-1] > energy_radiated_33[-1])

    def test_dP_dt_radiated(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(self.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle
        radiation_sphere = radiation_bundle.radiation_spheres[75]

        # check if it calls radiation_sphere.modes the correct amount of times for various situations

        placeholder_h_plus_dot = np.zeros(len(radiation_sphere.time))
        placeholder_h_cross_dot = np.zeros(len(radiation_sphere.time))

        # l=2, m=2
        radiation_mode = radiation_sphere.modes[(2, 2)]
        with patch.object(RadiationMode, 'h_plus_dot', new_callable=PropertyMock,
                          return_value=placeholder_h_plus_dot) as mock_h_plus_dot:
            with patch.object(RadiationMode, 'h_cross_dot', new_callable=PropertyMock,
                              return_value=placeholder_h_cross_dot) as mock_h_cross_dot:
                with patch.object(RadiationMode, 'radiation_sphere', new_callable=PropertyMock,
                                  return_value=radiation_sphere) as mock_radiationsphere:
                    radiation_mode.dP_dt_radiated
                    self.assertEqual(12, mock_radiationsphere.call_count)
                    self.assertEqual(4, mock_h_plus_dot.call_count)
                    self.assertEqual(4, mock_h_cross_dot.call_count)

        # l=3, m=1
        radiation_mode = radiation_sphere.modes[(3, 1)]
        with patch.object(RadiationMode, 'h_plus_dot', new_callable=PropertyMock,
                          return_value=placeholder_h_plus_dot) as mock_h_plus_dot:
            with patch.object(RadiationMode, 'h_cross_dot', new_callable=PropertyMock,
                              return_value=placeholder_h_cross_dot) as mock_h_cross_dot:
                with patch.object(RadiationMode, 'radiation_sphere', new_callable=PropertyMock,
                                  return_value=radiation_sphere) as mock_radiationsphere:
                    radiation_mode.dP_dt_radiated
                    self.assertEqual(18, mock_radiationsphere.call_count)
                    self.assertEqual(7, mock_h_plus_dot.call_count)
                    self.assertEqual(7, mock_h_cross_dot.call_count)

        # l=8, m=2
        radiation_mode = radiation_sphere.modes[(8, 2)]
        with patch.object(RadiationMode, 'h_plus_dot', new_callable=PropertyMock,
                          return_value=placeholder_h_plus_dot) as mock_h_plus_dot:
            with patch.object(RadiationMode, 'h_cross_dot', new_callable=PropertyMock,
                              return_value=placeholder_h_cross_dot) as mock_h_cross_dot:
                with patch.object(RadiationMode, 'radiation_sphere', new_callable=PropertyMock,
                                  return_value=radiation_sphere) as mock_radiationsphere:
                    radiation_mode.dP_dt_radiated
                    self.assertEqual(14, mock_radiationsphere.call_count)
                    self.assertEqual(5, mock_h_plus_dot.call_count)
                    self.assertEqual(5, mock_h_cross_dot.call_count)

        # l=8, m=8
        radiation_mode = radiation_sphere.modes[(8, 8)]
        with patch.object(RadiationMode, 'h_plus_dot', new_callable=PropertyMock,
                          return_value=placeholder_h_plus_dot) as mock_h_plus_dot:
            with patch.object(RadiationMode, 'h_cross_dot', new_callable=PropertyMock,
                              return_value=placeholder_h_cross_dot) as mock_h_cross_dot:
                with patch.object(RadiationMode, 'radiation_sphere', new_callable=PropertyMock,
                                  return_value=radiation_sphere) as mock_radiationsphere:
                    radiation_mode.dP_dt_radiated
                    self.assertEqual(8, mock_radiationsphere.call_count)
                    self.assertEqual(2, mock_h_plus_dot.call_count)
                    self.assertEqual(2, mock_h_cross_dot.call_count)

    def test_linear_momentum_radiated(self):
        # check with what it should be
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(self.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle
        radiation_sphere = radiation_bundle.radiation_spheres[75]
        radiation_mode = radiation_sphere.modes[(2, 2)]
        generated_linear_momentum_radiated = radiation_mode.linear_momentum_radiated

        dP_dt_radiated = radiation_mode.dP_dt_radiated
        time = radiation_mode.time
        expected_linear_momentum_radiated = scipy.integrate.cumulative_trapezoid(dP_dt_radiated, time, initial=0,
                                                                                 axis=0)

        self.assertTrue(np.all(expected_linear_momentum_radiated == generated_linear_momentum_radiated))

    def test_compute_and_store_strain(self):
        psi4_data_file = os.path.join(self.CURR_DIR, "resources/psi4_strain_test/Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc")
        time, psi4_real, psi4_imag = np.loadtxt(psi4_data_file, usecols=(0, 1, 2), unpack=True)
        self.radiation_mode = RadiationMode(l=2, m=2, rad=75, time=time, psi4_real=psi4_real, psi4_imaginary=psi4_imag)
        self.radiation_mode.compute_and_store_strain()

        # Check second derivative recovers psi4
        strain_plus = self.radiation_mode.strain_plus
        strain_cross = self.radiation_mode.strain_cross
        recovered_psi4_real = np.gradient(np.gradient(strain_plus, time), time)
        recovered_psi4_imag = np.gradient(np.gradient(strain_cross, time), time)

        self.assertTrue(np.all(np.isclose(75 * psi4_real, recovered_psi4_real, atol=1e-3)))
        self.assertTrue(np.all(np.isclose(75 * psi4_imag, -1 * recovered_psi4_imag, atol=1e-3)))

        # Compare to Healy toolkit
        expected_strain_data_file = os.path.join(self.CURR_DIR, "resources/psi4_strain_test/Strain_l2_m2_r75.txt")
        expected_time, expected_strain_plus, expected_strain_cross = np.loadtxt(expected_strain_data_file,
                                                                                usecols=(0, 1, 2), unpack=True)

        # crop off the beginning since the data from Healy Toolkit already windowed the beginning
        cut_index = np.argmax(time > 200)
        strain_plus = strain_plus[cut_index:]
        strain_cross = strain_cross[cut_index:]
        expected_strain_plus = expected_strain_plus[cut_index:]
        expected_strain_cross = expected_strain_cross[cut_index:]

        self.assertTrue(np.all(np.isclose(expected_strain_plus, strain_plus, atol=2e-3)))
        self.assertTrue(np.all(np.isclose(expected_strain_cross, strain_cross, atol=2e-3)))

    def test_get_mode_with_extrapolated_radius(self):
        psi4_data_file = os.path.join(self.CURR_DIR, "resources/psi4_strain_test/Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc")
        time, psi4_real, psi4_imag = np.loadtxt(psi4_data_file, usecols=(0, 1, 2), unpack=True)
        self.radiation_mode = RadiationMode(l=2, m=2, rad=75, time=time, psi4_real=psi4_real, psi4_imaginary=psi4_imag)
        extrapolated_radiation_mode = self.radiation_mode.get_mode_with_extrapolated_radius(order=2)

        # Compare to Healy toolkit by going to strain first
        strain_plus = extrapolated_radiation_mode.strain_plus
        strain_cross = extrapolated_radiation_mode.strain_cross

        expected_strain_data_file = os.path.join(self.CURR_DIR, "resources/psi4_strain_test/Strain_l2_m2_rinf.txt")
        expected_time, expected_strain_plus, expected_strain_cross = np.loadtxt(expected_strain_data_file,
                                                                                usecols=(0, 1, 2), unpack=True)

        # crop off the beginning since the data from Healy Toolkit already windowed the beginning
        cut_index = np.argmax(time>200)
        strain_plus = strain_plus[cut_index:]
        strain_cross = strain_cross[cut_index:]
        expected_strain_plus = expected_strain_plus[cut_index:]
        expected_strain_cross = expected_strain_cross[cut_index:]

        self.assertTrue(np.all(np.isclose(expected_strain_plus, strain_plus, atol=2e-3)))
        self.assertTrue(np.all(np.isclose(expected_strain_cross, strain_cross, atol=2e-3)))

    def test__ylm(self):
        # face on
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, 2, 0, 0)), 0.630783, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, -2, 0, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 3, 0, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 2, 0, 0)), 0.746353, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, -1, 0, 0)), 0, atol=1e-5))

        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, 2, 0, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, -2, 0, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 3, 0, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 2, 0, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, -1, 0, 0)), 0, atol=1e-5))

        # face on, pi/4
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, 2, 0, np.pi / 4)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, -2, 0, np.pi / 4)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 3, 0, np.pi / 4)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 2, 0, np.pi / 4)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, -1, 0, np.pi / 4)), 0, atol=1e-5))

        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, 2, 0, np.pi / 4)), 0.630783, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, -2, 0, np.pi / 4)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 3, 0, np.pi / 4)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 2, 0, np.pi / 4)), 0.746353, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, -1, 0, np.pi / 4)), 0, atol=1e-5))

        # pi/4
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, 2, np.pi / 4, 0)), 0.459559, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, -2, np.pi / 4, 0)), 0.0135282, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 3, np.pi / 4, 0)), -0.470908, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 2, np.pi / 4, 0)), 0.0659689, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, -1, np.pi / 4, 0)), 0.190716, atol=1e-5))

        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, 2, np.pi / 4, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, -2, np.pi / 4, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 3, np.pi / 4, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 2, np.pi / 4, 0)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, -1, np.pi / 4, 0)), 0, atol=1e-5))

        # pi/2
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, 2, np.pi / 2, np.pi)), 0.157696, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, -2, np.pi / 2, np.pi)), 0.157696, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 3, np.pi / 2, np.pi)), 0.228523, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 2, np.pi / 2, np.pi)), -0.373176, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, -1, np.pi / 2, np.pi)), -0.295022, atol=1e-5))

        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, 2, np.pi / 2, np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, -2, np.pi / 2, np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 3, np.pi / 2, np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 2, np.pi / 2, np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, -1, np.pi / 2, np.pi)), 0, atol=1e-5))

        # pi
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, 2, np.pi, 0.1 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, -2, np.pi, 0.1 * np.pi)), 0.510314, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 3, np.pi, 0.1 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 2, np.pi, 0.1 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, -1, np.pi, 0.1 * np.pi)), 0, atol=1e-5))

        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, 2, np.pi, 0.1 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, -2, np.pi, 0.1 * np.pi)), -0.370765, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 3, np.pi, 0.1 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 2, np.pi, 0.1 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, -1, np.pi, 0.1 * np.pi)), 0, atol=1e-5))

        # 0.1 * pi
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, 2, 0.1 * np.pi, 1.3 * np.pi)), -0.185499, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(2, -2, 0.1 * np.pi, 1.3 * np.pi)), -0.0001167, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 3, 0.1 * np.pi, 1.3 * np.pi)), -0.255657, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, 2, 0.1 * np.pi, 1.3 * np.pi)), -0.187258, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(3, -1, 0.1 * np.pi, 1.3 * np.pi)), -0.0101057, atol=1e-5))

        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, 2, 0.1 * np.pi, 1.3 * np.pi)), 0.570908, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(2, -2, 0.1 * np.pi, 1.3 * np.pi)), -0.000359, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 3, 0.1 * np.pi, 1.3 * np.pi)), 0.0830681, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, 2, 0.1 * np.pi, 1.3 * np.pi)), 0.576322, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(3, -1, 0.1 * np.pi, 1.3 * np.pi)), 0.0139094, atol=1e-5))

        # test more modes
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(4, 4, 0.1 * np.pi, 1.3 * np.pi)), -0.0823071, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(4, -2, 0.1 * np.pi, 1.3 * np.pi)), -0.00219086, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(4, 1, 0.1 * np.pi, 1.3 * np.pi)), -0.265387, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(4, 0, 0.1 * np.pi, 1.3 * np.pi)), 0.170312, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(5, 5, 0.1 * np.pi, 1.3 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(5, -5, 0.1 * np.pi, 1.3 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(5, 0, 0.1 * np.pi, 1.3 * np.pi)), 0.263738, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(5, 2, 0.1 * np.pi, 1.3 * np.pi)), -0.130856, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(7, 2, 0.1 * np.pi, 1.3 * np.pi)), -0.0240367, atol=1e-5))
        self.assertTrue(np.isclose(np.real(RadiationMode.ylm(8, 1, 0.1 * np.pi, 1.3 * np.pi)), -0.319124, atol=1e-5))

        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(4, 4, 0.1 * np.pi, 1.3 * np.pi)), -0.0597996, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(4, -2, 0.1 * np.pi, 1.3 * np.pi)), -0.00674276, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(4, 1, 0.1 * np.pi, 1.3 * np.pi)), -0.365273, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(4, 0, 0.1 * np.pi, 1.3 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(5, 5, 0.1 * np.pi, 1.3 * np.pi)), -0.0359765, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(5, -5, 0.1 * np.pi, 1.3 * np.pi)), -0.0000226, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(5, 0, 0.1 * np.pi, 1.3 * np.pi)), 0, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(5, 2, 0.1 * np.pi, 1.3 * np.pi)), 0.402732, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(7, 2, 0.1 * np.pi, 1.3 * np.pi)), 0.0739774, atol=1e-5))
        self.assertTrue(np.isclose(np.imag(RadiationMode.ylm(8, 1, 0.1 * np.pi, 1.3 * np.pi)), -0.439237, atol=1e-5))
