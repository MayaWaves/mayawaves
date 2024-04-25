import os
from unittest import TestCase
from unittest.mock import patch, PropertyMock
import h5py
import numpy as np
from mayawaves.radiation import RadiationBundle
from mayawaves.radiation import RadiationSphere
from unittest import mock


class TestRadiationBundle(TestCase):
    helper_radiation_spheres = {70: RadiationSphere(mode_dict={}, time=np.array([]), radius=70),
                                80: RadiationSphere(mode_dict={}, time=np.array([]), radius=80)}

    def setUp(self) -> None:
        radiation_spheres = {
            70: RadiationSphere(mode_dict={}, time=np.array([]), radius=70),
            80: RadiationSphere(mode_dict={}, time=np.array([]), radius=80)
        }
        self.radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)

        TestRadiationBundle.CURR_DIR = os.path.dirname(__file__)

    def test_create_radiation_bundle(self):
        temp_dir = os.path.join(TestRadiationBundle.CURR_DIR, "resources/temp")
        os.mkdir(temp_dir)
        temp_h5_filename = os.path.join(TestRadiationBundle.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        radiation_group = temp_h5_file.create_group('radiation')
        radiation_group.create_group('key')

        new_radiation_bundle = RadiationBundle.create_radiation_bundle(radiation_group=radiation_group)
        self.assertTrue(new_radiation_bundle is None)

        os.remove(temp_h5_filename)
        os.rmdir(temp_dir)

        temp_dir = os.path.join(TestRadiationBundle.CURR_DIR, "resources/temp")
        os.mkdir(temp_dir)
        temp_h5_filename = os.path.join(TestRadiationBundle.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        radiation_group = temp_h5_file.create_group('radiation')
        radiation_group.create_group('psi4')

        new_radiation_bundle = RadiationBundle.create_radiation_bundle(radiation_group=radiation_group)
        self.assertTrue(new_radiation_bundle is None)

        os.remove(temp_h5_filename)
        os.rmdir(temp_dir)

        # if it contains all the data
        temp_dir = os.path.join(TestRadiationBundle.CURR_DIR, "resources/temp")
        os.mkdir(temp_dir)
        temp_h5_filename = os.path.join(TestRadiationBundle.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')

        radiation_group = temp_h5_file.create_group('radiation')
        psi4_group = radiation_group.create_group('psi4')

        r70_group = psi4_group.create_group('r=70')
        modes_group = r70_group.create_group('modes')
        l2_group = modes_group.create_group('l=2')
        m1_group = l2_group.create_group('m=1')
        m1_group.create_dataset('real', data=np.array([0, 1, 2, 3, 4]))
        m1_group.create_dataset('imaginary', data=np.array([5, 6, 7, 8, 9]))
        m2_group = l2_group.create_group('m=2')
        m2_group.create_dataset('real', data=np.array([0, 1, 2, 3, 4]))
        m2_group.create_dataset('imaginary', data=np.array([5, 6, 7, 8, 9]))
        r70_group.create_dataset('time', np.array([9, 8, 7, 6, 5]))

        r80_group = psi4_group.create_group('r=80')
        modes_group = r80_group.create_group('modes')
        l2_group = modes_group.create_group('l=2')
        m1_group = l2_group.create_group('m=1')
        m1_group.create_dataset('real', data=np.array([0, 1, 2, 3, 4]))
        m1_group.create_dataset('imaginary', data=np.array([0, 1, 2, 3, 4]))
        m2_group = l2_group.create_group('m=2')
        m2_group.create_dataset('real', data=np.array([0, 1, 2, 3, 4]))
        m2_group.create_dataset('imaginary', data=np.array([0, 1, 2, 3, 4]))
        r80_group.create_dataset('time', np.array([9, 8, 7, 6, 5]))

        new_radiation_bundle = RadiationBundle.create_radiation_bundle(radiation_group=radiation_group)
        self.assertTrue(isinstance(new_radiation_bundle, RadiationBundle))
        self.assertTrue(len(new_radiation_bundle._RadiationBundle__radiation_spheres) == 2)
        self.assertTrue(isinstance(new_radiation_bundle._RadiationBundle__radiation_spheres[70], RadiationSphere))
        self.assertTrue(isinstance(new_radiation_bundle._RadiationBundle__radiation_spheres[80], RadiationSphere))
        self.assertFalse(new_radiation_bundle._RadiationBundle__radiation_spheres[70] ==
                         new_radiation_bundle._RadiationBundle__radiation_spheres[80])

        os.remove(temp_h5_filename)
        os.rmdir(temp_dir)

    def test_frame(self):
        from mayawaves.radiation import Frame

        self.radiation_bundle._RadiationBundle__frame = Frame.RAW
        self.assertEqual(Frame.RAW, self.radiation_bundle.frame)

        self.radiation_bundle._RadiationBundle__frame = Frame.COM_CORRECTED
        self.assertEqual(Frame.COM_CORRECTED, self.radiation_bundle.frame)

    def test_set_frame(self):
        from mayawaves.radiation import Frame

        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            # invalid frame
            try:
                self.radiation_bundle.set_frame('test')
                self.fail()
            except ImportError:
                self.assertEqual(Frame.RAW, self.radiation_bundle._RadiationBundle__frame)

            self.radiation_bundle._RadiationBundle__frame = Frame.COM_CORRECTED
            try:
                self.radiation_bundle.set_frame('test')
                self.fail()
            except ImportError:
                self.assertEqual(Frame.COM_CORRECTED, self.radiation_bundle._RadiationBundle__frame)

            # raw
            self.radiation_bundle._RadiationBundle__frame = Frame.RAW
            try:
                self.radiation_bundle.set_frame(Frame.RAW)
                self.fail()
            except ImportError:
                self.assertEqual(Frame.RAW, self.radiation_bundle._RadiationBundle__frame)

            # com
            self.__extrapolated_sphere = self.radiation_bundle.radiation_spheres[70]
            try:
                self.radiation_bundle.set_frame(Frame.COM_CORRECTED)
                self.fail()
            except ImportError:
                self.assertEqual(Frame.RAW, self.radiation_bundle._RadiationBundle__frame)

        else:
            # invalid frame
            self.radiation_bundle.set_frame('test')
            self.assertEqual(Frame.RAW, self.radiation_bundle._RadiationBundle__frame)

            self.radiation_bundle._RadiationBundle__frame = Frame.COM_CORRECTED
            self.radiation_bundle.set_frame('test')
            self.assertEqual(Frame.COM_CORRECTED, self.radiation_bundle._RadiationBundle__frame)

            # raw
            self.radiation_bundle.set_frame(Frame.RAW)
            self.assertEqual(Frame.RAW, self.radiation_bundle._RadiationBundle__frame)
            # check that it set the frame for all the spheres
            for sphere in self.radiation_bundle.radiation_spheres.values():
                self.assertEqual(Frame.RAW, sphere.frame)

            # com
            self.__extrapolated_sphere = self.radiation_bundle.radiation_spheres[70]
            self.radiation_bundle.set_frame(Frame.COM_CORRECTED)
            self.assertEqual(Frame.COM_CORRECTED, self.radiation_bundle._RadiationBundle__frame)
            # check that it set the frame for all the spheres
            for sphere in self.radiation_bundle.radiation_spheres.values():
                self.assertEqual(Frame.COM_CORRECTED, sphere.frame)
            self.assertEqual(Frame.COM_CORRECTED, self.__extrapolated_sphere.frame)

    def test_radiation_spheres(self):
        self.assertTrue(
            self.radiation_bundle._RadiationBundle__radiation_spheres == self.radiation_bundle.radiation_spheres)

    def test_extrapolated_sphere(self):
        # if there is not an extrapolated sphere
        def extrapolate_sphere_side_effect():
            self.radiation_bundle._RadiationBundle__extrapolated_sphere = RadiationSphere(mode_dict={},
                                                                                          time=np.array([]),
                                                                                          radius=0, extrapolated=True)

        with patch.object(RadiationBundle, 'create_extrapolated_sphere',
                          side_effect=extrapolate_sphere_side_effect) as mock_compute_extrapolated_sphere:
            extrapolated_sphere = self.radiation_bundle.extrapolated_sphere
            self.assertTrue(self.radiation_bundle._RadiationBundle__extrapolated_sphere == extrapolated_sphere)
            mock_compute_extrapolated_sphere.assert_called_once()

        # if there is an extrapolated sphere
        with patch.object(RadiationBundle, 'create_extrapolated_sphere',
                          side_effect=extrapolate_sphere_side_effect) as mock_compute_extrapolated_sphere:
            extrapolated_sphere = self.radiation_bundle.extrapolated_sphere
            self.assertTrue(self.radiation_bundle._RadiationBundle__extrapolated_sphere == extrapolated_sphere)
            mock_compute_extrapolated_sphere.assert_not_called()

    def test_radius_for_extrapolation(self):
        # Getter
        # if there is no radius for extrapolation set
        # if radii evenly split around 75
        radiation_spheres = {
            70: RadiationSphere(mode_dict={}, time=np.array([]), radius=70),
            80: RadiationSphere(mode_dict={}, time=np.array([]), radius=80)
        }
        self.radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        extrapolation_radius = self.radiation_bundle.radius_for_extrapolation
        self.assertTrue(70 == extrapolation_radius)

        # if no radius less than 75
        radiation_spheres = {
            80: RadiationSphere(mode_dict={}, time=np.array([]), radius=70),
            90: RadiationSphere(mode_dict={}, time=np.array([]), radius=80)
        }
        self.radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        extrapolation_radius = self.radiation_bundle.radius_for_extrapolation
        self.assertTrue(80 == extrapolation_radius)

        # if no radius greater than 75
        radiation_spheres = {
            60: RadiationSphere(mode_dict={}, time=np.array([]), radius=70),
            70: RadiationSphere(mode_dict={}, time=np.array([]), radius=80)
        }
        self.radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        extrapolation_radius = self.radiation_bundle.radius_for_extrapolation
        self.assertTrue(70 == extrapolation_radius)

        # if 75 is a radius
        radiation_spheres = {
            75: RadiationSphere(mode_dict={}, time=np.array([]), radius=70),
            80: RadiationSphere(mode_dict={}, time=np.array([]), radius=80)
        }
        self.radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        extrapolation_radius = self.radiation_bundle.radius_for_extrapolation
        self.assertTrue(75 == extrapolation_radius)

        # if there is an extrapolation radius set
        self.radiation_bundle._RadiationBundle__radius_for_extrapolation = 30
        extrapolation_radius = self.radiation_bundle.radius_for_extrapolation
        self.assertTrue(30 == extrapolation_radius)

        # Setter
        radiation_spheres = {
            70: RadiationSphere(mode_dict={}, time=np.array([]), radius=70),
            80: RadiationSphere(mode_dict={}, time=np.array([]), radius=80)
        }
        self.radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        # Radius does not have data
        self.radiation_bundle._RadiationBundle__radius_for_extrapolation = None
        self.radiation_bundle.radius_for_extrapolation = 100
        self.assertTrue(self.radiation_bundle._RadiationBundle__radius_for_extrapolation is None)

        # Radius does have data
        self.radiation_bundle.radius_for_extrapolation = 70
        self.assertTrue(70 == self.radiation_bundle._RadiationBundle__radius_for_extrapolation)

    def test_l_max(self):
        with patch.object(RadiationSphere, 'l_max', new_callable=PropertyMock, return_value=8) as mock_l_max:
            self.assertTrue(8 == self.radiation_bundle.l_max)
            self.assertTrue(2 == mock_l_max.call_count)

    def test_included_modes(self):
        with patch.object(RadiationSphere, 'included_modes', new_callable=PropertyMock,
                          return_value=[(2, 2), (3, 2), (3, 3)]) as mock_included_modes:
            self.assertTrue([(2, 2), (3, 2), (3, 3)] == self.radiation_bundle.included_modes)
            self.assertTrue(2 == mock_included_modes.call_count)

    def test_included_radii(self):
        radiation_spheres = {
            70: RadiationSphere(mode_dict={}, time=np.array([]), radius=70),
            80: RadiationSphere(mode_dict={}, time=np.array([]), radius=80)
        }
        self.radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        self.assertTrue([70, 80] == self.radiation_bundle.included_radii)

    def test_get_time(self):
        # if extraction radius not provided
        extrapolated_time = np.array([-1, -2, -3, -4, -5])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=extrapolated_time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'time', new_callable=PropertyMock,
                              return_value=extrapolated_time) as mock_time:
                time = self.radiation_bundle.get_time()
                self.assertTrue(np.all(extrapolated_time == time))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_time.assert_called_once()

        # if extraction radius is provided but doesn't exist
        non_extrapolated_time = np.array([0, 1, 2, 3, 4])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            time = self.radiation_bundle.get_time(extraction_radius=90)
            self.assertTrue(time is None)

        # if extraction radius is provided and exists
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'time', new_callable=PropertyMock,
                              return_value=non_extrapolated_time) as mock_time:
                time = self.radiation_bundle.get_time(extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_time == time))
                self.assertEqual(2, mock_radiation_spheres.call_count)
                mock_time.assert_called_once()

    def test_get_psi4_real_for_mode(self):
        # if extraction radius not provided
        extrapolated_psi4_real = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_psi4_real_for_mode',
                              return_value=extrapolated_psi4_real) as mock_psi4_real:
                psi4_real = self.radiation_bundle.get_psi4_real_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_psi4_real == psi4_real))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_psi4_real.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            psi4_real = self.radiation_bundle.get_psi4_real_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(psi4_real is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_psi4_real = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_psi4_real_for_mode',
                              return_value=non_extrapolated_psi4_real) as mock_psi4_real:
                psi4_real = self.radiation_bundle.get_psi4_real_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_psi4_real == psi4_real))
                self.assertEqual(2, mock_radiation_spheres.call_count)
                mock_psi4_real.assert_called_once_with(l=2, m=2)

    def test_get_psi4_imaginary_for_mode(self):
        # if extraction radius not provided
        extrapolated_psi4_imaginary = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_psi4_imaginary_for_mode',
                              return_value=extrapolated_psi4_imaginary) as mock_psi4_imaginary:
                psi4_imaginary = self.radiation_bundle.get_psi4_imaginary_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_psi4_imaginary == psi4_imaginary))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_psi4_imaginary.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            psi4_imaginary = self.radiation_bundle.get_psi4_imaginary_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(psi4_imaginary is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_psi4_imaginary = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_psi4_imaginary_for_mode',
                              return_value=non_extrapolated_psi4_imaginary) as mock_psi4_imaginary:
                psi4_imaginary = self.radiation_bundle.get_psi4_imaginary_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_psi4_imaginary == psi4_imaginary))
                self.assertTrue(2 == mock_radiation_spheres.call_count)
                mock_psi4_imaginary.assert_called_once_with(l=2, m=2)

    def test_get_psi4_amplitude_for_mode(self):
        # if extraction radius not provided
        extrapolated_psi4_amplitude = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_psi4_amplitude_for_mode',
                              return_value=extrapolated_psi4_amplitude) as mock_psi4_amplitude:
                psi4_amplitude = self.radiation_bundle.get_psi4_amplitude_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_psi4_amplitude == psi4_amplitude))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_psi4_amplitude.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            psi4_amplitude = self.radiation_bundle.get_psi4_amplitude_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(psi4_amplitude is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_psi4_amplitude = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_psi4_amplitude_for_mode',
                              return_value=non_extrapolated_psi4_amplitude) as mock_psi4_amplitude:
                psi4_amplitude = self.radiation_bundle.get_psi4_amplitude_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_psi4_amplitude == psi4_amplitude))
                self.assertEqual(2, mock_radiation_spheres.call_count)
                mock_psi4_amplitude.assert_called_once_with(l=2, m=2)

    def test_get_psi4_phase_for_mode(self):
        # if extraction radius not provided
        extrapolated_psi4_phase = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_psi4_phase_for_mode',
                              return_value=extrapolated_psi4_phase) as mock_psi4_phase:
                psi4_phase = self.radiation_bundle.get_psi4_phase_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_psi4_phase == psi4_phase))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_psi4_phase.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            psi4_phase = self.radiation_bundle.get_psi4_phase_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(psi4_phase is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_psi4_phase = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_psi4_phase_for_mode',
                              return_value=non_extrapolated_psi4_phase) as mock_psi4_phase:
                psi4_phase = self.radiation_bundle.get_psi4_phase_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_psi4_phase == psi4_phase))
                self.assertEqual(2, mock_radiation_spheres.call_count)
                mock_psi4_phase.assert_called_once_with(l=2, m=2)

    def test_get_strain_plus_for_mode(self):
        # if extraction radius not provided
        extrapolated_strain_plus = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_strain_plus_for_mode',
                              return_value=extrapolated_strain_plus) as mock_strain_plus:
                strain_plus = self.radiation_bundle.get_strain_plus_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_strain_plus == strain_plus))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_strain_plus.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            strain_plus = self.radiation_bundle.get_strain_plus_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(strain_plus is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_strain_plus = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_strain_plus_for_mode',
                              return_value=non_extrapolated_strain_plus) as mock_strain_plus:
                strain_plus = self.radiation_bundle.get_strain_plus_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_strain_plus == strain_plus))
                self.assertEqual(2, mock_radiation_spheres.call_count)
                mock_strain_plus.assert_called_once_with(l=2, m=2)

    def test_get_strain_cross_for_mode(self):
        # if extraction radius not provided
        extrapolated_strain_cross = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_strain_cross_for_mode',
                              return_value=extrapolated_strain_cross) as mock_strain_cross:
                strain_cross = self.radiation_bundle.get_strain_cross_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_strain_cross == strain_cross))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_strain_cross.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            strain_cross = self.radiation_bundle.get_strain_cross_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(strain_cross is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_strain_cross = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_strain_cross_for_mode',
                              return_value=non_extrapolated_strain_cross) as mock_strain_cross:
                strain_cross = self.radiation_bundle.get_strain_cross_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_strain_cross == strain_cross))
                self.assertEqual(2, mock_radiation_spheres.call_count)
                mock_strain_cross.assert_called_once_with(l=2, m=2)

    def test_get_strain_recomposed_at_sky_location(self):
        # if extraction radius not provided
        extrapolated_strain_recomposed = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_strain_recomposed_at_sky_location',
                              return_value=extrapolated_strain_recomposed) as mock_strain_recomposed:
                strain_recomposed = self.radiation_bundle.get_strain_recomposed_at_sky_location(theta=1.2 * np.pi,
                                                                                                phi=0.3 * np.pi)
                self.assertTrue(np.all(extrapolated_strain_recomposed == strain_recomposed))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_strain_recomposed.assert_called_once_with(theta=1.2 * np.pi, phi=0.3 * np.pi)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            strain_recomposed = self.radiation_bundle.get_strain_recomposed_at_sky_location(theta=1.2 * np.pi,
                                                                                            phi=0.3 * np.pi,
                                                                                            extraction_radius=90)
            self.assertTrue(strain_recomposed is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_strain_recomposed = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_strain_recomposed_at_sky_location',
                              return_value=non_extrapolated_strain_recomposed) as mock_strain_recomposed:
                strain_recomposed = self.radiation_bundle.get_strain_recomposed_at_sky_location(theta=1.2 * np.pi,
                                                                                                phi=0.3 * np.pi,
                                                                                                extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_strain_recomposed == strain_recomposed))
                self.assertTrue(2 == mock_radiation_spheres.call_count)
                mock_strain_recomposed.assert_called_once_with(theta=1.2 * np.pi, phi=0.3 * np.pi)

    def test_get_strain_amplitude_for_mode(self):
        # if extraction radius not provided
        extrapolated_strain_amplitude = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_strain_amplitude_for_mode',
                              return_value=extrapolated_strain_amplitude) as mock_strain_amplitude:
                strain_amplitude = self.radiation_bundle.get_strain_amplitude_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_strain_amplitude == strain_amplitude))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_strain_amplitude.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            strain_amplitude = self.radiation_bundle.get_strain_amplitude_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(strain_amplitude is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_strain_amplitude = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_strain_amplitude_for_mode',
                              return_value=non_extrapolated_strain_amplitude) as mock_strain_amplitude:
                strain_amplitude = self.radiation_bundle.get_strain_amplitude_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_strain_amplitude == strain_amplitude))
                self.assertTrue(2 == mock_radiation_spheres.call_count)
                mock_strain_amplitude.assert_called_once_with(l=2, m=2)

    def test_get_strain_phase_for_mode(self):
        # if extraction radius not provided
        extrapolated_strain_phase = np.array([-5, -4, -3, -2, -1])
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_strain_phase_for_mode',
                              return_value=extrapolated_strain_phase) as mock_strain_phase:
                strain_phase = self.radiation_bundle.get_strain_phase_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_strain_phase == strain_phase))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_strain_phase.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            strain_phase = self.radiation_bundle.get_strain_phase_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(strain_phase is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_strain_phase = np.array([5, 4, 3, 2, 1])
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_strain_phase_for_mode',
                              return_value=non_extrapolated_strain_phase) as mock_strain_phase:
                strain_phase = self.radiation_bundle.get_strain_phase_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_strain_phase == strain_phase))
                self.assertEqual(2, mock_radiation_spheres.call_count)
                mock_strain_phase.assert_called_once_with(l=2, m=2)

    def test_get_psi4_max_time_for_mode(self):
        # if extraction radius not provided
        extrapolated_psi4_max_time = 50
        time = np.array([0, 1, 2, 3, 4])
        extrapolated_sphere = RadiationSphere(mode_dict={}, time=time, radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=extrapolated_sphere) as mock_extrapolated_sphere:
            with patch.object(RadiationSphere, 'get_psi4_max_time_for_mode',
                              return_value=extrapolated_psi4_max_time) as mock_psi4_max_time:
                psi4_max_time = self.radiation_bundle.get_psi4_max_time_for_mode(l=2, m=2)
                self.assertTrue(np.all(extrapolated_psi4_max_time == psi4_max_time))
                self.assertEqual(2, mock_extrapolated_sphere.call_count)
                mock_psi4_max_time.assert_called_once_with(l=2, m=2)

        # if extraction radius is provided but doesn't exist
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            psi4_max_time = self.radiation_bundle.get_psi4_max_time_for_mode(l=2, m=2, extraction_radius=90)
            self.assertTrue(psi4_max_time is None)
            mock_radiation_spheres.assert_called_once()

        # if extraction radius is provided and exists
        non_extrapolated_psi4_max_time = 30
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationSphere, 'get_psi4_max_time_for_mode',
                              return_value=non_extrapolated_psi4_max_time) as mock_psi4_max_time:
                psi4_max_time = self.radiation_bundle.get_psi4_max_time_for_mode(l=2, m=2, extraction_radius=70)
                self.assertTrue(np.all(non_extrapolated_psi4_max_time == psi4_max_time))
                self.assertEqual(2, mock_radiation_spheres.call_count)
                mock_psi4_max_time.assert_called_once_with(l=2, m=2)

    def test_get_dEnergy_dt_radiated(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(TestRadiationBundle.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle

        # compare with psi4analysis
        generated_time, generated_denergy_dt_radiated = radiation_bundle.get_dEnergy_dt_radiated(lmin=2, lmax=8,
                                                                                                 extraction_radius=75)

        psi4_analysis_data = np.loadtxt(os.path.join(TestRadiationBundle.CURR_DIR,
                                                     'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
                                        usecols=5)

        tolerance = 1e-2 * max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_denergy_dt_radiated, atol=tolerance))

        # check kwargs get passed along
        return_placeholder = None, None

        # nothing
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated()
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call, mock_get_dEnergy_dt_radiated.call_args)

        # lmin only
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated(lmin=2)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(lmin=2), mock_get_dEnergy_dt_radiated.call_args)

        # lmax only
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated(lmax=4)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(lmax=4), mock_get_dEnergy_dt_radiated.call_args)

        # lmin and lmax
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated(lmin=3, lmax=7)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(lmin=3, lmax=7), mock_get_dEnergy_dt_radiated.call_args)

        # l and m
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated(l=4, m=1)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4, m=1), mock_get_dEnergy_dt_radiated.call_args)

        # l only
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated(l=4)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4), mock_get_dEnergy_dt_radiated.call_args)

        # m only
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated(m=1)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(m=1), mock_get_dEnergy_dt_radiated.call_args)

        # l and lmin
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated(l=4, lmin=3)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4, lmin=3), mock_get_dEnergy_dt_radiated.call_args)

        # extraction radius
        with patch.object(RadiationSphere, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            radiation_bundle.get_dEnergy_dt_radiated(extraction_radius=75)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(), mock_get_dEnergy_dt_radiated.call_args)

        # check if extrapolated sphere is called
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=radiation_bundle.extrapolated_sphere) as mock_extrapolated_sphere:
            radiation_bundle.get_dEnergy_dt_radiated()
            self.assertEqual(2, mock_extrapolated_sphere.call_count)

        # check if correct extraction radius is called
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=radiation_bundle.radiation_spheres) as mock_radiation_spheres:
            radiation_bundle.get_dEnergy_dt_radiated(extraction_radius=radiation_bundle.included_radii[0])
            self.assertEqual(3, mock_radiation_spheres.call_count)

        # check that None is returned if invalid radius is called
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=radiation_bundle.radiation_spheres) as mock_radiation_spheres:
            result = radiation_bundle.get_dEnergy_dt_radiated(extraction_radius=-1)
            self.assertEqual(1, mock_radiation_spheres.call_count)
            self.assertTrue(result is None)

    def test_get_energy_radiated(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(TestRadiationBundle.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle

        # compare with psi4analysis
        generated_time, generated_energy_radiated = radiation_bundle.get_energy_radiated(lmin=2, lmax=8,
                                                                                         extraction_radius=75)

        psi4_analysis_data = np.loadtxt(os.path.join(TestRadiationBundle.CURR_DIR,
                                                     'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
                                        usecols=1)

        tolerance = 1e-2 * max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_energy_radiated, atol=tolerance))

        # check kwargs get passed along
        return_placeholder = None, None

        # nothing
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated()
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call, mock_get_energy_radiated.call_args)

        # lmin only
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated(lmin=2)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(lmin=2), mock_get_energy_radiated.call_args)

        # lmax only
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated(lmax=4)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(lmax=4), mock_get_energy_radiated.call_args)

        # lmin and lmax
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated(lmin=3, lmax=7)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(lmin=3, lmax=7), mock_get_energy_radiated.call_args)

        # l and m
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated(l=4, m=1)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(l=4, m=1), mock_get_energy_radiated.call_args)

        # l only
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated(l=4)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(l=4), mock_get_energy_radiated.call_args)

        # m only
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated(m=1)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(m=1), mock_get_energy_radiated.call_args)

        # l and lmin
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated(l=4, lmin=3)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(l=4, lmin=3), mock_get_energy_radiated.call_args)

        # extraction radius
        with patch.object(RadiationSphere, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            radiation_bundle.get_energy_radiated(extraction_radius=75)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(), mock_get_energy_radiated.call_args)

        # check if extrapolated sphere is called
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=radiation_bundle.extrapolated_sphere) as mock_extrapolated_sphere:
            radiation_bundle.get_energy_radiated()
            self.assertEqual(2, mock_extrapolated_sphere.call_count)

        # check if correct extraction radius is called
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=radiation_bundle.radiation_spheres) as mock_radiation_spheres:
            radiation_bundle.get_energy_radiated(extraction_radius=radiation_bundle.included_radii[0])
            self.assertEqual(3, mock_radiation_spheres.call_count)

        # check that None is returned if invalid radius is called
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=radiation_bundle.radiation_spheres) as mock_radiation_spheres:
            result = radiation_bundle.get_energy_radiated(extraction_radius=-1)
            self.assertEqual(1, mock_radiation_spheres.call_count)
            self.assertTrue(result[0] is None)
            self.assertTrue(result[1] is None)

    def test_get_dP_dt_radiated(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(TestRadiationBundle.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle

        # compare with psi4analysis
        generated_time, generated_dP_dt = radiation_bundle.get_dP_dt_radiated(lmin=2, lmax=8, extraction_radius=75)

        psi4_analysis_data = np.loadtxt(os.path.join(TestRadiationBundle.CURR_DIR,
                                                     'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
                                        usecols=(6, 7, 8))

        tolerance = 5e-2 * np.max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_dP_dt, atol=tolerance))

        # check kwargs get passed along
        return_placeholder = None, None

        # nothing
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated()
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call, mock_get_dP_dt_radiated.call_args)

        # lmin only
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated(lmin=2)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(lmin=2), mock_get_dP_dt_radiated.call_args)

        # lmax only
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated(lmax=4)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(lmax=4), mock_get_dP_dt_radiated.call_args)

        # lmin and lmax
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated(lmin=3, lmax=7)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(lmin=3, lmax=7), mock_get_dP_dt_radiated.call_args)

        # l and m
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated(l=4, m=1)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4, m=1), mock_get_dP_dt_radiated.call_args)

        # l only
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated(l=4)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4), mock_get_dP_dt_radiated.call_args)

        # m only
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated(m=1)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(m=1), mock_get_dP_dt_radiated.call_args)

        # l and lmin
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated(l=4, lmin=3)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4, lmin=3), mock_get_dP_dt_radiated.call_args)

        # extraction radius
        with patch.object(RadiationSphere, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            radiation_bundle.get_dP_dt_radiated(extraction_radius=75)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(), mock_get_dP_dt_radiated.call_args)

        # check if extrapolated sphere is called
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=radiation_bundle.extrapolated_sphere) as mock_extrapolated_sphere:
            radiation_bundle.get_dP_dt_radiated()
            self.assertEqual(2, mock_extrapolated_sphere.call_count)

        # check if correct extraction radius is called
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=radiation_bundle.radiation_spheres) as mock_radiation_spheres:
            radiation_bundle.get_dP_dt_radiated(extraction_radius=radiation_bundle.included_radii[0])
            self.assertEqual(3, mock_radiation_spheres.call_count)

        # check that None is returned if invalid radius is called
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=radiation_bundle.radiation_spheres) as mock_radiation_spheres:
            result = radiation_bundle.get_dP_dt_radiated(extraction_radius=-1)
            self.assertEqual(1, mock_radiation_spheres.call_count)
            self.assertTrue(result is None)

    def test_get_linear_momentum_radiated(self):
        from mayawaves.coalescence import Coalescence
        coalescence = Coalescence(os.path.join(TestRadiationBundle.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        radiation_bundle = coalescence.radiationbundle

        # compare with psi4analysis
        generated_time, generated_momentum = radiation_bundle.get_linear_momentum_radiated(lmin=2, lmax=8,
                                                                                           extraction_radius=75)

        psi4_analysis_data = np.loadtxt(os.path.join(TestRadiationBundle.CURR_DIR,
                                                     'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
                                        usecols=(2, 3, 4))

        tolerance = 5e-2 * np.max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_momentum, atol=tolerance))

        # check kwargs get passed along
        return_placeholder = None, None

        # nothing
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated()
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call, mock_get_linear_momentum_radiated.call_args)

        # lmin only
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated(lmin=2)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(lmin=2), mock_get_linear_momentum_radiated.call_args)

        # lmax only
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated(lmax=4)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(lmax=4), mock_get_linear_momentum_radiated.call_args)

        # lmin and lmax
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated(lmin=3, lmax=7)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(lmin=3, lmax=7), mock_get_linear_momentum_radiated.call_args)

        # l and m
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated(l=4, m=1)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(l=4, m=1), mock_get_linear_momentum_radiated.call_args)

        # l only
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated(l=4)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(l=4), mock_get_linear_momentum_radiated.call_args)

        # m only
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated(m=1)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(m=1), mock_get_linear_momentum_radiated.call_args)

        # l and lmin
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated(l=4, lmin=3)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(l=4, lmin=3), mock_get_linear_momentum_radiated.call_args)

        # extraction radius
        with patch.object(RadiationSphere, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            radiation_bundle.get_linear_momentum_radiated(extraction_radius=75)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(), mock_get_linear_momentum_radiated.call_args)

        # check if extrapolated sphere is called
        with patch.object(RadiationBundle, 'extrapolated_sphere', new_callable=PropertyMock,
                          return_value=radiation_bundle.extrapolated_sphere) as mock_extrapolated_sphere:
            radiation_bundle.get_linear_momentum_radiated()
            self.assertEqual(2, mock_extrapolated_sphere.call_count)

        # check if correct extraction radius is called
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=radiation_bundle.radiation_spheres) as mock_radiation_spheres:
            radiation_bundle.get_linear_momentum_radiated(extraction_radius=radiation_bundle.included_radii[0])
            self.assertEqual(3, mock_radiation_spheres.call_count)

        # check that None is returned if invalid radius is called
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=radiation_bundle.radiation_spheres) as mock_radiation_spheres:
            result = radiation_bundle.get_linear_momentum_radiated(extraction_radius=-1)
            self.assertEqual(1, mock_radiation_spheres.call_count)
            self.assertTrue(result is None)

    def test_create_extrapolated_sphere(self):
        # no arguments provided
        expected_extrapolated_sphere = RadiationSphere(mode_dict={}, time=np.array([]), radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationBundle, 'radius_for_extrapolation', new_callable=PropertyMock,
                              return_value=70) as mock_radius_for_extrapolation:
                with patch.object(RadiationSphere, 'get_extrapolated_sphere',
                                  return_value=expected_extrapolated_sphere) as mock_extrapolated_sphere:
                    self.radiation_bundle.create_extrapolated_sphere()
                    self.assertTrue(
                        expected_extrapolated_sphere == self.radiation_bundle._RadiationBundle__extrapolated_sphere)
                    mock_radiation_spheres.assert_called_once()
                    mock_radius_for_extrapolation.assert_called_once()
                    mock_extrapolated_sphere.assert_called_once_with(order=2)

        # arguments provided
        expected_extrapolated_sphere = RadiationSphere(mode_dict={}, time=np.array([]), radius=0, extrapolated=True)
        with patch.object(RadiationBundle, 'radiation_spheres', new_callable=PropertyMock,
                          return_value=self.helper_radiation_spheres) as mock_radiation_spheres:
            with patch.object(RadiationBundle, 'radius_for_extrapolation', new_callable=PropertyMock,
                              return_value=70) as mock_radius_for_extrapolation:
                with patch.object(RadiationSphere, 'get_extrapolated_sphere',
                                  return_value=expected_extrapolated_sphere) as mock_extrapolated_sphere:
                    self.radiation_bundle.create_extrapolated_sphere(order=1)
                    self.assertTrue(
                        expected_extrapolated_sphere == self.radiation_bundle._RadiationBundle__extrapolated_sphere)
                    mock_radiation_spheres.asssert_called_once()
                    mock_radius_for_extrapolation.assert_called_once()
                    mock_extrapolated_sphere.assert_called_once_with(order=1)
