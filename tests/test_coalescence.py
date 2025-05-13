from unittest import TestCase
from unittest.mock import PropertyMock, patch
import h5py
from mayawaves.coalescence import Coalescence
import numpy as np
from shutil import copy2
import os
from unittest import mock
from mayawaves.radiation import RadiationSphere, RadiationMode, RadiationBundle
import shutil


class TestCoalescence(TestCase):
    simulation_name = None
    h5_file = None
    coalescence = None

    CURR_DIR = os.path.dirname(__file__)

    def setUp(self) -> None:
        TestCoalescence.simulation_name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        TestCoalescence.h5_file = os.path.join(TestCoalescence.CURR_DIR,
                                               "resources/temp/%s.h5") % TestCoalescence.simulation_name

        if os.path.exists(TestCoalescence.h5_file):
            os.remove(TestCoalescence.h5_file)
        if os.path.exists(os.path.join(TestCoalescence.CURR_DIR, "resources/temp")):
            shutil.rmtree(os.path.join(TestCoalescence.CURR_DIR, "resources/temp"))

        h5_file_original = os.path.join(TestCoalescence.CURR_DIR, "resources/%s.h5") % TestCoalescence.simulation_name
        os.mkdir(os.path.join(TestCoalescence.CURR_DIR, "resources/temp"))
        copy2(h5_file_original, TestCoalescence.h5_file)

        TestCoalescence.coalescence = Coalescence(TestCoalescence.h5_file)

    def tearDown(self) -> None:
        try:
            TestCoalescence.coalescence.close()
        except (ValueError, RuntimeError) as e:
            pass
        if os.path.exists(TestCoalescence.h5_file):
            os.remove(TestCoalescence.h5_file)
        if os.path.exists(os.path.join(TestCoalescence.CURR_DIR, "resources/temp")):
            shutil.rmtree(os.path.join(TestCoalescence.CURR_DIR, "resources/temp"))

    def test_name(self):
        self.assertEqual(TestCoalescence.simulation_name, TestCoalescence.coalescence.name)
        self.assertEqual(TestCoalescence.coalescence._Coalescence__h5_file.attrs["name"],
                         TestCoalescence.coalescence.name)

    def test_catalog_id(self):
        self.assertIsNone(TestCoalescence.coalescence.catalog_id)

        h5_filename = os.path.join(TestCoalescence.CURR_DIR, "resources/catalog0000.h5")
        temp_coalescence = Coalescence(h5_filename)
        self.assertEqual('catalog0000', temp_coalescence.catalog_id)

    def test_parameter_files(self):
        parameter_file_dict = TestCoalescence.coalescence.parameter_files
        self.assertTrue('rpar' in parameter_file_dict)
        self.assertTrue('par' in parameter_file_dict)
        self.assertEqual(TestCoalescence.coalescence._Coalescence__h5_file["parfile"].attrs["rpar_content"],
                         parameter_file_dict['rpar'])
        self.assertEqual(TestCoalescence.coalescence._Coalescence__h5_file["parfile"].attrs["par_content"],
                         parameter_file_dict['par'])

    def test_h5_filepath(self):
        self.assertEqual(TestCoalescence.h5_file, TestCoalescence.coalescence.h5_filepath)
        self.assertEqual(TestCoalescence.coalescence._Coalescence__h5_filepath,
                         TestCoalescence.coalescence.h5_filepath)

    def test_primary_compact_object(self):
        expected_compact_object = TestCoalescence.coalescence._Coalescence__compact_objects[
            TestCoalescence.coalescence._Coalescence__primary_compact_object_index]
        actual_compact_object = TestCoalescence.coalescence.primary_compact_object
        self.assertEqual(expected_compact_object, actual_compact_object)

    def test_secondary_compact_object(self):
        expected_compact_object = TestCoalescence.coalescence._Coalescence__compact_objects[
            TestCoalescence.coalescence._Coalescence__secondary_compact_object_index]
        actual_compact_object = TestCoalescence.coalescence.secondary_compact_object
        self.assertEqual(expected_compact_object, actual_compact_object)

    def test_final_compact_object(self):
        expected_compact_object = TestCoalescence.coalescence._Coalescence__compact_objects[
            TestCoalescence.coalescence._Coalescence__final_compact_object_index]
        actual_compact_object = TestCoalescence.coalescence.final_compact_object
        self.assertEqual(expected_compact_object, actual_compact_object)

    def test_compact_objects(self):
        expected_compact_objects = TestCoalescence.coalescence._Coalescence__compact_objects
        actual_compact_objects = TestCoalescence.coalescence.compact_objects
        self.assertTrue(np.all(expected_compact_objects == actual_compact_objects))

    def test_radiation_bundle(self):
        expected_radiation_bundle = TestCoalescence.coalescence._Coalescence__radiation_mode_bundle
        actual_radiation_bundle = TestCoalescence.coalescence.radiationbundle
        self.assertEqual(expected_radiation_bundle, actual_radiation_bundle)

    def test_compact_object_by_id(self):
        expected_compact_object = TestCoalescence.coalescence._Coalescence__compact_objects[0]
        actual_compact_object = TestCoalescence.coalescence.compact_object_by_id(0)
        self.assertEqual(expected_compact_object, actual_compact_object)

        expected_compact_object = TestCoalescence.coalescence._Coalescence__compact_objects[1]
        actual_compact_object = TestCoalescence.coalescence.compact_object_by_id(1)
        self.assertEqual(expected_compact_object, actual_compact_object)

        try:
            TestCoalescence.coalescence.compact_object_by_id(4)
            self.fail()
        except:
            pass

    def test_mass_ratio(self):
        self.assertTrue(np.isclose(1, TestCoalescence.coalescence.mass_ratio, atol=1e-4))
        self.assertTrue(TestCoalescence.coalescence._Coalescence__h5_file.attrs["mass ratio"],
                        TestCoalescence.coalescence.mass_ratio)

    def test_symmetric_mass_ratio(self):
        eta = TestCoalescence.coalescence.symmetric_mass_ratio
        q = TestCoalescence.coalescence.mass_ratio

        self.assertTrue(np.isclose(0.25, eta, atol=0.001))
        self.assertTrue(np.isclose(q / ((1 + q) ** 2), eta, atol=0.001))

        q = 4
        primary_object = TestCoalescence.coalescence.primary_compact_object
        secondary_object = TestCoalescence.coalescence.secondary_compact_object
        with patch('mayawaves.compactobject.CompactObject.initial_horizon_mass', 0.0):
            primary_object.initial_horizon_mass = 0.8
            secondary_object.initial_horizon_mass = 0.2
            eta = TestCoalescence.coalescence.symmetric_mass_ratio
            self.assertTrue(np.isclose(eta, q / ((1 + q) ** 2), atol=0.001))

    def test_spin_configuration(self):
        self.assertEqual("non-spinning", TestCoalescence.coalescence.spin_configuration)
        self.assertEqual(TestCoalescence.coalescence._Coalescence__h5_file.attrs["spin configuration"],
                         TestCoalescence.coalescence.spin_configuration)

    def test_initial_separation(self):
        self.assertTrue(np.isclose(TestCoalescence.coalescence.initial_separation, 2.33, atol=1e-2))
        self.assertTrue(TestCoalescence.coalescence._Coalescence__h5_file.attrs["initial separation"],
                        TestCoalescence.coalescence.initial_separation)

    def test_initial_orbital_frequency(self):
        self.assertTrue(np.isclose(TestCoalescence.coalescence.initial_orbital_frequency, 0.43, atol=1e-2))
        self.assertTrue(TestCoalescence.coalescence._Coalescence__h5_file.attrs["initial orbital frequency"],
                        TestCoalescence.coalescence.initial_orbital_frequency)

    def test_separation_vector(self):
        generated_time, generated_separation_vector = TestCoalescence.coalescence.separation_vector

        shifttracker_time = np.loadtxt(os.path.join(TestCoalescence.CURR_DIR,
                                                    "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc") % TestCoalescence.simulation_name,
                                       usecols=1)
        shifttracker_position0 = np.loadtxt(os.path.join(TestCoalescence.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc") % TestCoalescence.simulation_name,
                                            usecols=(2, 3, 4))
        shifttracker_position1 = np.loadtxt(os.path.join(TestCoalescence.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ShiftTracker1.asc") % TestCoalescence.simulation_name,
                                            usecols=(2, 3, 4))
        shifttracker_separation = shifttracker_position1 - shifttracker_position0

        time, generated_indices, shifttracker_indices = np.intersect1d(generated_time, shifttracker_time,
                                                                       return_indices=True)
        generated_separation_vector = generated_separation_vector[generated_indices]
        expected_separation_vector = shifttracker_separation[shifttracker_indices]

        self.assertTrue(np.all(generated_separation_vector == expected_separation_vector))

    @mock.patch("mayawaves.coalescence.Coalescence.separation_vector", new_callable=PropertyMock)
    def test_orbital_angular_momentum_unit_vector(self, coalescence_separation_vector):
        shifttracker_0_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/precessing/ShiftTracker0.asc"), usecols=(1, 2, 3, 4))
        time = shifttracker_0_data[:, 0]
        position_0 = shifttracker_0_data[:, 1:]

        shifttracker_1_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/precessing/ShiftTracker1.asc"), usecols=(1, 2, 3, 4))
        position_1 = shifttracker_1_data[:, 1:]

        separation_vector = position_1 - position_0

        cut_index = np.argmax(time > 200)
        time = time[:cut_index]
        separation_vector = separation_vector[:cut_index]

        coalescence_separation_vector.return_value = (time, separation_vector)

        generated_time, generated_orbital_angular_momentum_unit_vector = TestCoalescence.coalescence.orbital_angular_momentum_unit_vector

        # check if orthogonal to separation vector
        dot_separation_angular_momentum = np.sum(generated_orbital_angular_momentum_unit_vector * separation_vector,
                                                 axis=1)
        self.assertTrue(np.all(np.abs(dot_separation_angular_momentum < 1e-5)))

        # check if orthogonal to derivative of separation vector
        separation_derivative = np.gradient(separation_vector, axis=0) / np.gradient(time).reshape((len(time), 1))
        dot_separation_derivative_angular_momentum = np.sum(
            generated_orbital_angular_momentum_unit_vector * separation_derivative, axis=1)
        self.assertTrue(np.all(dot_separation_derivative_angular_momentum < 3e-5))

    @mock.patch("mayawaves.coalescence.Coalescence.separation_vector", new_callable=PropertyMock)
    def test_orbital_phase_in_xy_plane(self, coalescence_separation_vector):
        shifttracker_0_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/precessing/ShiftTracker0.asc"), usecols=(1, 2, 3, 4))
        time = shifttracker_0_data[:, 0]
        position_0 = shifttracker_0_data[:, 1:]

        shifttracker_1_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/precessing/ShiftTracker1.asc"), usecols=(1, 2, 3, 4))
        position_1 = shifttracker_1_data[:, 1:]

        separation_vector = position_1 - position_0

        cut_index = np.argmax(time > 200)
        time = time[:cut_index]
        separation_vector = separation_vector[:cut_index]

        coalescence_separation_vector.return_value = (time, separation_vector)

        generated_time, generated_orbital_phase = TestCoalescence.coalescence.orbital_phase_in_xy_plane

        xy_phase = np.unwrap(np.arctan2(separation_vector[:, 1], separation_vector[:, 0]))

        self.assertTrue(np.allclose(xy_phase, generated_orbital_phase, atol=1e-3))

    @mock.patch("mayawaves.coalescence.Coalescence.separation_vector", new_callable=PropertyMock)
    def test_orbital_frequency(self, coalescence_separation_vector):
        shifttracker_0_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/precessing/ShiftTracker0.asc"), usecols=(1, 2, 3, 4))
        time = shifttracker_0_data[:, 0]
        position_0 = shifttracker_0_data[:, 1:]

        shifttracker_1_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/precessing/ShiftTracker1.asc"), usecols=(1, 2, 3, 4))
        position_1 = shifttracker_1_data[:, 1:]

        separation_vector = position_1 - position_0

        cut_index = np.argmax(time > 200)
        time = time[:cut_index]
        separation_vector = separation_vector[:cut_index]

        coalescence_separation_vector.return_value = (time, separation_vector)

        generated_time, generated_orbital_frequency = TestCoalescence.coalescence.orbital_frequency

        # check if close to derivative of phase in x-y plane. WON'T BE EXACT!
        xy_phase = np.arctan2(separation_vector[:, 1], separation_vector[:, 0])
        xy_frequency = np.gradient(xy_phase) / np.gradient(time)

        difference = abs(xy_frequency - generated_orbital_frequency)
        difference[difference > 100] = np.nan

        average_difference = np.nanmedian(difference)
        self.assertTrue(average_difference < 1e-4)

    def test_separation_at_time(self):
        shifttracker_time = np.loadtxt(os.path.join(TestCoalescence.CURR_DIR,
                                                    "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc") % TestCoalescence.simulation_name,
                                       usecols=1)
        shifttracker_position0 = np.loadtxt(os.path.join(TestCoalescence.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc") % TestCoalescence.simulation_name,
                                            usecols=(2, 3, 4))
        shifttracker_position1 = np.loadtxt(os.path.join(TestCoalescence.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ShiftTracker1.asc") % TestCoalescence.simulation_name,
                                            usecols=(2, 3, 4))
        shifttracker_separation_vector = shifttracker_position1 - shifttracker_position0
        shifttracker_separation_magnitude = np.linalg.norm(shifttracker_separation_vector, axis=1)

        generated_separation_at_time_0 = TestCoalescence.coalescence.separation_at_time(desired_time=0)
        expected_separation_at_time_0 = shifttracker_separation_magnitude[np.argmax(shifttracker_time > 0)]
        self.assertEqual(expected_separation_at_time_0, generated_separation_at_time_0)

        generated_separation_at_time_10 = TestCoalescence.coalescence.separation_at_time(desired_time=10)
        expected_separation_at_time_10 = shifttracker_separation_magnitude[np.argmax(shifttracker_time > 10)]
        self.assertEqual(expected_separation_at_time_10, generated_separation_at_time_10)

        generated_separation_at_time_20 = TestCoalescence.coalescence.separation_at_time(desired_time=20)
        expected_separation_at_time_20 = shifttracker_separation_magnitude[np.argmax(shifttracker_time > 20)]
        self.assertEqual(expected_separation_at_time_20, generated_separation_at_time_20)

    def test_orbital_frequency_at_time(self):
        generated_orbital_frequency_20M = TestCoalescence.coalescence.orbital_frequency_at_time(20)
        expected_orbital_frequency_20M = 0.287
        self.assertTrue(np.isclose(expected_orbital_frequency_20M, generated_orbital_frequency_20M, atol=1e-3))

    # todo test precessing case
    def test_orbital_angular_momentum_unit_vector_at_time(self):
        generated_angular_momentum_unit_vector_at_20M = TestCoalescence.coalescence.orbital_angular_momentum_unit_vector_at_time(
            20)
        expected_angular_momentum_unit_vector_at_20M = [0, 0, 1]
        self.assertTrue(np.all(
            np.isclose(generated_angular_momentum_unit_vector_at_20M, expected_angular_momentum_unit_vector_at_20M,
                       atol=0.001)))

    def test_separation_unit_vector_at_time(self):
        generated_separation_unit_vector_at_20M = TestCoalescence.coalescence.separation_unit_vector_at_time(20)
        expected_separation_unit_vector_at_20M = (0.922, 0.387, 0)
        self.assertTrue(np.all(
            np.isclose(generated_separation_unit_vector_at_20M, expected_separation_unit_vector_at_20M, atol=0.001)))

    def test__crop_to_three_or_four_orbits(self):
        time = np.linspace(0, 10000, 1000)
        orbital_phase = 0.01 * time
        start_time = 150
        data = 0.5 * time + np.power(time, 2)
        separation_magnitude = np.linspace(10, 0, 1000)
        cropped_time, cropped_data = self.coalescence._crop_to_three_or_four_orbits(start_time, time, orbital_phase,
                                                                                    separation_magnitude, data)
        initial_index = np.argmax(time > start_time)
        end_index = np.argmax(orbital_phase > (orbital_phase[initial_index] + 8 * np.pi))
        expected_cropped_time = time[initial_index: end_index]
        expected_cropped_data = data[initial_index: end_index]
        self.assertTrue(np.all(expected_cropped_data == cropped_data))
        self.assertTrue(np.all(expected_cropped_time == cropped_time))

        # not enough data before merger to do full 4 orbits
        # merge time > 500
        time = np.linspace(0, 10000, 1000)
        orbital_phase = 0.0005 * time
        start_time = 150
        data = 0.5 * time + np.power(time, 2)
        separation_magnitude = np.linspace(10, 0, 1000)
        cropped_time, cropped_data = self.coalescence._crop_to_three_or_four_orbits(start_time, time, orbital_phase,
                                                                                    separation_magnitude, data)
        self.assertIsNone(cropped_time)
        self.assertIsNone(cropped_data)

        # merge time < 500
        time = np.linspace(0, 1000, 1000)
        orbital_phase = 0.005 * time
        start_time = 150
        data = 0.5 * time + np.power(time, 2)
        separation_magnitude = np.linspace(0.11, 0, 1000)
        cropped_time, cropped_data = self.coalescence._crop_to_three_or_four_orbits(start_time, time, orbital_phase,
                                                                                    separation_magnitude, data)
        initial_index = np.argmax(time > start_time)
        self.assertIsNone(cropped_time)
        self.assertIsNone(cropped_data)

        # there aren't 4 orbits
        time = np.linspace(0, 10000, 1000)
        orbital_phase = 0.001 * time
        start_time = 150
        data = 0.5 * time + np.power(time, 2)
        separation_magnitude = np.linspace(10, 0, 1000)
        cropped_time, cropped_data = self.coalescence._crop_to_three_or_four_orbits(start_time, time, orbital_phase,
                                                                                    separation_magnitude, data)
        initial_index = np.argmax(time > start_time)
        self.assertIsNone(cropped_time)
        self.assertIsNone(cropped_data)

        # doesn't merge
        time = np.linspace(0, 10000, 1000)
        orbital_phase = 0.001 * time
        start_time = 150
        data = 0.5 * time + np.power(time, 2)
        separation_magnitude = np.linspace(10, 1, 1000)
        end_index = np.argmax(time > (10000 - 100))
        cropped_time, cropped_data = self.coalescence._crop_to_three_or_four_orbits(start_time, time, orbital_phase,
                                                                                    separation_magnitude, data)
        self.assertIsNone(cropped_time)
        self.assertIsNone(cropped_data)

        # end time is less than 50 from start time
        time = np.linspace(0, 1000, 1000)
        orbital_phase = 0.005 * time
        start_time = 150
        data = 0.5 * time + np.power(time, 2)
        separation_magnitude = np.linspace(0.11, 0, 1000)
        cropped_time, cropped_data = self.coalescence._crop_to_three_or_four_orbits(start_time, time, orbital_phase,
                                                                                    separation_magnitude, data)
        self.assertIsNone(cropped_time)
        self.assertIsNone(cropped_data)

        time = np.linspace(0, 1000, 1000)
        orbital_phase = 0.005 * time
        start_time = 150
        data = 0.5 * time + np.power(time, 2)
        separation_magnitude = np.linspace(0.115, 0, 1000)
        cropped_time, cropped_data = self.coalescence._crop_to_three_or_four_orbits(start_time, time, orbital_phase,
                                                                                    separation_magnitude, data)
        self.assertIsNone(cropped_time)
        self.assertIsNone(cropped_data)

    def test__anomaly_from_apsis_times(self):
        periapsis_times = np.array([0, 1, 2, 3, 4])
        apoapsis_times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        desired_time = 0.25
        expected_anomaly = 2 * np.pi * (desired_time - 0) / (1 - 0)
        mean_anomaly = self.coalescence._anomaly_from_apsis_times(periapsis_times, apoapsis_times, desired_time)
        self.assertEqual(expected_anomaly, mean_anomaly)

        desired_time = 0.75
        expected_anomaly = 2 * np.pi * (desired_time - 0) / (1 - 0)
        mean_anomaly = self.coalescence._anomaly_from_apsis_times(periapsis_times, apoapsis_times, desired_time)
        self.assertEqual(expected_anomaly, mean_anomaly)

        desired_time = 0
        expected_anomaly = 2 * np.pi * (desired_time - 0) / (1 - 0)
        mean_anomaly = self.coalescence._anomaly_from_apsis_times(periapsis_times, apoapsis_times, desired_time)
        self.assertEqual(expected_anomaly, mean_anomaly)

        desired_time = 1
        expected_anomaly = 2 * np.pi * (desired_time - 1) / (1 - 0)
        mean_anomaly = self.coalescence._anomaly_from_apsis_times(periapsis_times, apoapsis_times, desired_time)
        self.assertEqual(expected_anomaly, mean_anomaly)

        desired_time = 1.25
        expected_anomaly = 2 * np.pi * (desired_time - 1) / (2 - 1)
        mean_anomaly = self.coalescence._anomaly_from_apsis_times(periapsis_times, apoapsis_times, desired_time)
        self.assertEqual(expected_anomaly, mean_anomaly)

        periapsis_times = np.array([1, 2, 3, 4])
        apoapsis_times = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        desired_time = 0.25
        expected_anomaly = 2 * np.pi * (desired_time - 0) / (1 - 0)
        mean_anomaly = self.coalescence._anomaly_from_apsis_times(periapsis_times, apoapsis_times, desired_time)
        self.assertEqual(expected_anomaly, mean_anomaly)

    def test__anomaly_from_eccentric_timeseries(self):
        time = np.linspace(0, 200, 1000)
        freq = np.pi / 20
        period = 2 * np.pi / freq
        eccentric_timeseries = 0.02 * np.sin(freq * time)
        desired_time = 20
        periapsis = - period * 0.25
        expected_anomaly = 2 * np.pi * (
                    ((desired_time - periapsis) / period) - int((desired_time - periapsis) / period))
        mean_anomaly = self.coalescence._anomaly_from_eccentric_timeseries(time, eccentric_timeseries, desired_time)
        self.assertTrue(np.isclose(expected_anomaly, mean_anomaly, atol=0.05))

        time = np.linspace(0, 200, 1000)
        freq = np.pi / 20
        eccentric_timeseries = 0.02 * np.sin(freq * time)
        desired_time = 30
        mean_anomaly = self.coalescence._anomaly_from_eccentric_timeseries(time, eccentric_timeseries, desired_time)
        self.assertTrue(np.isclose(0, mean_anomaly, atol=0.05) or np.isclose(2 * np.pi, mean_anomaly, atol=0.05))

        time = np.linspace(0, 200, 1000)
        freq = np.pi / 20
        period = 2 * np.pi / freq
        eccentric_timeseries = 0.02 * np.sin(freq * time)
        desired_time = 13
        periapsis = - period * 0.25
        expected_anomaly = 2 * np.pi * (
                ((desired_time - periapsis) / period) - int((desired_time - periapsis) / period))
        mean_anomaly = self.coalescence._anomaly_from_eccentric_timeseries(time, eccentric_timeseries, desired_time)
        self.assertTrue(np.isclose(expected_anomaly, mean_anomaly, atol=0.05))

        time = np.linspace(0, 200, 1000)
        freq = np.pi / 20
        period = 2 * np.pi / freq
        eccentric_timeseries = 0.02 * np.sin(freq * time)
        desired_time = 111
        periapsis = - period * 0.25
        expected_anomaly = 2 * np.pi * (
                ((desired_time - periapsis) / period) - int((desired_time - periapsis) / period))
        mean_anomaly = self.coalescence._anomaly_from_eccentric_timeseries(time, eccentric_timeseries, desired_time)
        self.assertTrue(np.isclose(expected_anomaly, mean_anomaly, atol=0.05))

    @mock.patch("mayawaves.coalescence.Coalescence.separation_vector", new_callable=PropertyMock)
    @mock.patch("mayawaves.coalescence.Coalescence.symmetric_mass_ratio", new_callable=PropertyMock)
    @mock.patch("mayawaves.coalescence.Coalescence.mass_ratio", new_callable=PropertyMock)
    @mock.patch("mayawaves.compactobject.CompactObject.initial_dimensionless_spin", new_callable=PropertyMock)
    @mock.patch("mayawaves.compactobject.CompactObject.initial_dimensional_spin", new_callable=PropertyMock)
    @mock.patch("mayawaves.compactobject.CompactObject.momentum_vector", new_callable=PropertyMock)
    def test_eccentricity_and_mean_anomaly_at_time(self, compactobject_momentum_vector,
                                                   compactobject_initial_dimensional_spin,
                                                   compactobject_initial_dimensionless_spin, coalescence_mass_ratio,
                                                   coalescence_symmetric_mass_ratio, coalescence_separation_vector):
        compactobject_initial_dimensional_spin.return_value = np.array([0, 0, 0])
        compactobject_initial_dimensionless_spin.side_effect = [np.array([0, 0, 0]), np.array([0, 0, 0.8])]
        compactobject_momentum_vector.return_value = (
        np.array([0.000000000000000000e+00, 8.333333333333343973e-02, 1.666666666666668795e-01]),
        np.array([[-1.458339075119886505e-04, 3.953053093924969313e-02, 1.212470436223733352e-13],
                  [-1.637970488626104136e-04, 3.955624016580517460e-02, 2.923097326974689185e-11],
                  [-1.854114063024255596e-04, 3.958181403059909953e-02, 4.356985234386932981e-11]]))
        coalescence_mass_ratio.return_value = 7
        coalescence_symmetric_mass_ratio.return_value = 0.109375

        shifttracker0_filepath = os.path.join(TestCoalescence.CURR_DIR,
                                              "resources/eccentricity/D11_q7_a1_0_0_0_a2_0_0_0.8_m384/ShiftTracker0.asc")
        shifttracker1_filepath = os.path.join(TestCoalescence.CURR_DIR,
                                              "resources/eccentricity/D11_q7_a1_0_0_0_a2_0_0_0.8_m384/ShiftTracker1.asc")

        time = np.loadtxt(shifttracker0_filepath, usecols=1)
        position_0 = np.loadtxt(shifttracker0_filepath, usecols=(2, 3, 4))
        position_1 = np.loadtxt(shifttracker1_filepath, usecols=(2, 3, 4))

        separation_vector = position_1 - position_0

        coalescence_separation_vector.return_value = (time, separation_vector)

        psi4_max_time_22 = 3880.8333
        with patch.object(Coalescence, 'psi4_max_time_for_mode',
                          return_value=psi4_max_time_22):
            eccentricity, mean_anomaly = TestCoalescence.coalescence.eccentricity_and_mean_anomaly_at_time(150, 200)

        self.assertTrue(np.isclose(3.35, mean_anomaly, atol=5e-1))
        self.assertTrue(np.isclose(0.009, eccentricity, atol=1e-2))

        # q1 nonspinning
        compactobject_initial_dimensional_spin.return_value = np.array([0, 0, 0])
        compactobject_initial_dimensionless_spin.side_effect = [np.array([0, 0, 0]), np.array([0, 0, 0])]
        compactobject_momentum_vector.return_value = (
            np.array([0.000000000000000000e+00, 1.333333333439999935e-01, 2.666666666879999870e-01]),
            np.array([[-7.119936250646606872e-04, 8.640906335812675865e-02, -6.551820353368876662e-13],
                      [-7.862983617107136263e-04, 8.657925920618150806e-02, -8.478040608480163182e-13],
                      [-8.711803180260802330e-04, 8.688562397722053576e-02, -3.095867610769775195e-14]]))
        coalescence_mass_ratio.return_value = 1
        coalescence_symmetric_mass_ratio.return_value = 0.25

        shifttracker0_filepath = os.path.join(TestCoalescence.CURR_DIR,
                                              "resources/eccentricity/D11_q1_a1_0_0_0_a2_0_0_0_m240_e0.08/ShiftTracker0.asc")
        shifttracker1_filepath = os.path.join(TestCoalescence.CURR_DIR,
                                              "resources/eccentricity/D11_q1_a1_0_0_0_a2_0_0_0_m240_e0.08/ShiftTracker1.asc")

        time = np.loadtxt(shifttracker0_filepath, usecols=1)
        position_0 = np.loadtxt(shifttracker0_filepath, usecols=(2, 3, 4))
        position_1 = np.loadtxt(shifttracker1_filepath, usecols=(2, 3, 4))

        separation_vector = position_1 - position_0

        coalescence_separation_vector.return_value = (time, separation_vector)

        psi4_max_time_22 = 847.4667
        with patch.object(Coalescence, 'psi4_max_time_for_mode',
                          return_value=psi4_max_time_22):
            eccentricity, mean_anomaly = TestCoalescence.coalescence.eccentricity_and_mean_anomaly_at_time(150, 200)

        self.assertTrue(np.isclose(0.75, mean_anomaly, atol=5e-1))
        self.assertTrue(np.isclose(0.08, eccentricity, atol=5e-2))

        # q3 nonspinning
        compactobject_initial_dimensional_spin.return_value = np.array([0, 0, 0])
        compactobject_initial_dimensionless_spin.side_effect = [np.array([0, 0, 0]), np.array([0, 0, 0])]
        compactobject_momentum_vector.return_value = (
            np.array([0.000000000000000000e+00, 1.333333332999999954e-01, 2.666666666999999991e-01]),
            np.array([[-4.160709049710624354e-04, 6.642651293652294953e-02, 8.367124102548930339e-14],
                      [-4.709808996302757857e-04, 6.650814747434080632e-02, -6.775444198083689958e-12],
                      [-5.299727078054737106e-04, 6.661119332591292075e-02, -1.339124768043367087e-11]]))
        coalescence_mass_ratio.return_value = 3
        coalescence_symmetric_mass_ratio.return_value = 0.1875

        shifttracker0_filepath = os.path.join(TestCoalescence.CURR_DIR,
                                              "resources/eccentricity/D11_q3_a1_0_0_0_a2_0_0_0_m240_e0.04/ShiftTracker0.asc")
        shifttracker1_filepath = os.path.join(TestCoalescence.CURR_DIR,
                                              "resources/eccentricity/D11_q3_a1_0_0_0_a2_0_0_0_m240_e0.04/ShiftTracker1.asc")

        time = np.loadtxt(shifttracker0_filepath, usecols=1)
        position_0 = np.loadtxt(shifttracker0_filepath, usecols=(2, 3, 4))
        position_1 = np.loadtxt(shifttracker1_filepath, usecols=(2, 3, 4))

        separation_vector = position_1 - position_0

        coalescence_separation_vector.return_value = (time, separation_vector)

        psi4_max_time_22 = 1362
        with patch.object(Coalescence, 'psi4_max_time_for_mode',
                          return_value=psi4_max_time_22):
            eccentricity, mean_anomaly = TestCoalescence.coalescence.eccentricity_and_mean_anomaly_at_time(150, 200)

        self.assertTrue(np.isclose(0.4, mean_anomaly, atol=5e-1))
        self.assertTrue(np.isclose(0.04, eccentricity, atol=3e-2))

    def test_average_run_speed(self):
        self.assertTrue(np.isclose(TestCoalescence.coalescence.average_run_speed, 54.5613306725, atol=1e-4))

        #runstats not available
        temp_coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                                    "resources/sample_etk_simulations/GW150914.h5"))
        runspeed = temp_coalescence.average_run_speed
        temp_coalescence.close()
        self.assertIsNone(runspeed)

    def test_l_max(self):
        self.assertEqual(TestCoalescence.coalescence.l_max, 4)

    def test_included_modes(self):
        expected_modes = [(2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (3, -3), (3, -2), (3, -1), (3, 0), (3, 1), (3, 2),
                          (3, 3), (4, -4), (4, -3), (4, -2), (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
        expected_modes.sort()
        generated_modes = TestCoalescence.coalescence.included_modes
        generated_modes.sort()
        self.assertTrue(np.all(generated_modes == expected_modes))

    def test_included_extraction_radii(self):
        expected_radii = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        expected_radii.sort()
        generated_radii = TestCoalescence.coalescence.included_extraction_radii
        generated_radii.sort()
        self.assertTrue(np.all(generated_radii == expected_radii))

    @mock.patch("mayawaves.coalescence.Coalescence.separation_vector", new_callable=PropertyMock)
    def test_merge_time(self, coalescence_separation_vector):
        separation_magnitude = np.arange(1, 0, -0.001)
        separation_phase = np.linspace(0, 6 * np.pi, len(separation_magnitude))
        time = np.linspace(0, 100, len(separation_magnitude))
        expected_merge_time = 99.0990990990991

        separation_vector = np.zeros((len(separation_magnitude), 3))
        separation_vector[:, 0] = separation_magnitude * np.cos(separation_phase)
        separation_vector[:, 1] = separation_magnitude * np.sin(separation_phase)

        coalescence_separation_vector.return_value = (time, separation_vector)

        generated_merge_time = TestCoalescence.coalescence.merge_time

        self.assertEqual(expected_merge_time, generated_merge_time)

    def test_radiation_frame(self):
        from mayawaves.radiation import Frame

        TestCoalescence.coalescence.radiationbundle._RadiationBundle__frame = Frame.RAW
        self.assertEqual(Frame.RAW, TestCoalescence.coalescence.radiation_frame)

        TestCoalescence.coalescence.radiationbundle._RadiationBundle__frame = Frame.COM_CORRECTED
        self.assertEqual(Frame.COM_CORRECTED, TestCoalescence.coalescence.radiation_frame)

    def test_set_radiation_frame(self):
        from mayawaves.radiation import Frame

        # center of mass
        #  center of mass data is None
        with patch.object(RadiationBundle, 'set_frame') as mock_set_frame:
            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(None, None)) as mock_center_of_mass:
                TestCoalescence.coalescence.set_radiation_frame(center_of_mass_corrected=True)

                mock_center_of_mass.assert_called_once()
                mock_set_frame.assert_not_called()

        #  center of mass data is not None
        with patch.object(RadiationBundle, 'set_frame') as mock_set_frame:
            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(
                              np.array([1, 2, 3]),
                              np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))) as mock_center_of_mass:
                TestCoalescence.coalescence.set_radiation_frame(center_of_mass_corrected=True)

                mock_center_of_mass.assert_called_once()
                self.assertEqual(Frame.COM_CORRECTED, mock_set_frame.call_args[0][0])
                self.assertEqual(2, len(mock_set_frame.call_args[1]))
                self.assertTrue(np.all(np.array([1, 2, 3]) == mock_set_frame.call_args[1]['time']))
                self.assertTrue(np.all(
                    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == mock_set_frame.call_args[1]['center_of_mass']))

        # resetting to raw frame
        with patch.object(RadiationBundle, 'set_frame') as mock_set_frame:
            TestCoalescence.coalescence.set_radiation_frame()
            mock_set_frame.assert_called_once_with(Frame.RAW)

    @mock.patch("mayawaves.compactobject.CompactObject.initial_horizon_mass", new_callable=PropertyMock)
    @mock.patch("mayawaves.compactobject.CompactObject.horizon_mass", new_callable=PropertyMock)
    @mock.patch("mayawaves.compactobject.CompactObject.position_vector", new_callable=PropertyMock)
    def test_center_of_mass(self, compactobject_position_vector, compactobject_horizon_mass,
                            compactobject_initial_horizon_mass):
        # if not enough mass data to use mass evolution
        time_pos_0 = np.array([0, 10, 20, 30, 40])
        position_vector_0x = [1, 0.9, 0.8, 0.7, 0.6]
        position_vector_0y = [0, 0, 0, 0, 0]
        position_vector_0z = [0, 0, 0, 0, 0]
        position_vector_0 = np.column_stack([position_vector_0x, position_vector_0y, position_vector_0z])
        position_0_return_value = (time_pos_0, position_vector_0)

        time_pos_1 = np.array([10, 20, 30, 40])
        position_vector_1x = [-2, -1.8, -1.6, -1.4]
        position_vector_1y = [0, 0, 0, 0]
        position_vector_1z = [0, 0, 0, 0]
        position_vector_1 = np.column_stack([position_vector_1x, position_vector_1y, position_vector_1z])
        position_1_return_value = (time_pos_1, position_vector_1)

        compactobject_position_vector.side_effect = [position_0_return_value, position_1_return_value]

        time_mass_0 = np.array([0, 10])
        horizon_mass_0 = np.array([0.2, 0.1])
        mass_0_return_value = (time_mass_0, horizon_mass_0)

        time_mass_1 = np.array([10])
        horizon_mass_1 = np.array([0.4])
        mass_1_return_value = (time_mass_1, horizon_mass_1)

        compactobject_horizon_mass.side_effect = [mass_0_return_value, mass_1_return_value]

        initial_horizon_mass_0 = 0.3
        initial_horizon_mass_1 = 0.6

        compactobject_initial_horizon_mass.side_effect = [initial_horizon_mass_0, initial_horizon_mass_1]

        expected_time = np.array([10, 20, 30, 40])
        expected_com = np.array([[-0.93, 0, 0], [-0.84, 0, 0], [-0.75, 0, 0], [-0.66, 0, 0]])

        generated_time, generated_com = TestCoalescence.coalescence.center_of_mass

        self.assertTrue(np.all(expected_time == generated_time))
        self.assertTrue(np.allclose(expected_com, generated_com, atol=1e-6))

        # if there is enough mass data to use mass evolution
        time_pos_0 = np.array([0, 10, 20, 30, 40])
        position_vector_0x = [0, 0, 0, 0, 0]
        position_vector_0y = [1, 0.9, 0.8, 0.7, 0.6]
        position_vector_0z = [0, 0, 0, 0, 0]
        position_vector_0 = np.column_stack([position_vector_0x, position_vector_0y, position_vector_0z])
        position_0_return_value = (time_pos_0, position_vector_0)

        time_pos_1 = np.array([10, 20, 30, 40])
        position_vector_1x = [0, 0, 0, 0]
        position_vector_1y = [0, 0, 0, 0]
        position_vector_1z = [-2, -1.8, -1.6, -1.4]
        position_vector_1 = np.column_stack([position_vector_1x, position_vector_1y, position_vector_1z])
        position_1_return_value = (time_pos_1, position_vector_1)

        compactobject_position_vector.side_effect = [position_0_return_value, position_1_return_value]

        time_mass_0 = np.array([0, 10, 20, 30, 40])
        horizon_mass_0 = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        mass_0_return_value = (time_mass_0, horizon_mass_0)

        time_mass_1 = np.array([20, 30, 40])
        horizon_mass_1 = np.array([0.4, 0.3, 0.2])
        mass_1_return_value = (time_mass_1, horizon_mass_1)

        compactobject_horizon_mass.side_effect = [mass_0_return_value, mass_1_return_value]

        expected_time = np.array([20, 30, 40])
        expected_com = np.array([[0, 0.24, -0.72], [0, 0.14, -0.48], [0, 0.06, -0.28]])
        generated_time, generated_com = TestCoalescence.coalescence.center_of_mass

        self.assertTrue(np.all(expected_time == generated_time))
        self.assertTrue(np.allclose(expected_com, generated_com, atol=1e-6))

        # if there is no position data
        time_pos_0 = None
        position_vector_0 = None
        position_0_return_value = (time_pos_0, position_vector_0)

        time_pos_1 = np.array([10, 20, 30, 40])
        position_vector_1x = [-2, -1.8, -1.6, -1.4]
        position_vector_1y = [0, 0, 0, 0]
        position_vector_1z = [0, 0, 0, 0]
        position_vector_1 = np.column_stack([position_vector_1x, position_vector_1y, position_vector_1z])
        position_1_return_value = (time_pos_1, position_vector_1)

        compactobject_position_vector.side_effect = [position_0_return_value, position_1_return_value]

        time_mass_0 = np.array([0, 10])
        horizon_mass_0 = np.array([0.2, 0.1])
        mass_0_return_value = (time_mass_0, horizon_mass_0)

        time_mass_1 = np.array([10])
        horizon_mass_1 = np.array([0.4])
        mass_1_return_value = (time_mass_1, horizon_mass_1)

        compactobject_horizon_mass.side_effect = [mass_0_return_value, mass_1_return_value]

        initial_horizon_mass_0 = 0.3
        initial_horizon_mass_1 = 0.6

        compactobject_initial_horizon_mass.side_effect = [initial_horizon_mass_0, initial_horizon_mass_1]

        generated_time, generated_com = TestCoalescence.coalescence.center_of_mass

        self.assertIsNone(generated_time)
        self.assertIsNone(generated_com)

        # if there is no position data
        time_pos_1 = None
        position_vector_1 = None
        position_1_return_value = (time_pos_1, position_vector_1)

        time_pos_0 = np.array([10, 20, 30, 40])
        position_vector_0x = [-2, -1.8, -1.6, -1.4]
        position_vector_0y = [0, 0, 0, 0]
        position_vector_0z = [0, 0, 0, 0]
        position_vector_0 = np.column_stack([position_vector_0x, position_vector_0y, position_vector_0z])
        position_0_return_value = (time_pos_0, position_vector_0)

        compactobject_position_vector.side_effect = [position_0_return_value, position_1_return_value]

        time_mass_0 = np.array([0, 10])
        horizon_mass_0 = np.array([0.2, 0.1])
        mass_0_return_value = (time_mass_0, horizon_mass_0)

        time_mass_1 = np.array([10])
        horizon_mass_1 = np.array([0.4])
        mass_1_return_value = (time_mass_1, horizon_mass_1)

        compactobject_horizon_mass.side_effect = [mass_0_return_value, mass_1_return_value]

        initial_horizon_mass_0 = 0.3
        initial_horizon_mass_1 = 0.6

        compactobject_initial_horizon_mass.side_effect = [initial_horizon_mass_0, initial_horizon_mass_1]

        generated_time, generated_com = TestCoalescence.coalescence.center_of_mass

        self.assertIsNone(generated_time)
        self.assertIsNone(generated_com)

        # if mass data is None
        time_pos_0 = np.array([0, 10, 20, 30, 40])
        position_vector_0x = [1, 0.9, 0.8, 0.7, 0.6]
        position_vector_0y = [0, 0, 0, 0, 0]
        position_vector_0z = [0, 0, 0, 0, 0]
        position_vector_0 = np.column_stack([position_vector_0x, position_vector_0y, position_vector_0z])
        position_0_return_value = (time_pos_0, position_vector_0)

        time_pos_1 = np.array([10, 20, 30, 40])
        position_vector_1x = [-2, -1.8, -1.6, -1.4]
        position_vector_1y = [0, 0, 0, 0]
        position_vector_1z = [0, 0, 0, 0]
        position_vector_1 = np.column_stack([position_vector_1x, position_vector_1y, position_vector_1z])
        position_1_return_value = (time_pos_1, position_vector_1)

        compactobject_position_vector.side_effect = [position_0_return_value, position_1_return_value]

        time_mass_0 = None
        horizon_mass_0 = None
        mass_0_return_value = (time_mass_0, horizon_mass_0)

        time_mass_1 = np.array([10])
        horizon_mass_1 = np.array([0.4])
        mass_1_return_value = (time_mass_1, horizon_mass_1)

        compactobject_horizon_mass.side_effect = [mass_0_return_value, mass_1_return_value]

        initial_horizon_mass_0 = 0.3
        initial_horizon_mass_1 = 0.6

        compactobject_initial_horizon_mass.side_effect = [initial_horizon_mass_0, initial_horizon_mass_1]

        expected_time = np.array([10, 20, 30, 40])
        expected_com = np.array([[-0.93, 0, 0], [-0.84, 0, 0], [-0.75, 0, 0], [-0.66, 0, 0]])

        generated_time, generated_com = TestCoalescence.coalescence.center_of_mass

        self.assertTrue(np.all(expected_time == generated_time))
        self.assertTrue(np.allclose(expected_com, generated_com, atol=1e-6))

        # if mass data is None
        time_pos_0 = np.array([0, 10, 20, 30, 40])
        position_vector_0x = [1, 0.9, 0.8, 0.7, 0.6]
        position_vector_0y = [0, 0, 0, 0, 0]
        position_vector_0z = [0, 0, 0, 0, 0]
        position_vector_0 = np.column_stack([position_vector_0x, position_vector_0y, position_vector_0z])
        position_0_return_value = (time_pos_0, position_vector_0)

        time_pos_1 = np.array([10, 20, 30, 40])
        position_vector_1x = [-2, -1.8, -1.6, -1.4]
        position_vector_1y = [0, 0, 0, 0]
        position_vector_1z = [0, 0, 0, 0]
        position_vector_1 = np.column_stack([position_vector_1x, position_vector_1y, position_vector_1z])
        position_1_return_value = (time_pos_1, position_vector_1)

        compactobject_position_vector.side_effect = [position_0_return_value, position_1_return_value]

        time_mass_0 = np.array([0, 10])
        horizon_mass_0 = np.array([0.2, 0.1])
        mass_0_return_value = (time_mass_0, horizon_mass_0)

        time_mass_1 = None
        horizon_mass_1 = None
        mass_1_return_value = (time_mass_1, horizon_mass_1)

        compactobject_horizon_mass.side_effect = [mass_0_return_value, mass_1_return_value]

        initial_horizon_mass_0 = 0.3
        initial_horizon_mass_1 = 0.6

        compactobject_initial_horizon_mass.side_effect = [initial_horizon_mass_0, initial_horizon_mass_1]

        expected_time = np.array([10, 20, 30, 40])
        expected_com = np.array([[-0.93, 0, 0], [-0.84, 0, 0], [-0.75, 0, 0], [-0.66, 0, 0]])

        generated_time, generated_com = TestCoalescence.coalescence.center_of_mass

        self.assertTrue(np.all(expected_time == generated_time))
        self.assertTrue(np.allclose(expected_com, generated_com, atol=1e-6))

        # if not enough mass data to use mass evolution and initial horizon mass is None
        time_pos_0 = np.array([0, 10, 20, 30, 40])
        position_vector_0x = [1, 0.9, 0.8, 0.7, 0.6]
        position_vector_0y = [0, 0, 0, 0, 0]
        position_vector_0z = [0, 0, 0, 0, 0]
        position_vector_0 = np.column_stack([position_vector_0x, position_vector_0y, position_vector_0z])
        position_0_return_value = (time_pos_0, position_vector_0)

        time_pos_1 = np.array([10, 20, 30, 40])
        position_vector_1x = [-2, -1.8, -1.6, -1.4]
        position_vector_1y = [0, 0, 0, 0]
        position_vector_1z = [0, 0, 0, 0]
        position_vector_1 = np.column_stack([position_vector_1x, position_vector_1y, position_vector_1z])
        position_1_return_value = (time_pos_1, position_vector_1)

        compactobject_position_vector.side_effect = [position_0_return_value, position_1_return_value]

        time_mass_0 = np.array([0, 10])
        horizon_mass_0 = np.array([0.2, 0.1])
        mass_0_return_value = (time_mass_0, horizon_mass_0)

        time_mass_1 = np.array([10])
        horizon_mass_1 = np.array([0.4])
        mass_1_return_value = (time_mass_1, horizon_mass_1)

        compactobject_horizon_mass.side_effect = [mass_0_return_value, mass_1_return_value]

        initial_horizon_mass_0 = None
        initial_horizon_mass_1 = 0.6

        compactobject_initial_horizon_mass.side_effect = [initial_horizon_mass_0, initial_horizon_mass_1]

        generated_time, generated_com = TestCoalescence.coalescence.center_of_mass

        self.assertIsNone(generated_time)
        self.assertIsNone(generated_com)

        # if mass data is None and initial horizon mass is None
        time_pos_0 = np.array([0, 10, 20, 30, 40])
        position_vector_0x = [1, 0.9, 0.8, 0.7, 0.6]
        position_vector_0y = [0, 0, 0, 0, 0]
        position_vector_0z = [0, 0, 0, 0, 0]
        position_vector_0 = np.column_stack([position_vector_0x, position_vector_0y, position_vector_0z])
        position_0_return_value = (time_pos_0, position_vector_0)

        time_pos_1 = np.array([10, 20, 30, 40])
        position_vector_1x = [-2, -1.8, -1.6, -1.4]
        position_vector_1y = [0, 0, 0, 0]
        position_vector_1z = [0, 0, 0, 0]
        position_vector_1 = np.column_stack([position_vector_1x, position_vector_1y, position_vector_1z])
        position_1_return_value = (time_pos_1, position_vector_1)

        compactobject_position_vector.side_effect = [position_0_return_value, position_1_return_value]

        time_mass_0 = None
        horizon_mass_0 = None
        mass_0_return_value = (time_mass_0, horizon_mass_0)

        time_mass_1 = np.array([10])
        horizon_mass_1 = np.array([0.4])
        mass_1_return_value = (time_mass_1, horizon_mass_1)

        compactobject_horizon_mass.side_effect = [mass_0_return_value, mass_1_return_value]

        initial_horizon_mass_0 = None
        initial_horizon_mass_1 = 0.6

        compactobject_initial_horizon_mass.side_effect = [initial_horizon_mass_0, initial_horizon_mass_1]

        generated_time, generated_com = TestCoalescence.coalescence.center_of_mass

        self.assertIsNone(generated_time)
        self.assertIsNone(generated_com)

    @mock.patch("mayawaves.radiation.RadiationBundle.radius_for_extrapolation", new_callable=PropertyMock)
    @mock.patch("mayawaves.coalescence.Coalescence._set_default_radius_for_extrapolation", new_callable=PropertyMock)
    def test_radius_for_extrapolation(self, mock_Coalescence_set_default_radius_for_extrapolation, mock_RadiationBundle_radius_for_extrapolation):
        # getter
        # is already set
        mock_RadiationBundle_radius_for_extrapolation.return_value = 100
        mock_RadiationBundle_radius_for_extrapolation.reset_mock()
        extrap_radius = TestCoalescence.coalescence.radius_for_extrapolation
        mock_Coalescence_set_default_radius_for_extrapolation.assert_not_called()
        self.assertEqual(2, mock_RadiationBundle_radius_for_extrapolation.call_count)
        self.assertEqual(100, extrap_radius)

        # is not already set
        mock_RadiationBundle_radius_for_extrapolation.return_value = None
        mock_RadiationBundle_radius_for_extrapolation.reset_mock()
        mock_Coalescence_set_default_radius_for_extrapolation.reset_mock()        
        extrap_radius = TestCoalescence.coalescence.radius_for_extrapolation
        mock_Coalescence_set_default_radius_for_extrapolation.assert_called_once()
        self.assertEqual(2, mock_RadiationBundle_radius_for_extrapolation.call_count)
                
        # setter
        mock_RadiationBundle_radius_for_extrapolation.reset_mock()
        mock_Coalescence_set_default_radius_for_extrapolation.reset_mock()
        TestCoalescence.coalescence.radius_for_extrapolation = 5
        mock_Coalescence_set_default_radius_for_extrapolation.assert_not_called()
        mock_RadiationBundle_radius_for_extrapolation.assert_called_once_with(5)
        
        # valid radius
        mock_RadiationBundle_radius_for_extrapolation.reset_mock()
        mock_Coalescence_set_default_radius_for_extrapolation.reset_mock()
        TestCoalescence.coalescence.radius_for_extrapolation = 40
        mock_RadiationBundle_radius_for_extrapolation.assert_called_once_with(40)

    @mock.patch("mayawaves.coalescence.Coalescence.grid_structure", new_callable=PropertyMock)
    @mock.patch("mayawaves.coalescence.Coalescence.orbital_frequency_at_time")
    @mock.patch("mayawaves.coalescence.Coalescence.merge_time", new_callable=PropertyMock)
    @mock.patch("mayawaves.radiation.RadiationBundle.get_time")
    def test__set_default_radius_for_extrapolation(self, mock_get_time, mock_merge_time, mock_orbital_frequency_at_time, mock_grid_structure):
        # extraction radii included in sample simulation
        # [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]

        # case where there is a radius with dx < 1 / (2 * orbital frequency at merger)
        grid_structure = {
            1:{
                'center': [0, 0, 0],
                'levels':{
                    0:{
                        'dx': 3,
                        'radius': 400
                    },
                    1:{
                        'dx': 1.5,
                        'radius': 200
                    },
                    2:{
                        'dx': 0.75,
                        'radius': 100
                    },
                    3:{
                        'dx': 0.375,
                        'radius': 50
                    },
                    4:{
                        'dx': 0.1875,
                        'radius': 25
                    },
                    5:{
                        'dx': 0.09375,
                        'radius': 12.5
                    }
                }
            }
        }

        orbital_frequency_at_merger = 0.5
        expected_default_radius = 90
        
        mock_grid_structure.return_value = grid_structure
        mock_orbital_frequency_at_time.return_value = orbital_frequency_at_merger
        mock_merge_time.return_value = 1500
        mock_get_time.return_value = np.linspace(0, 2000, 10)

        TestCoalescence.coalescence.radius_for_extrapolation = None
        TestCoalescence.coalescence._set_default_radius_for_extrapolation()
        self.assertEqual(expected_default_radius, TestCoalescence.coalescence.radius_for_extrapolation)

        # case where there is not a radius with dx < 1 / (2 * orbital frequency at merger)
        # should choose the largest radius on the first refinement level with extraction spheres
        grid_structure = {
            1:{
                'center': [0, 0, 0],
                'levels':{
                    0:{
                        'dx': 48,
                        'radius': 420
                    },
                    1:{
                        'dx': 24,
                        'radius': 210
                    },
                    2:{
                        'dx': 12,
                        'radius': 105
                    }
                }
            }
        }
        
        orbital_frequency_at_merger = 0.5
        expected_default_radius = 100
        
        mock_grid_structure.return_value = grid_structure
        mock_orbital_frequency_at_time.return_value = orbital_frequency_at_merger
        mock_merge_time.return_value = 1500
        mock_get_time.return_value = np.linspace(0, 2000, 10)

        TestCoalescence.coalescence.radius_for_extrapolation = None
        TestCoalescence.coalescence._set_default_radius_for_extrapolation()
        self.assertEqual(expected_default_radius, TestCoalescence.coalescence.radius_for_extrapolation)

        # consider case where radius has to be reduced due to merge_time
        grid_structure = {
            1:{
                'center': [0, 0, 0],
                'levels':{
                    0:{
                        'dx': 3,
                        'radius': 400
                    },
                    1:{
                        'dx': 1.5,
                        'radius': 200
                    },
                    2:{
                        'dx': 0.75,
                        'radius': 100
                    },
                    3:{
                        'dx': 0.375,
                        'radius': 50
                    },
                    4:{
                        'dx': 0.1875,
                        'radius': 25
                    },
                    5:{
                        'dx': 0.09375,
                        'radius': 12.5
                    }
                }
            }
        }

        orbital_frequency_at_merger = 0.5
        expected_default_radius = 80
        
        mock_grid_structure.return_value = grid_structure
        mock_orbital_frequency_at_time.return_value = orbital_frequency_at_merger
        mock_merge_time.return_value = 1500 
        mock_get_time.return_value = np.linspace(0, 1735, 10)

        TestCoalescence.coalescence.radius_for_extrapolation = None
        TestCoalescence.coalescence._set_default_radius_for_extrapolation()
        self.assertEqual(expected_default_radius, TestCoalescence.coalescence.radius_for_extrapolation)
        
        # can't find a good radius should throw a value error
        grid_structure = {
            1:{
                'center': [0, 0, 0],
                'levels':{
                    0:{
                        'dx': 3,
                        'radius': 400
                    },
                    1:{
                        'dx': 1.5,
                        'radius': 200
                    },
                    2:{
                        'dx': 0.75,
                        'radius': 100
                    },
                    3:{
                        'dx': 0.375,
                        'radius': 50
                    },
                    4:{
                        'dx': 0.1875,
                        'radius': 25
                    },
                    5:{
                        'dx': 0.09375,
                        'radius': 12.5
                    }
                }
            }
        }

        orbital_frequency_at_merger = 0.5
        
        mock_grid_structure.return_value = grid_structure
        mock_orbital_frequency_at_time.return_value = orbital_frequency_at_merger
        mock_merge_time.return_value = 1500 # 1650 + 60
        mock_get_time.return_value = np.linspace(0, 1710, 10)

        TestCoalescence.coalescence.radius_for_extrapolation = None
        try:
            TestCoalescence.coalescence._set_default_radius_for_extrapolation()
            self.fail()
        except ValueError:
            pass

    def test_reset_radius_for_extrapolation_to_default(self):
        TestCoalescence.coalescence.radiationbundle._RadiationBundle__radius_for_extrapolation_to_default = 120
        with patch.object(Coalescence, '_set_default_radius_for_extrapolation') as mock_set_default_radius_for_extrapolation:
            TestCoalescence.coalescence.reset_radius_for_extrapolation_to_default()
            self.assertIsNone(TestCoalescence.coalescence.radiationbundle._RadiationBundle__radius_for_extrapolation)
            mock_set_default_radius_for_extrapolation.assert_called_once()
        
    def test_grid_structure(self):
        # equal mass simulation
        generated_grid_structure = TestCoalescence.coalescence.grid_structure
        expected_grid_structure = {
            1:{
                'center': [1.168642873, 0, 0],
                'levels':{
                    0:{
                        'dx': 6,
                        'radius': 384
                    },
                    1:{
                        'dx': 3,
                        'radius': 192
                    },
                    2:{
                        'dx':  1.5,
                        'radius': 96
                    },
                    3:{
                        'dx': 0.75,
                        'radius': 48
                    },
                    4:{
                        'dx': 0.375,
                        'radius': 12
                    },
                    5:{
                        'dx': 0.1875,
                        'radius': 6
                    },
                    6:{
                        'dx': 0.09375,
                        'radius': 3
                    },
                    7:{
                        'dx': 0.046875,
                        'radius': 1.5
                    },
                    8:{
                        'dx': 0.0234375,
                        'radius': 0.75
                    },
                }
            },
            2:{
                'center': [-1.168642873, 0, 0],
                'levels':{
                    0:{
                        'dx': 6,
                        'radius': 384
                    },
                    1:{
                        'dx': 3,
                        'radius': 192
                    },
                    2:{
                        'dx':  1.5,
                        'radius': 96
                    },
                    3:{
                        'dx': 0.75,
                        'radius': 48
                    },
                    4:{
                        'dx': 0.375,
                        'radius': 12
                    },
                    5:{
                        'dx': 0.1875,
                        'radius': 6
                    },
                    6:{
                        'dx': 0.09375,
                        'radius': 3
                    },
                    7:{
                        'dx': 0.046875,
                        'radius': 1.5
                    },
                    8:{
                        'dx': 0.0234375,
                        'radius': 0.75
                    },                
                }
            },
            3:{
                'center': [0, 0, 0],
                'levels':{
                    0:{
                        'dx': 6,
                        'radius': 384
                    },
                    1:{
                        'dx': 3,
                        'radius': 192
                    },
                    2:{
                        'dx':  1.5,
                        'radius': 96
                    },
                    3:{
                        'dx': 0.75,
                        'radius': 48
                    }
                }
            }
        }
        self.assertEqual(expected_grid_structure, generated_grid_structure)

        # unequal mass
        coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                               "resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5"))
        generated_grid_structure = coalescence.grid_structure
        expected_grid_structure = {
            1:{
                'center': [3.666667, 0, 0],
                'levels':{
                    0:{
                        'dx': 3.6266666666666669272,
                        'radius': 696.32
                    },
                    1:{
                        'dx': 1.8133333333333334636,
                        'radius': 348.16
                    },
                    2:{
                        'dx': 0.9066666666666667318,
                        'radius': 174.08
                    },
                    3:{
                        'dx': 0.4533333333333333659,
                        'radius': 43.52
                    },
                    4:{
                        'dx': 0.22666666666666668295,
                        'radius': 21.76
                    },
                    5:{
                        'dx': 0.113333333333333341475,
                        'radius': 5.44
                    },
                    6:{
                        'dx': 0.0566666666666666707375,
                        'radius': 2.72
                    },
                    7:{
                        'dx': 0.02833333333333333536875,
                        'radius': 1.36
                    },
                    8:{
                        'dx': 0.014166666666666667684375,
                        'radius': 0.68
                    },
                    9:{
                        'dx': 0.0070833333333333338421875,
                        'radius': 0.34
                    },
                }
            },
            2:{
                'center': [-7.333333, 0, 0],
                'levels':{
                    0:{
                        'dx': 3.6266666666666669272,
                        'radius': 696.32
                    },
                    1:{
                        'dx': 1.8133333333333334636,
                        'radius': 348.16
                    },
                    2:{
                        'dx': 0.9066666666666667318,
                        'radius': 174.08
                    },
                    3:{
                        'dx': 0.4533333333333333659,
                        'radius': 43.52
                    },
                    4:{
                        'dx': 0.22666666666666668295,
                        'radius': 21.76
                    },
                    5:{
                        'dx': 0.113333333333333341475,
                        'radius': 5.44
                    },
                    6:{
                        'dx': 0.0566666666666666707375,
                        'radius': 2.72
                    },
                    7:{
                        'dx': 0.02833333333333333536875,
                        'radius': 1.36
                    },
                    8:{
                        'dx': 0.014166666666666667684375,
                        'radius': 0.68
                    },
                    9:{
                        'dx': 0.0070833333333333338421875,
                        'radius': 0.34
                    },
                    10:{
                        'dx': 0.00354166666666666692109375,
                        'radius': 0.17
                    },
                }
            },
            3:{
                'center': [0, 0, 0],
                'levels':{
                    0:{
                        'dx': 3.6266666666666669272,
                        'radius': 696.32
                    },
                    1:{
                        'dx': 1.8133333333333334636,
                        'radius': 348.16
                    },
                    2:{
                        'dx': 0.9066666666666667318,
                        'radius': 174.08
                    },
                    3:{
                        'dx': 0.4533333333333333659,
                        'radius': 43.52
                    },
                }
            }
        }
        coalescence.close()
        self.assertEqual(expected_grid_structure, generated_grid_structure)
        
        # GW150914 from etk
        coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                               "resources/sample_etk_simulations/GW150914.h5"))
        generated_grid_structure = coalescence.grid_structure
        coalescence.close()
        self.assertIsNone(generated_grid_structure)
             

    def test_recoil_velocity(self):
        # equal mass should be close to zero
        TestCoalescence.coalescence.radius_for_extrapolation = 70
        kick_velocity = TestCoalescence.coalescence.recoil_velocity()
        self.assertTrue(np.allclose([0, 0, 0], kick_velocity, atol=1e-7))

        # check for a nonequal mass simulation
        coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                               "resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5"))

        # check if it is consistent with expected values from fits
        expected = 0.0002654485085365188
        generated_kick_vector = coalescence.recoil_velocity()
        generated_magnitude = np.linalg.norm(generated_kick_vector)
        self.assertTrue(np.isclose(expected, generated_magnitude, atol=5e-5))

        # check conversion of units
        geometric_units = coalescence.recoil_speed()
        SI_units = coalescence.recoil_speed(km_per_sec=True)
        self.assertTrue(np.isclose(SI_units, geometric_units * 299792.458))

    def test_recoil_speed(self):
        # equal mass so magnitude should be 0
        TestCoalescence.coalescence.radius_for_extrapolation = 70
        kick_vector = TestCoalescence.coalescence.recoil_velocity()
        expected_magnitude = np.linalg.norm(kick_vector)
        generated_magnitude = TestCoalescence.coalescence.recoil_speed()
        self.assertTrue(np.isclose(expected_magnitude, generated_magnitude, atol=1e-12))

        # check conversion of units
        geometric_units = TestCoalescence.coalescence.recoil_speed()
        SI_units = TestCoalescence.coalescence.recoil_speed(km_per_sec=True)
        self.assertTrue(np.isclose(SI_units, geometric_units * 299792.458))

        # expected value
        coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                               "resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5"))
        kick_vector = coalescence.recoil_velocity()
        expected_magnitude = np.linalg.norm(kick_vector)
        generated_magnitude = coalescence.recoil_speed()
        self.assertTrue(np.isclose(expected_magnitude, generated_magnitude, atol=1e-12))

        # check conversion of units
        geometric_units = coalescence.recoil_speed()
        SI_units = coalescence.recoil_speed(km_per_sec=True)
        self.assertTrue(np.isclose(SI_units, geometric_units * 299792.458))

    def test_extrapolate_psi4_to_infinite_radius(self):
        time, psi4_real, psi4_imag = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/psi4_strain_test/Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc"),
            usecols=(0, 1, 2), unpack=True)

        temp_h5_filename = os.path.join(TestCoalescence.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        psi4_group = temp_h5_file.create_group('psi4')
        psi4_group.create_dataset('real', data=psi4_real)
        psi4_group.create_dataset('imaginary', data=psi4_imag)

        radiation_mode = RadiationMode(l=2, m=2, rad=75, time=time, psi4_group=psi4_group)
        radiation_spheres = {75: RadiationSphere(mode_dict={(2, 2): radiation_mode}, time=time, radius=75)}
        radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        self.coalescence._Coalescence__radiation_mode_bundle = radiation_bundle

        # make sure the extrapolated waveform is not identical to the original waveform
        time_r75, psi4_real_r75, psi4_imag_r75 = TestCoalescence.coalescence.psi4_real_imag_for_mode(2, 2, 75)

        TestCoalescence.coalescence.extrapolate_psi4_to_infinite_radius(order=2, extraction_radius=75)

        time_extrap, psi4_real_extrap, psi4_imag_extrap = TestCoalescence.coalescence.psi4_real_imag_for_mode(2, 2)

        self.assertFalse(np.all(75 * psi4_real_r75 == psi4_real_extrap))
        self.assertFalse(np.all(75 * psi4_imag_r75 == psi4_imag_extrap))

        os.remove(temp_h5_filename)

        self.tearDown()

        # compare to Healy Toolkit version
        self.setUp()

        extrapolated_strain_filepath = os.path.join(TestCoalescence.CURR_DIR,
                                                    "resources/psi4_strain_test/Strain_l2_m2_rinf.txt")

        expected_time, expected_strain_plus, expected_strain_cross = np.loadtxt(extrapolated_strain_filepath,
                                                                                usecols=(0, 1, 2), unpack=True)

        expected_psi4_real = np.gradient(np.gradient(expected_strain_plus) / np.gradient(expected_time)) / np.gradient(
            expected_time) / 75
        expected_psi4_imag = -1 * np.gradient(
            np.gradient(expected_strain_cross) / np.gradient(expected_time)) / np.gradient(
            expected_time) /75

        self.assertTrue(np.all(np.isclose(expected_psi4_real, psi4_real_extrap, atol=2e-3)))
        self.assertTrue(np.all(np.isclose(expected_psi4_imag, psi4_imag_extrap, atol=2e-3)))

    def test_psi4_real_imag_for_mode(self):
        stitched_data_directory = os.path.join(TestCoalescence.CURR_DIR,
                                               "resources/main_test_simulation/stitched/%s") % TestCoalescence.simulation_name
        for mode in TestCoalescence.coalescence.included_modes:
            l_value = mode[0]
            m_value = mode[1]
            for extraction_radius in TestCoalescence.coalescence.included_extraction_radii:
                expected_psi4_data = np.loadtxt(os.path.join(stitched_data_directory,
                                                             "Ylm_WEYLSCAL4::Psi4r_l%s_m%s_r%.2f.asc" % (
                                                                 l_value, m_value, extraction_radius)))
                time, real_psi4, imag_psi4 = TestCoalescence.coalescence.psi4_real_imag_for_mode(l_value, m_value,
                                                                                                 extraction_radius)
                generated_psi4_data = np.empty((time.shape[0], 3))
                generated_psi4_data[:, 0] = time
                generated_psi4_data[:, 1] = real_psi4
                generated_psi4_data[:, 2] = imag_psi4
                self.assertTrue(np.all(np.isclose(generated_psi4_data, expected_psi4_data, atol=1e-6)))

        with patch.object(RadiationBundle, 'get_psi4_real_for_mode') as mock_psi4_real:
            with patch.object(RadiationBundle, 'get_psi4_imaginary_for_mode') as mock_psi4_imaginary:
                with patch.object(RadiationBundle, 'get_time') as mock_time:
                    time, real_psi4, imaginary_psi4 = TestCoalescence.coalescence.psi4_real_imag_for_mode(2, 1, 75)
                    mock_psi4_real.assert_called_once_with(2, 1, extraction_radius=75)
                    mock_psi4_imaginary.assert_called_once_with(2, 1, extraction_radius=75)
                    mock_time.assert_called_once_with(75)


        with patch.object(RadiationBundle, 'get_psi4_real_for_mode') as mock_psi4_real:
            with patch.object(RadiationBundle, 'get_psi4_imaginary_for_mode') as mock_psi4_imaginary:
                with patch.object(RadiationBundle, 'get_time') as mock_time:
                    time, real_psi4, imaginary_psi4 = TestCoalescence.coalescence.psi4_real_imag_for_mode(2, 1)
                    mock_psi4_real.assert_called_once_with(2, 1, extraction_radius=None)
                    mock_psi4_imaginary.assert_called_once_with(2, 1, extraction_radius=None)
                    mock_time.assert_called_once_with(None)

    def test_psi4_amp_phase_for_mode(self):
        stitched_data_directory = os.path.join(TestCoalescence.CURR_DIR,
                                               "resources/main_test_simulation/stitched/%s") % TestCoalescence.simulation_name
        for mode in TestCoalescence.coalescence.included_modes:
            l_value = mode[0]
            m_value = mode[1]
            for extraction_radius in TestCoalescence.coalescence.included_extraction_radii:
                expected_psi4_data_real_imag = np.loadtxt(os.path.join(stitched_data_directory,
                                                                       "Ylm_WEYLSCAL4::Psi4r_l%s_m%s_r%.2f.asc" % (
                                                                           l_value, m_value, extraction_radius)))
                expected_psi4_data = np.empty((expected_psi4_data_real_imag.shape[0], 3))
                expected_psi4_data[:, 0] = expected_psi4_data_real_imag[:, 0]
                expected_psi4_data[:, 1] = np.linalg.norm(expected_psi4_data_real_imag[:, 1:], axis=1)
                expected_psi4_data[:, 2] = -1 * np.unwrap(
                    np.arctan2(expected_psi4_data_real_imag[:, 2], expected_psi4_data_real_imag[:, 1]))

                time, amp_psi4, phase_psi4 = TestCoalescence.coalescence.psi4_amp_phase_for_mode(l_value, m_value,
                                                                                                 extraction_radius)
                generated_psi4_data = np.empty((time.shape[0], 3))
                generated_psi4_data[:, 0] = time
                generated_psi4_data[:, 1] = amp_psi4
                generated_psi4_data[:, 2] = phase_psi4
                self.assertTrue(np.all(np.isclose(generated_psi4_data, expected_psi4_data, atol=1e-6)))

        with patch.object(RadiationBundle, 'get_psi4_amplitude_for_mode') as mock_psi4_amplitude:
            with patch.object(RadiationBundle, 'get_psi4_phase_for_mode') as mock_psi4_phase:
                with patch.object(RadiationBundle, 'get_time') as mock_time:
                    time, amp_psi4, phase_psi4 = TestCoalescence.coalescence.psi4_amp_phase_for_mode(2, 1, 75)
                    mock_psi4_amplitude.assert_called_once_with(2, 1, extraction_radius=75)
                    mock_psi4_phase.assert_called_once_with(2, 1, extraction_radius=75)
                    mock_time.assert_called_once_with(75)


        with patch.object(RadiationBundle, 'get_psi4_amplitude_for_mode') as mock_psi4_amplitude:
            with patch.object(RadiationBundle, 'get_psi4_phase_for_mode') as mock_psi4_phase:
                with patch.object(RadiationBundle, 'get_time') as mock_time:
                    time, amp_psi4, phase_psi4 = TestCoalescence.coalescence.psi4_amp_phase_for_mode(2, 1)
                    mock_psi4_amplitude.assert_called_once_with(2, 1, extraction_radius=None)
                    mock_psi4_phase.assert_called_once_with(2, 1, extraction_radius=None)
                    mock_time.assert_called_once_with(None)

    def test_psi4_max_time_for_mode(self):
        max_time_r70 = TestCoalescence.coalescence.psi4_max_time_for_mode(2, 2, 70)
        self.assertEqual(max_time_r70, 100.5)

        TestCoalescence.coalescence.radius_for_extrapolation = 70
        max_time_extrap = TestCoalescence.coalescence.psi4_max_time_for_mode(2, 2)
        self.assertEqual(max_time_extrap, 100.5)

    def test_strain_for_mode(self):
        time, psi4_real, psi4_imag = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/psi4_strain_test/Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc"),
            usecols=(0, 1, 2), unpack=True)

        temp_h5_filename = os.path.join(TestCoalescence.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        psi4_group = temp_h5_file.create_group('psi4')
        psi4_group.create_dataset('real', data=psi4_real)
        psi4_group.create_dataset('imaginary', data=psi4_imag)

        radiation_mode = RadiationMode(l=2, m=2, rad=75, time=time, psi4_group=psi4_group)
        radiation_spheres = {75: RadiationSphere(mode_dict={(2, 2): radiation_mode}, time=time, radius=75)}
        radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        self.coalescence._Coalescence__radiation_mode_bundle = radiation_bundle

        time, strain_l2_m2_r75_plus, strain_l2_m2_r75_cross = TestCoalescence.coalescence.strain_for_mode(2, 2,
                                                                                                          75)

        strain_plus_derivative = np.gradient(strain_l2_m2_r75_plus, time)
        strain_cross_derivative = np.gradient(strain_l2_m2_r75_cross, time)

        strain_plus_second_derivative = np.gradient(strain_plus_derivative, time)
        strain_cross_second_derivative = np.gradient(strain_cross_derivative, time)

        time, psi4_l2_m2_r75_real, psi4_l2_m2_r75_imag = TestCoalescence.coalescence.psi4_real_imag_for_mode(2, 2, 75)

        self.assertTrue(np.all(np.isclose(strain_plus_second_derivative, psi4_l2_m2_r75_real, atol=0.1)))
        self.assertTrue(np.all(np.isclose(strain_cross_second_derivative, -1 * psi4_l2_m2_r75_imag, atol=0.1)))

        # compare to matlab version of r75
        time, strain_l2_m2_r75_plus, strain_l2_m2_r75_cross = TestCoalescence.coalescence.strain_for_mode(2, 2,
                                                                                                          75)

        strain_data_filepath = os.path.join(TestCoalescence.CURR_DIR, "resources/psi4_strain_test/Strain_l2_m2_r75.txt")

        expected_strain_time, expected_strain_plus, expected_strain_cross = np.loadtxt(strain_data_filepath,
                                                                                       usecols=(0, 1, 2), unpack=True)

        # crop the beginning since the matlab version windows the beginning
        cut_index = np.argmax(time > 200)
        strain_l2_m2_r75_plus = strain_l2_m2_r75_plus[cut_index:]
        strain_l2_m2_r75_cross = strain_l2_m2_r75_cross[cut_index:]
        expected_strain_plus = expected_strain_plus[cut_index:]
        expected_strain_cross = expected_strain_cross[cut_index:]

        self.assertTrue(np.all(np.isclose(expected_strain_time, time, atol=1e-4)))
        self.assertTrue(np.all(np.isclose(expected_strain_plus, strain_l2_m2_r75_plus, atol=2e-3)))
        self.assertTrue(np.all(np.isclose(expected_strain_cross, strain_l2_m2_r75_cross, atol=2e-3)))

        os.remove(temp_h5_filename)

        with patch.object(RadiationBundle, 'get_strain_plus_for_mode') as mock_strain_plus:
            with patch.object(RadiationBundle, 'get_strain_cross_for_mode') as mock_strain_cross:
                with patch.object(RadiationBundle, 'get_time') as mock_time:
                    time, strain_plus, strain_cross = TestCoalescence.coalescence.strain_for_mode(2, 1, 75)
                    mock_strain_plus.assert_called_once_with(2, 1, extraction_radius=75)
                    mock_strain_cross.assert_called_once_with(2, 1, extraction_radius=75)
                    mock_time.assert_called_once_with(75)

        with patch.object(RadiationBundle, 'get_strain_plus_for_mode') as mock_strain_plus:
            with patch.object(RadiationBundle, 'get_strain_cross_for_mode') as mock_strain_cross:
                with patch.object(RadiationBundle, 'get_time') as mock_time:
                    time, strain_plus, strain_cross = TestCoalescence.coalescence.strain_for_mode(2, 1)
                    mock_strain_plus.assert_called_once_with(2, 1, extraction_radius=None)
                    mock_strain_cross.assert_called_once_with(2, 1, extraction_radius=None)
                    mock_time.assert_called_once_with(None)

    def test_strain_recomposed_at_sky_location(self):
        # if extraction radius is provided
        extrapolated_strain = np.array([1, 1, 1, 1, 1]), np.array([2, 2, 2, 2, 2])
        time = np.array([0, 1, 2, 3, 4])
        with patch.object(RadiationBundle, 'get_strain_recomposed_at_sky_location',
                          return_value=extrapolated_strain) as mock_strain_recomposed:
            with patch.object(RadiationBundle, 'get_time', return_value=time) as mock_time:
                time_recovered, plus_recovered, cross_recovered = self.coalescence.strain_recomposed_at_sky_location(
                    theta=0.1 * np.pi, phi=0.4 * np.pi, extraction_radius=70)
                self.assertTrue(np.all(extrapolated_strain[0] == plus_recovered))
                self.assertTrue(np.all(extrapolated_strain[1] == cross_recovered))
                self.assertTrue(np.all(time == time_recovered))
                self.assertEqual(1, mock_time.call_count)
                mock_strain_recomposed.assert_called_once_with(theta=0.1 * np.pi, phi=0.4 * np.pi, extraction_radius=70)

        # if extraction radius not provided
        recomposed_strain = np.array([1, 1, 1, 1, 1]), np.array([2, 2, 2, 2, 2])
        time = np.array([0, 1, 2, 3, 4])
        with patch.object(RadiationBundle, 'get_strain_recomposed_at_sky_location',
                          return_value=recomposed_strain) as mock_strain_recomposed:
            with patch.object(RadiationBundle, 'get_time', return_value=time) as mock_time:
                time_recovered, plus_recovered, cross_recovered = self.coalescence.strain_recomposed_at_sky_location(
                    theta=0.1 * np.pi, phi=0.4 * np.pi)
                self.assertTrue(np.all(extrapolated_strain[0] == plus_recovered))
                self.assertTrue(np.all(extrapolated_strain[1] == cross_recovered))
                self.assertTrue(np.all(time == time_recovered))
                self.assertEqual(1, mock_time.call_count)
                mock_strain_recomposed.assert_called_once_with(theta=0.1 * np.pi, phi=0.4 * np.pi, extraction_radius=None)

    def test_strain_amp_phase_for_mode(self):
        time, psi4_real, psi4_imag = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, "resources/psi4_strain_test/Ylm_WEYLSCAL4::Psi4r_l2_m2_r75.00.asc"),
            usecols=(0, 1, 2), unpack=True)

        temp_h5_filename = os.path.join(TestCoalescence.CURR_DIR, "resources/temp/temp_mode.h5")
        temp_h5_file = h5py.File(temp_h5_filename, 'w')
        psi4_group = temp_h5_file.create_group('psi4')
        psi4_group.create_dataset('real', data=psi4_real)
        psi4_group.create_dataset('imaginary', data=psi4_imag)

        radiation_mode = RadiationMode(l=2, m=2, rad=75, time=time, psi4_group=psi4_group)
        radiation_spheres = {75: RadiationSphere(mode_dict={(2, 2): radiation_mode}, time=time, radius=75)}
        radiation_bundle = RadiationBundle(radiation_spheres=radiation_spheres)
        self.coalescence._Coalescence__radiation_mode_bundle = radiation_bundle

        # compare to matlab version of r75
        time, strain_l2_m2_r75_amp, strain_l2_m2_r75_phase = TestCoalescence.coalescence.strain_amp_phase_for_mode(2, 2,
                                                                                                                   75)
        time, strain_l2_m2_r75_plus, strain_l2_m2_r75_cross = TestCoalescence.coalescence.strain_for_mode(2, 2,
                                                                                                          75)

        expected_strain_amp = np.sqrt(
            strain_l2_m2_r75_plus * strain_l2_m2_r75_plus + strain_l2_m2_r75_cross * strain_l2_m2_r75_cross)
        expected_strain_phase = -1 * np.unwrap(np.arctan2(strain_l2_m2_r75_cross, strain_l2_m2_r75_plus))

        self.assertTrue(np.all(np.isclose(expected_strain_amp, strain_l2_m2_r75_amp, atol=1e-4)))
        self.assertTrue(np.all(np.isclose(expected_strain_phase, strain_l2_m2_r75_phase, atol=1e-4)))

        os.remove(temp_h5_filename)

        with patch.object(RadiationBundle, 'get_strain_amplitude_for_mode') as mock_strain_amp:
            with patch.object(RadiationBundle, 'get_strain_phase_for_mode') as mock_strain_phase:
                with patch.object(RadiationBundle, 'get_time') as mock_time:
                    time, strain_amp, strain_phase = TestCoalescence.coalescence.strain_amp_phase_for_mode(2, 1, 75)
                    mock_strain_amp.assert_called_once_with(2, 1, extraction_radius=75)
                    mock_strain_phase.assert_called_once_with(2, 1, extraction_radius=75)
                    mock_time.assert_called_once_with(75)

        with patch.object(RadiationBundle, 'get_strain_amplitude_for_mode') as mock_strain_amp:
            with patch.object(RadiationBundle, 'get_strain_phase_for_mode') as mock_strain_phase:
                with patch.object(RadiationBundle, 'get_time') as mock_time:
                    time, strain_amp, strain_phase = TestCoalescence.coalescence.strain_amp_phase_for_mode(2, 1)
                    mock_strain_amp.assert_called_once_with(2, 1, extraction_radius=None)
                    mock_strain_phase.assert_called_once_with(2, 1, extraction_radius=None)
                    mock_time.assert_called_once_with(None)

    def test_dEnergy_dt_radiated(self):
        coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        # compare with psi4analysis
        generated_time, generated_denergy_dt_radiated = coalescence.dEnergy_dt_radiated(extraction_radius=75)

        psi4_analysis_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, 'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
            usecols=5)

        tolerance = 1e-2 * max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_denergy_dt_radiated, atol=tolerance))

        # check kwargs get passed along
        return_placeholder = None, None

        # nothing
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated()
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call, mock_get_dEnergy_dt_radiated.call_args)

        # lmin only
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated(lmin=2)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(lmin=2), mock_get_dEnergy_dt_radiated.call_args)

        # lmax only
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated(lmax=4)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(lmax=4), mock_get_dEnergy_dt_radiated.call_args)

        # lmin and lmax
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated(lmin=3, lmax=7)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(lmin=3, lmax=7), mock_get_dEnergy_dt_radiated.call_args)

        # l and m
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated(l=4, m=1)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4, m=1), mock_get_dEnergy_dt_radiated.call_args)

        # l only
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated(l=4)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4), mock_get_dEnergy_dt_radiated.call_args)

        # m only
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated(m=1)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(m=1), mock_get_dEnergy_dt_radiated.call_args)

        # l and lmin
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated(l=4, lmin=3)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4, lmin=3), mock_get_dEnergy_dt_radiated.call_args)

        # extraction radius
        with patch.object(RadiationBundle, 'get_dEnergy_dt_radiated',
                          return_value=return_placeholder) as mock_get_dEnergy_dt_radiated:
            coalescence.dEnergy_dt_radiated(extraction_radius=70)
            self.assertEqual(1, mock_get_dEnergy_dt_radiated.call_count)
            self.assertEqual(mock.call(extraction_radius=70), mock_get_dEnergy_dt_radiated.call_args)

    def test_energy_radiated(self):
        coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        # compare with psi4analysis
        generated_time, generated_energy_radiated = coalescence.energy_radiated(extraction_radius=75)

        psi4_analysis_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, 'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
            usecols=1)

        tolerance = 1e-2 * max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_energy_radiated, atol=tolerance))

        # check kwargs get passed along
        return_placeholder = None, None

        # nothing
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated()
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call, mock_get_energy_radiated.call_args)

        # lmin only
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated(lmin=2)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(lmin=2), mock_get_energy_radiated.call_args)

        # lmax only
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated(lmax=4)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(lmax=4), mock_get_energy_radiated.call_args)

        # lmin and lmax
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated(lmin=3, lmax=7)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(lmin=3, lmax=7), mock_get_energy_radiated.call_args)

        # l and m
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated(l=4, m=1)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(l=4, m=1), mock_get_energy_radiated.call_args)

        # l only
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated(l=4)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(l=4), mock_get_energy_radiated.call_args)

        # m only
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated(m=1)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(m=1), mock_get_energy_radiated.call_args)

        # l and lmin
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated(l=4, lmin=3)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(l=4, lmin=3), mock_get_energy_radiated.call_args)

        # extraction radius
        with patch.object(RadiationBundle, 'get_energy_radiated',
                          return_value=return_placeholder) as mock_get_energy_radiated:
            coalescence.energy_radiated(extraction_radius=70)
            self.assertEqual(1, mock_get_energy_radiated.call_count)
            self.assertEqual(mock.call(extraction_radius=70), mock_get_energy_radiated.call_args)

    def test_dP_dt_radiated(self):
        coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        # compare with psi4analysis
        generated_time, generated_dP_dt = coalescence.dP_dt_radiated(extraction_radius=75)

        psi4_analysis_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, 'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
            usecols=(6, 7, 8))

        tolerance = 5e-2 * np.max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_dP_dt, atol=tolerance))

        # check kwargs get passed along
        return_placeholder = None, None

        # nothing
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated()
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call, mock_get_dP_dt_radiated.call_args)

        # lmin only
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated(lmin=2)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(lmin=2), mock_get_dP_dt_radiated.call_args)

        # lmax only
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated(lmax=4)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(lmax=4), mock_get_dP_dt_radiated.call_args)

        # lmin and lmax
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated(lmin=3, lmax=7)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(lmin=3, lmax=7), mock_get_dP_dt_radiated.call_args)

        # l and m
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated(l=4, m=1)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4, m=1), mock_get_dP_dt_radiated.call_args)

        # l only
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated(l=4)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4), mock_get_dP_dt_radiated.call_args)

        # m only
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated(m=1)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(m=1), mock_get_dP_dt_radiated.call_args)

        # l and lmin
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated(l=4, lmin=3)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(l=4, lmin=3), mock_get_dP_dt_radiated.call_args)

        # extraction radius
        with patch.object(RadiationBundle, 'get_dP_dt_radiated',
                          return_value=return_placeholder) as mock_get_dP_dt_radiated:
            coalescence.dP_dt_radiated(extraction_radius=70)
            self.assertEqual(1, mock_get_dP_dt_radiated.call_count)
            self.assertEqual(mock.call(extraction_radius=70), mock_get_dP_dt_radiated.call_args)

    def test_linear_momentum_radiated(self):
        coalescence = Coalescence(os.path.join(TestCoalescence.CURR_DIR,
                                               'resources/radiative_quantities_resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35.h5'))

        # compare with psi4analysis
        generated_time, generated_linear_momentum = coalescence.linear_momentum_radiated(extraction_radius=75)

        psi4_analysis_data = np.loadtxt(
            os.path.join(TestCoalescence.CURR_DIR, 'resources/radiative_quantities_resources/psi4analysis_r75.00.asc'),
            usecols=(2, 3, 4))

        tolerance = 5e-2 * np.max(psi4_analysis_data)
        self.assertTrue(np.allclose(psi4_analysis_data, generated_linear_momentum, atol=tolerance))

        # check kwargs get passed along
        return_placeholder = None, None

        # nothing
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated()
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call, mock_get_linear_momentum_radiated.call_args)

        # lmin only
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated(lmin=2)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(lmin=2), mock_get_linear_momentum_radiated.call_args)

        # lmax only
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated(lmax=4)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(lmax=4), mock_get_linear_momentum_radiated.call_args)

        # lmin and lmax
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated(lmin=3, lmax=7)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(lmin=3, lmax=7), mock_get_linear_momentum_radiated.call_args)

        # l and m
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated(l=4, m=1)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(l=4, m=1), mock_get_linear_momentum_radiated.call_args)

        # l only
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated(l=4)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(l=4), mock_get_linear_momentum_radiated.call_args)

        # m only
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated(m=1)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(m=1), mock_get_linear_momentum_radiated.call_args)

        # l and lmin
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated(l=4, lmin=3)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(l=4, lmin=3), mock_get_linear_momentum_radiated.call_args)

        # extraction radius
        with patch.object(RadiationBundle, 'get_linear_momentum_radiated',
                          return_value=return_placeholder) as mock_get_linear_momentum_radiated:
            coalescence.linear_momentum_radiated(extraction_radius=70)
            self.assertEqual(1, mock_get_linear_momentum_radiated.call_count)
            self.assertEqual(mock.call(extraction_radius=70), mock_get_linear_momentum_radiated.call_args)

    def test_object_numbers(self):
        generated_object_numbers = TestCoalescence.coalescence.object_numbers
        generated_object_numbers.sort()
        expected_object_numbers = [0, 1, 2]
        self.assertEqual(generated_object_numbers, expected_object_numbers)

    def test_compact_object_data_for_object(self):
        expected_compact_object_data_0 = \
            TestCoalescence.coalescence._Coalescence__h5_file["compact_object"]["object=0"][()]
        expected_compact_object_data_1 = \
            TestCoalescence.coalescence._Coalescence__h5_file["compact_object"]["object=1"][()]
        expected_compact_object_data_2 = \
            TestCoalescence.coalescence._Coalescence__h5_file["compact_object"]["object=2"][()]

        actual_compact_object_data_0 = TestCoalescence.coalescence.compact_object_data_for_object(0)[()]
        actual_compact_object_data_1 = TestCoalescence.coalescence.compact_object_data_for_object(1)[()]
        actual_compact_object_data_2 = TestCoalescence.coalescence.compact_object_data_for_object(2)[()]

        self.assertTrue(np.all(np.isnan(expected_compact_object_data_0) == np.isnan(actual_compact_object_data_0)))
        self.assertTrue(np.all(expected_compact_object_data_0[np.where(~np.isnan(expected_compact_object_data_0))] ==
                               actual_compact_object_data_0[np.where(~np.isnan(actual_compact_object_data_0))]))

        self.assertTrue(np.all(np.isnan(expected_compact_object_data_1) == np.isnan(actual_compact_object_data_1)))
        self.assertTrue(np.all(expected_compact_object_data_1[np.where(~np.isnan(expected_compact_object_data_1))] ==
                               actual_compact_object_data_1[np.where(~np.isnan(actual_compact_object_data_1))]))

        self.assertTrue(np.all(np.isnan(expected_compact_object_data_2) == np.isnan(actual_compact_object_data_2)))
        self.assertTrue(np.all(expected_compact_object_data_2[np.where(~np.isnan(expected_compact_object_data_2))] ==
                               actual_compact_object_data_2[np.where(~np.isnan(actual_compact_object_data_2))]))

    def test_compact_object_metadata_dict(self):
        metadata_dict = TestCoalescence.coalescence.compact_object_metadata_dict()
        expected_metadata_dict = {}
        self.assertEqual(expected_metadata_dict, metadata_dict)

    def test_runstats_data(self):
        stitched_data_directory = os.path.join(TestCoalescence.CURR_DIR,
                                               "resources/main_test_simulation/stitched/%s") % TestCoalescence.simulation_name
        expected_runstats_data = np.loadtxt(os.path.join(stitched_data_directory, "runstats.asc"))
        generated_runstats_data = TestCoalescence.coalescence.runstats_data
        self.assertTrue(np.all(expected_runstats_data == generated_runstats_data))
        self.assertTrue(np.all(expected_runstats_data[:, 0] == generated_runstats_data['iteration']))
        self.assertTrue(np.all(expected_runstats_data[:, 1] == generated_runstats_data['coord_time']))
        self.assertTrue(np.all(expected_runstats_data[:, 2] == generated_runstats_data['wall_time']))
        self.assertTrue(np.all(expected_runstats_data[:, 3] == generated_runstats_data['speed (hours^-1)']))
        self.assertTrue(np.all(expected_runstats_data[:, 4] == generated_runstats_data['period (minutes)']))
        self.assertTrue(np.all(expected_runstats_data[:, 5] == generated_runstats_data['cputime (cpu hours)']))

    def test_psi4_source(self):
        psi4_source = TestCoalescence.coalescence.psi4_source
        self.assertEqual(psi4_source, "YLM_WEYLSCAL4_ASC")
