from unittest import TestCase
import numpy as np
import os
from shutil import copy2
from mayawaves.coalescence import Coalescence
from mayawaves.compactobject import CompactObject
from unittest import mock


class TestCompactObject(TestCase):
    primary_compact_object = None
    secondary_compact_object = None
    final_compact_object = None

    simulation_name = None
    coalescence = None
    h5_file = None

    CURR_DIR = os.path.dirname(__file__)

    def setUp(self) -> None:
        TestCompactObject.simulation_name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        TestCompactObject.h5_file = os.path.join(TestCompactObject.CURR_DIR,
                                                 "resources/temp/%s.h5") % TestCompactObject.simulation_name

        h5_file_original = os.path.join(TestCompactObject.CURR_DIR,
                                        "resources/%s.h5") % TestCompactObject.simulation_name
        os.mkdir(os.path.join(TestCompactObject.CURR_DIR, "resources/temp"))
        copy2(h5_file_original, TestCompactObject.h5_file)

        TestCompactObject.coalescence = Coalescence(TestCompactObject.h5_file)
        TestCompactObject.primary_compact_object = TestCompactObject.coalescence.primary_compact_object
        TestCompactObject.secondary_compact_object = TestCompactObject.coalescence.secondary_compact_object
        TestCompactObject.final_compact_object = TestCompactObject.coalescence.final_compact_object

    def tearDown(self) -> None:
        os.remove(TestCompactObject.h5_file)
        os.rmdir(os.path.join(TestCompactObject.CURR_DIR, "resources/temp"))
        TestCompactObject.coalescence.close()

    def test_get_data_from_columns(self):
        # one column
        for column in CompactObject.Column:
            column_header_list = TestCompactObject.primary_compact_object._CompactObject__header_list
            if column.header_text in column_header_list:
                column_idx = column_header_list.index(column.header_text)
                generated_data = TestCompactObject.primary_compact_object.get_data_from_columns([column])
                nan_columns = np.isnan(TestCompactObject.primary_compact_object._CompactObject__data_array[()][:, column_idx])
                expected_data = TestCompactObject.primary_compact_object._CompactObject__data_array[()][~nan_columns][:, column_idx]
                self.assertTrue(np.array_equal(expected_data, generated_data))

        # multiple columns
        columns = [CompactObject.Column.TIME, CompactObject.Column.AX, CompactObject.Column.AREA]
        column_header_list = TestCompactObject.primary_compact_object._CompactObject__header_list
        column_idxs = [column_header_list.index(col.header_text) for col in columns]
        generated_data = TestCompactObject.primary_compact_object.get_data_from_columns(columns)
        nan_columns = np.any(
            np.isnan(TestCompactObject.primary_compact_object._CompactObject__data_array[()][:, column_idxs]), axis=1)
        expected_data = TestCompactObject.primary_compact_object._CompactObject__data_array[()][~nan_columns][:,
                        column_idxs]
        self.assertTrue(np.array_equal(expected_data, generated_data))

        # requested columns don't exist in data array
        test_compactobject = CompactObject(np.array([[np.nan, 2, 3, 4], [5, 6, np.nan, 8], [8, 7, 6, 5], [4, 3, 2, 1]]),
                                           [CompactObject.Column.TIME, CompactObject.Column.AX, CompactObject.Column.AY, CompactObject.Column.AZ],
                                           0,
                                           "BH",
                                           0,
                                           0,
                                           np.array([0, 0, 0]),
                                           np.array([0, 0, 0]))
        columns = [CompactObject.Column.TIME, CompactObject.Column.AX, CompactObject.Column.AREA]
        generated_data = test_compactobject.get_data_from_columns(columns)
        self.assertIsNone(generated_data)

    def test_available_data_columns(self):
        expected_columns = [CompactObject.Column.ITT, CompactObject.Column.TIME, CompactObject.Column.X,
                            CompactObject.Column.Y, CompactObject.Column.Z, CompactObject.Column.VX,
                            CompactObject.Column.VY, CompactObject.Column.VZ, CompactObject.Column.AX,
                            CompactObject.Column.AY, CompactObject.Column.AZ, CompactObject.Column.SX,
                            CompactObject.Column.SY, CompactObject.Column.SZ, CompactObject.Column.PX,
                            CompactObject.Column.PY, CompactObject.Column.PZ, CompactObject.Column.MIN_RADIUS,
                            CompactObject.Column.MAX_RADIUS, CompactObject.Column.MEAN_RADIUS,
                            CompactObject.Column.QUADRUPOLE_XX, CompactObject.Column.QUADRUPOLE_XY,
                            CompactObject.Column.QUADRUPOLE_XZ, CompactObject.Column.QUADRUPOLE_YY,
                            CompactObject.Column.QUADRUPOLE_YZ, CompactObject.Column.QUADRUPOLE_ZZ,
                            CompactObject.Column.MIN_X, CompactObject.Column.MAX_X, CompactObject.Column.MIN_Y,
                            CompactObject.Column.MAX_Y, CompactObject.Column.MIN_Z, CompactObject.Column.MAX_Z,
                            CompactObject.Column.XY_PLANE_CIRCUMFERENCE, CompactObject.Column.XZ_PLANE_CIRCUMFERENCE,
                            CompactObject.Column.YZ_PLANE_CIRCUMFERENCE,
                            CompactObject.Column.RATIO_OF_XZ_XY_PLANE_CIRCUMFERENCES,
                            CompactObject.Column.RATIO_OF_YZ_XY_PLANE_CIRCUMFERENCES,
                            CompactObject.Column.AREA, CompactObject.Column.M_IRREDUCIBLE,
                            CompactObject.Column.AREAL_RADIUS, CompactObject.Column.EXPANSION_THETA_L,
                            CompactObject.Column.INNER_EXPANSION_THETA_N,
                            CompactObject.Column.PRODUCT_OF_THE_EXPANSIONS, CompactObject.Column.MEAN_CURVATURE,
                            CompactObject.Column.GRADIENT_OF_THE_AREAL_RADIUS,
                            CompactObject.Column.GRADIENT_OF_THE_EXPANSION_THETA_L,
                            CompactObject.Column.GRADIENT_OF_THE_INNER_EXPANSION_THETA_N,
                            CompactObject.Column.GRADIENT_OF_THE_PRODUCT_OF_THE_EXPANSIONS,
                            CompactObject.Column.GRADIENT_OF_THE_MEAN_CURVATURE,
                            CompactObject.Column.MINIMUM_OF_THE_MEAN_CURVATURE,
                            CompactObject.Column.MAXIMUM_OF_THE_MEAN_CURVATURE,
                            CompactObject.Column.INTEGRAL_OF_THE_MEAN_CURVATURE]
        actual_columns = TestCompactObject.primary_compact_object.available_data_columns
        self.assertEqual(set(expected_columns), set(actual_columns))

        test_coalescence = Coalescence(os.path.join(TestCompactObject.CURR_DIR, 'resources/sample_etk_simulations/GW150914.h5'))
        secondary_object = test_coalescence.secondary_compact_object
        expected_columns = [CompactObject.Column.ITT, CompactObject.Column.TIME, CompactObject.Column.X,
                            CompactObject.Column.Y, CompactObject.Column.Z, CompactObject.Column.SX,
                            CompactObject.Column.SY, CompactObject.Column.SZ, CompactObject.Column.MIN_RADIUS,
                            CompactObject.Column.MAX_RADIUS, CompactObject.Column.MEAN_RADIUS,
                            CompactObject.Column.QUADRUPOLE_XX, CompactObject.Column.QUADRUPOLE_XY,
                            CompactObject.Column.QUADRUPOLE_XZ, CompactObject.Column.QUADRUPOLE_YY,
                            CompactObject.Column.QUADRUPOLE_YZ, CompactObject.Column.QUADRUPOLE_ZZ,
                            CompactObject.Column.MIN_X, CompactObject.Column.MAX_X, CompactObject.Column.MIN_Y,
                            CompactObject.Column.MAX_Y, CompactObject.Column.MIN_Z, CompactObject.Column.MAX_Z,
                            CompactObject.Column.XY_PLANE_CIRCUMFERENCE, CompactObject.Column.XZ_PLANE_CIRCUMFERENCE,
                            CompactObject.Column.YZ_PLANE_CIRCUMFERENCE,
                            CompactObject.Column.RATIO_OF_XZ_XY_PLANE_CIRCUMFERENCES,
                            CompactObject.Column.RATIO_OF_YZ_XY_PLANE_CIRCUMFERENCES,
                            CompactObject.Column.AREA, CompactObject.Column.M_IRREDUCIBLE,
                            CompactObject.Column.AREAL_RADIUS, CompactObject.Column.EXPANSION_THETA_L,
                            CompactObject.Column.INNER_EXPANSION_THETA_N,
                            CompactObject.Column.PRODUCT_OF_THE_EXPANSIONS, CompactObject.Column.MEAN_CURVATURE,
                            CompactObject.Column.GRADIENT_OF_THE_AREAL_RADIUS,
                            CompactObject.Column.GRADIENT_OF_THE_EXPANSION_THETA_L,
                            CompactObject.Column.GRADIENT_OF_THE_INNER_EXPANSION_THETA_N,
                            CompactObject.Column.GRADIENT_OF_THE_PRODUCT_OF_THE_EXPANSIONS,
                            CompactObject.Column.GRADIENT_OF_THE_MEAN_CURVATURE,
                            CompactObject.Column.MINIMUM_OF_THE_MEAN_CURVATURE,
                            CompactObject.Column.MAXIMUM_OF_THE_MEAN_CURVATURE,
                            CompactObject.Column.INTEGRAL_OF_THE_MEAN_CURVATURE, CompactObject.Column.EQUATORIAL_CIRCUMFERENCE,
                            CompactObject.Column.POLAR_CIRCUMFERENCE_0, CompactObject.Column.POLAR_CIRCUMFERENCE_PI_2,
                            CompactObject.Column.SPIN_GUESS, CompactObject.Column.MASS_GUESS,
                            CompactObject.Column.KILLING_EIGENVALUE_REAL, CompactObject.Column.KILLING_EIGENVALUE_IMAG,
                            CompactObject.Column.SPIN_MAGNITUDE, CompactObject.Column.NPSPIN,
                            CompactObject.Column.WSSPIN, CompactObject.Column.SPIN_FROM_PHI_COORDINATE_VECTOR,
                            CompactObject.Column.HORIZON_MASS, CompactObject.Column.ADM_ENERGY,
                            CompactObject.Column.ADM_MOMENTUM_X, CompactObject.Column.ADM_MOMENTUM_Y,
                            CompactObject.Column.ADM_MOMENTUM_Z, CompactObject.Column.ADM_ANGULAR_MOMENTUM_X,
                            CompactObject.Column.ADM_ANGULAR_MOMENTUM_Y, CompactObject.Column.ADM_ANGULAR_MOMENTUM_Z,
                            CompactObject.Column.WEINBERG_ENERGY, CompactObject.Column.WEINBERG_MOMENTUM_X,
                            CompactObject.Column.WEINBERG_MOMENTUM_Y, CompactObject.Column.WEINBERG_MOMENTUM_Z,
                            CompactObject.Column.WEINBERG_ANGULAR_MOMENTUM_X,
                            CompactObject.Column.WEINBERG_ANGULAR_MOMENTUM_Y,
                            CompactObject.Column.WEINBERG_ANGULAR_MOMENTUM_Z]
        actual_columns = secondary_object.available_data_columns
        self.assertEqual(set(expected_columns), set(actual_columns))

    def test_initial_dimensionless_spin(self):
        self.assertTrue(np.all(
            np.isclose([0, 0, 0], TestCompactObject.primary_compact_object.initial_dimensionless_spin,
                       atol=1e-3)))
        self.assertTrue(np.all(
            np.isclose([0, 0, 0], TestCompactObject.secondary_compact_object.initial_dimensionless_spin,
                       atol=1e-3)))
        self.assertTrue(np.all(TestCompactObject.coalescence._Coalescence__h5_file.attrs[
                                   "dimensionless spin 0"] == TestCompactObject.primary_compact_object.initial_dimensionless_spin))
        self.assertTrue(np.all(TestCompactObject.coalescence._Coalescence__h5_file.attrs[
                                   "dimensionless spin 1"] == TestCompactObject.secondary_compact_object.initial_dimensionless_spin))

    def test_initial_dimensional_spin(self):
        self.assertTrue(np.all(
            np.isclose(TestCompactObject.primary_compact_object.initial_dimensional_spin, [0, 0, 0],
                       atol=1e-3)))
        self.assertTrue(np.all(
            np.isclose(TestCompactObject.secondary_compact_object.initial_dimensional_spin, [0, 0, 0], atol=1e-3)))

        self.assertTrue(np.all(TestCompactObject.coalescence._Coalescence__h5_file.attrs[
                                   "dimensional spin 0"] == TestCompactObject.primary_compact_object.initial_dimensional_spin))
        self.assertTrue(np.all(TestCompactObject.coalescence._Coalescence__h5_file.attrs[
                                   "dimensional spin 1"] == TestCompactObject.secondary_compact_object.initial_dimensional_spin))

    def test_initial_irreducible_mass(self):
        self.assertTrue(
            np.isclose(TestCompactObject.primary_compact_object.initial_irreducible_mass, 5.16797e-1, atol=1e-4))
        self.assertTrue(
            np.isclose(TestCompactObject.secondary_compact_object.initial_irreducible_mass, 5.16797e-1, atol=1e-4))

        self.assertTrue(np.all(TestCompactObject.coalescence._Coalescence__h5_file.attrs[
                                   "irreducible mass 0"] == TestCompactObject.primary_compact_object.initial_irreducible_mass))
        self.assertTrue(np.all(TestCompactObject.coalescence._Coalescence__h5_file.attrs[
                                   "irreducible mass 1"] == TestCompactObject.secondary_compact_object.initial_irreducible_mass))

    def test_initial_horizon_mass(self):
        self.assertTrue(
            np.isclose(TestCompactObject.primary_compact_object.initial_horizon_mass, 5.16797e-1, atol=1e-4))
        self.assertTrue(
            np.isclose(TestCompactObject.secondary_compact_object.initial_horizon_mass, 5.16797e-1, atol=1e-4))

        self.assertTrue(np.all(TestCompactObject.coalescence._Coalescence__h5_file.attrs[
                                   "horizon mass 0"] == TestCompactObject.primary_compact_object.initial_horizon_mass))
        self.assertTrue(np.all(TestCompactObject.coalescence._Coalescence__h5_file.attrs[
                                   "horizon mass 1"] == TestCompactObject.secondary_compact_object.initial_horizon_mass))

    def test_final_dimensionless_spin(self):
        expected_dimensionless_spin = TestCompactObject.primary_compact_object.dimensionless_spin_vector[1]
        expected_final_dimensionless_spin = expected_dimensionless_spin[-1]
        generated_final_dimensionless_spin = TestCompactObject.primary_compact_object.final_dimensionless_spin
        self.assertTrue(np.allclose(expected_final_dimensionless_spin, generated_final_dimensionless_spin, atol=1e-4))

        expected_dimensionless_spin = TestCompactObject.secondary_compact_object.dimensionless_spin_vector[1]
        expected_final_dimensionless_spin = expected_dimensionless_spin[-1]
        generated_final_dimensionless_spin = TestCompactObject.secondary_compact_object.final_dimensionless_spin
        self.assertTrue(np.allclose(expected_final_dimensionless_spin, generated_final_dimensionless_spin, atol=1e-4))

        expected_dimensionless_spin = TestCompactObject.final_compact_object.dimensionless_spin_vector[1]
        expected_final_dimensionless_spin = expected_dimensionless_spin[-1]
        generated_final_dimensionless_spin = TestCompactObject.final_compact_object.final_dimensionless_spin
        self.assertTrue(np.allclose(expected_final_dimensionless_spin, generated_final_dimensionless_spin, atol=1e-4))

    def test_final_dimensional_spin(self):
        expected_dimensional_spin = TestCompactObject.primary_compact_object.dimensional_spin_vector[1]
        expected_final_dimensional_spin = expected_dimensional_spin[-1]
        generated_final_dimensional_spin = TestCompactObject.primary_compact_object.final_dimensional_spin
        self.assertTrue(np.allclose(expected_final_dimensional_spin, generated_final_dimensional_spin, atol=1e-4))

        expected_dimensional_spin = TestCompactObject.secondary_compact_object.dimensional_spin_vector[1]
        expected_final_dimensional_spin = expected_dimensional_spin[-1]
        generated_final_dimensional_spin = TestCompactObject.secondary_compact_object.final_dimensional_spin
        self.assertTrue(np.allclose(expected_final_dimensional_spin, generated_final_dimensional_spin, atol=1e-4))

        expected_dimensional_spin = TestCompactObject.final_compact_object.dimensional_spin_vector[1]
        expected_final_dimensional_spin = expected_dimensional_spin[-1]
        generated_final_dimensional_spin = TestCompactObject.final_compact_object.final_dimensional_spin
        self.assertTrue(np.allclose(expected_final_dimensional_spin, generated_final_dimensional_spin, atol=1e-4))

    def test_final_irreducible_mass(self):
        expected_irreducible_mass = TestCompactObject.primary_compact_object.irreducible_mass[1]
        expected_final_irreducible_mass = expected_irreducible_mass[-1]
        generated_final_irreducible_mass = TestCompactObject.primary_compact_object.final_irreducible_mass
        self.assertTrue(np.allclose(expected_final_irreducible_mass, generated_final_irreducible_mass, atol=1e-4))

        expected_irreducible_mass = TestCompactObject.secondary_compact_object.irreducible_mass[1]
        expected_final_irreducible_mass = expected_irreducible_mass[-1]
        generated_final_irreducible_mass = TestCompactObject.secondary_compact_object.final_irreducible_mass
        self.assertTrue(np.allclose(expected_final_irreducible_mass, generated_final_irreducible_mass, atol=1e-4))

        expected_irreducible_mass = TestCompactObject.final_compact_object.irreducible_mass[1]
        expected_final_irreducible_mass = expected_irreducible_mass[-1]
        generated_final_irreducible_mass = TestCompactObject.final_compact_object.final_irreducible_mass
        self.assertTrue(np.allclose(expected_final_irreducible_mass, generated_final_irreducible_mass, atol=1e-4))

    def test_final_horizon_mass(self):
        expected_horizon_mass = TestCompactObject.primary_compact_object.horizon_mass[1]
        expected_final_horizon_mass = expected_horizon_mass[-1]
        generated_final_horizon_mass = TestCompactObject.primary_compact_object.final_horizon_mass
        self.assertTrue(np.allclose(expected_final_horizon_mass, generated_final_horizon_mass, atol=1e-4))

        expected_horizon_mass = TestCompactObject.secondary_compact_object.horizon_mass[1]
        expected_final_horizon_mass = expected_horizon_mass[-1]
        generated_final_horizon_mass = TestCompactObject.secondary_compact_object.final_horizon_mass
        self.assertTrue(np.allclose(expected_final_horizon_mass, generated_final_horizon_mass, atol=1e-4))

        expected_horizon_mass = TestCompactObject.final_compact_object.horizon_mass[1]
        expected_final_horizon_mass = expected_horizon_mass[-1]
        generated_final_horizon_mass = TestCompactObject.final_compact_object.final_horizon_mass
        self.assertTrue(np.allclose(expected_final_horizon_mass, generated_final_horizon_mass, atol=1e-4))

    def test_position_vector(self):
        shifttracker_time = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                    "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc")
                                       % TestCompactObject.simulation_name, usecols=1)

        generated_time, generated_position_vector_0 = TestCompactObject.primary_compact_object.position_vector
        shifttracker_position0 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc")
                                            % TestCompactObject.simulation_name, usecols=(2, 3, 4))

        generated_time, generated_position_vector_1 = TestCompactObject.secondary_compact_object.position_vector
        shifttracker_position1 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/%s/ShiftTracker1.asc")
                                            % TestCompactObject.simulation_name, usecols=(2, 3, 4))

        time, generated_indices, shifttracker_indices = np.intersect1d(generated_time, shifttracker_time,
                                                                       return_indices=True)

        generated_position_vector_0 = generated_position_vector_0[generated_indices]
        generated_position_vector_1 = generated_position_vector_1[generated_indices]
        shifttracker_position0 = shifttracker_position0[shifttracker_indices]
        shifttracker_position1 = shifttracker_position1[shifttracker_indices]

        self.assertTrue(np.all(generated_position_vector_0 == shifttracker_position0))
        self.assertTrue(np.all(generated_position_vector_1 == shifttracker_position1))

    def test_dimensional_spin_vector(self):
        ihspin_time_01 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                 "resources/main_test_simulation/stitched/%s/ihspin_hn_0.asc")
                                    % TestCompactObject.simulation_name, usecols=0)

        generated_time_01, generated_dimensional_spin_0 = TestCompactObject.primary_compact_object.dimensional_spin_vector
        ihspin0 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                          "resources/main_test_simulation/stitched/%s/ihspin_hn_0.asc")
                             % TestCompactObject.simulation_name, usecols=(1, 2, 3))

        generated_time_01, generated_dimensional_spin_1 = TestCompactObject.secondary_compact_object.dimensional_spin_vector
        ihspin1 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                          "resources/main_test_simulation/stitched/%s/ihspin_hn_1.asc")
                             % TestCompactObject.simulation_name, usecols=(1, 2, 3))

        ihspin_time_2 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                "resources/main_test_simulation/stitched/%s/ihspin_hn_2.asc")
                                   % TestCompactObject.simulation_name, usecols=0)

        generated_time_2, generated_dimensional_spin_2 = TestCompactObject.final_compact_object.dimensional_spin_vector
        ihspin2 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                          "resources/main_test_simulation/stitched/%s/ihspin_hn_2.asc")
                             % TestCompactObject.simulation_name, usecols=(1, 2, 3))

        time_01, generated_indices_01, ihspin_indices_01 = np.intersect1d(generated_time_01, ihspin_time_01,
                                                                          return_indices=True)

        time_2, generated_indices_2, ihspin_indices_2 = np.intersect1d(generated_time_2, ihspin_time_2,
                                                                       return_indices=True)

        generated_dimensional_spin_0 = generated_dimensional_spin_0[generated_indices_01]
        generated_dimensional_spin_1 = generated_dimensional_spin_1[generated_indices_01]
        generated_dimensional_spin_2 = generated_dimensional_spin_2[generated_indices_2]
        ihspin0 = ihspin0[ihspin_indices_01]
        ihspin1 = ihspin1[ihspin_indices_01]
        ihspin2 = ihspin2[ihspin_indices_2]

        self.assertTrue(np.all(generated_dimensional_spin_0 == ihspin0))
        self.assertTrue(np.all(generated_dimensional_spin_1 == ihspin1))
        self.assertTrue(np.all(generated_dimensional_spin_2 == ihspin2))

    @mock.patch("mayawaves.compactobject.CompactObject.get_data_from_columns")
    def test_dimensionless_spin_vector(self, compactobject_get_data_from_columns):
        time = [0, 1, 2, 3]
        dimensional_spinx = [0, 0, 0, 0]
        dimensional_spiny = [0, 0, 0, 0]
        dimensional_spinz = [0.5, 0.5, 0.5, 0.5]
        m_irr = [0.9, 0.9, 0.9, 0.9]
        compactobject_get_data_from_columns.return_value = np.column_stack(
            [time, dimensional_spinx, dimensional_spiny, dimensional_spinz, m_irr])

        generated_time, generated_dimensionless_spin = TestCompactObject.primary_compact_object.dimensionless_spin_vector

        dimensional_spin = np.column_stack([dimensional_spinx, dimensional_spiny, dimensional_spinz])
        horizon_mass = np.sqrt(
            np.power(m_irr, 2) + np.power(np.linalg.norm(dimensional_spin, axis=1), 2) / (4 * np.power(m_irr, 2)))

        expected_dimensionless_spin = dimensional_spin / np.power(horizon_mass.reshape(len(horizon_mass), 1), 2)

        self.assertTrue(np.all(expected_dimensionless_spin == generated_dimensionless_spin))

    @mock.patch("mayawaves.compactobject.CompactObject.get_data_from_columns")
    def test_horizon_mass(self, compactobject_get_data_from_columns):
        time = [0, 1, 2, 3]
        dimensional_spinx = [0, 0, 0, 0]
        dimensional_spiny = [0, 0, 0, 0]
        dimensional_spinz = [0.5, 0.5, 0.5, 0.5]
        m_irr = [0.9, 0.9, 0.9, 0.9]
        compactobject_get_data_from_columns.return_value = np.column_stack(
            [time, dimensional_spinx, dimensional_spiny, dimensional_spinz, m_irr])

        generated_time, generated_horizon_mass = TestCompactObject.primary_compact_object.horizon_mass

        dimensional_spin = np.column_stack([dimensional_spinx, dimensional_spiny, dimensional_spinz])
        expected_horizon_mass = np.sqrt(
            np.power(m_irr, 2) + np.power(np.linalg.norm(dimensional_spin, axis=1), 2) / (4 * np.power(m_irr, 2)))

        self.assertTrue(np.all(expected_horizon_mass == generated_horizon_mass))

    def test_momentum_vector(self):
        ihspin_time_01 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                 "resources/main_test_simulation/stitched/%s/ihspin_hn_0.asc")
                                    % TestCompactObject.simulation_name, usecols=0)

        generated_time_01, generated_momentum_0 = TestCompactObject.primary_compact_object.momentum_vector
        ihspin0 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                          "resources/main_test_simulation/stitched/%s/ihspin_hn_0.asc")
                             % TestCompactObject.simulation_name, usecols=(4, 5, 6))

        generated_time_01, generated_momentum_1 = TestCompactObject.secondary_compact_object.momentum_vector
        ihspin1 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                          "resources/main_test_simulation/stitched/%s/ihspin_hn_1.asc")
                             % TestCompactObject.simulation_name, usecols=(4, 5, 6))

        ihspin_time_2 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                "resources/main_test_simulation/stitched/%s/ihspin_hn_2.asc")
                                   % TestCompactObject.simulation_name, usecols=0)

        generated_time_2, generated_momentum_2 = TestCompactObject.final_compact_object.momentum_vector
        ihspin2 = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                          "resources/main_test_simulation/stitched/%s/ihspin_hn_2.asc")
                             % TestCompactObject.simulation_name, usecols=(4, 5, 6))

        time_01, generated_indices_01, ihspin_indices_01 = np.intersect1d(generated_time_01, ihspin_time_01,
                                                                          return_indices=True)

        time_2, generated_indices_2, ihspin_indices_2 = np.intersect1d(generated_time_2, ihspin_time_2,
                                                                       return_indices=True)

        generated_momentum_0 = generated_momentum_0[generated_indices_01]
        generated_momentum_1 = generated_momentum_1[generated_indices_01]
        generated_momentum_2 = generated_momentum_2[generated_indices_2]
        ihspin0 = ihspin0[ihspin_indices_01]
        ihspin1 = ihspin1[ihspin_indices_01]
        ihspin2 = ihspin2[ihspin_indices_2]

        self.assertTrue(np.all(generated_momentum_0 == ihspin0))
        self.assertTrue(np.all(generated_momentum_1 == ihspin1))
        self.assertTrue(np.all(generated_momentum_2 == ihspin2))

    def test_velocity_vector(self):
        shifttracker_time = np.loadtxt(
            os.path.join(TestCompactObject.CURR_DIR, "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc") %
            TestCompactObject.simulation_name, usecols=1)

        generated_time, generated_velocity_vector_0 = TestCompactObject.primary_compact_object.velocity_vector
        shifttracker_velocity0 = np.loadtxt(
            os.path.join(TestCompactObject.CURR_DIR, "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc") %
            TestCompactObject.simulation_name, usecols=(5, 6, 7))

        generated_time, generated_velocity_vector_1 = TestCompactObject.secondary_compact_object.velocity_vector
        shifttracker_velocity1 = np.loadtxt(
            os.path.join(TestCompactObject.CURR_DIR, "resources/main_test_simulation/stitched/%s/ShiftTracker1.asc") %
            TestCompactObject.simulation_name, usecols=(5, 6, 7))

        time, generated_indices, shifttracker_indices = np.intersect1d(generated_time, shifttracker_time,
                                                                       return_indices=True)

        generated_velocity_vector_0 = generated_velocity_vector_0[generated_indices]
        generated_velocity_vector_1 = generated_velocity_vector_1[generated_indices]
        shifttracker_velocity0 = shifttracker_velocity0[shifttracker_indices]
        shifttracker_velocity1 = shifttracker_velocity1[shifttracker_indices]

        self.assertTrue(np.all(generated_velocity_vector_0 == shifttracker_velocity0))
        self.assertTrue(np.all(generated_velocity_vector_1 == shifttracker_velocity1))

        time, final_position_vector = TestCompactObject.coalescence.final_compact_object.position_vector
        expected_velocity_vector = np.gradient(final_position_vector, axis=1) / np.gradient(time)[:, None]
        time, generated_velocity_vector = TestCompactObject.coalescence.final_compact_object.velocity_vector
        self.assertTrue(np.allclose(expected_velocity_vector, generated_velocity_vector, atol=1e-6))

    def test_apparent_horizon_mean_curvature(self):
        generated_time_1, generated_mean_curvature_1 = TestCompactObject.primary_compact_object.apparent_horizon_mean_curvature
        ah1_time, ah1_mean_curvature = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                                  % TestCompactObject.simulation_name, usecols=(1, 31), unpack=True)
        generated_time_2, generated_mean_curvature_2 = TestCompactObject.secondary_compact_object.apparent_horizon_mean_curvature
        ah2_time, ah2_mean_curvature = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                                  % TestCompactObject.simulation_name,
                                                  usecols=(1, 31), unpack=True)

        time_1, generated_indices_1, ah1_indices = np.intersect1d(generated_time_1, ah1_time, return_indices=True)
        time_2, generated_indices_2, ah2_indices = np.intersect1d(generated_time_2, ah2_time, return_indices=True)

        generated_mean_curvature_1 = generated_mean_curvature_1[generated_indices_1]
        generated_mean_curvature_2 = generated_mean_curvature_2[generated_indices_2]
        ah1_mean_curvature = ah1_mean_curvature[ah1_indices]
        ah2_mean_curvature = ah2_mean_curvature[ah2_indices]

        self.assertTrue(np.all(generated_mean_curvature_1 == ah1_mean_curvature))
        self.assertTrue(np.all(generated_mean_curvature_2 == ah2_mean_curvature))

    def test_apparent_horizon_area(self):
        generated_time_1, generated_area_1 = TestCompactObject.primary_compact_object.apparent_horizon_area
        ah1_time, ah1_area = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                            "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                        % TestCompactObject.simulation_name, usecols=(1, 25), unpack=True)

        generated_time_2, generated_area = TestCompactObject.secondary_compact_object.apparent_horizon_area
        ah2_time, ah2_area = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                            "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                        % TestCompactObject.simulation_name, usecols=(1, 25), unpack=True)

        ah1_time, generated_indices_1, ah1_indices = np.intersect1d(generated_time_1, ah1_time, return_indices=True)
        ah2_time, generated_indices_2, ah2_indices = np.intersect1d(generated_time_2, ah2_time, return_indices=True)

        generated_area_1 = generated_area_1[generated_indices_1]
        generated_area = generated_area[generated_indices_2]
        ah1_area = ah1_area[ah1_indices]
        ah2_area = ah2_area[ah2_indices]

        self.assertTrue(np.all(generated_area_1 == ah1_area))
        self.assertTrue(np.all(generated_area == ah2_area))

    def test_apparent_horizon_circumferences(self):
        generated_time_1, generated_xx_1, generated_xy_1, generated_yz_1 = \
            TestCompactObject.primary_compact_object.apparent_horizon_circumferences

        ah1_time, ah1_xx, ah1_xy, ah1_yz = \
            np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
            "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
            % TestCompactObject.simulation_name, usecols=(1,20,21,22),
            unpack=True)

        generated_time_2, generated_xx_2, generated_xy_2, generated_yz_2 = \
            TestCompactObject.secondary_compact_object.apparent_horizon_circumferences

        ah2_time, ah2_xx, ah2_xy, ah2_yz = \
             np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
            "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
            % TestCompactObject.simulation_name, usecols=(1,20,21,22),
            unpack=True)

        ah1_time, generated_indices_1, ah1_indices = np.intersect1d(
            generated_time_1, ah1_time, return_indices=True)
        ah2_time, generated_indices_2, ah2_indices = np.intersect1d(
            generated_time_2, ah2_time, return_indices=True)

        generated_xx_1 = generated_xx_1[generated_indices_1]
        generated_xy_1 = generated_xy_1[generated_indices_1]
        generated_yz_1 = generated_yz_1[generated_indices_1]

        generated_xx_2 = generated_xx_2[generated_indices_2]
        generated_xy_2 = generated_xy_2[generated_indices_2]
        generated_yz_2 = generated_yz_2[generated_indices_2]

        ah1_xx = ah1_xx[ah1_indices]
        ah1_xy = ah1_xy[ah1_indices]
        ah1_yz = ah1_yz[ah1_indices]

        ah2_xx = ah2_xx[ah2_indices]
        ah2_xy = ah2_xy[ah2_indices]
        ah2_yz = ah2_yz[ah2_indices]

        self.assertTrue(np.all(generated_xx_1 == ah1_xx))
        self.assertTrue(np.all(generated_xy_1 == ah1_xy))
        self.assertTrue(np.all(generated_yz_1 == ah1_yz))

        self.assertTrue(np.all(generated_xx_2 == ah2_xx))
        self.assertTrue(np.all(generated_xy_2 == ah2_xy))
        self.assertTrue(np.all(generated_yz_2 == ah2_yz))

    def test_irreducible_mass(self):
        ah1_time = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                              % TestCompactObject.simulation_name, usecols=1)

        generated_time, generated_mass_1 = TestCompactObject.primary_compact_object.irreducible_mass
        ah1_mass = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                              % TestCompactObject.simulation_name, usecols=(26))

        generated_time, generated_mass_2 = TestCompactObject.secondary_compact_object.irreducible_mass
        ah2_mass = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                              % TestCompactObject.simulation_name, usecols=(26))

        time, generated_indices, ah_indices = np.intersect1d(generated_time, ah1_time, return_indices=True)

        generated_mass_1 = generated_mass_1[generated_indices]
        generated_mass_2 = generated_mass_2[generated_indices]
        ah1_mass = ah1_mass[ah_indices]
        ah2_mass = ah2_mass[ah_indices]

        self.assertTrue(np.all(generated_mass_1 == ah1_mass))
        self.assertTrue(np.all(generated_mass_2 == ah2_mass))

    def test_apparent_horizon_areal_radius(self):
        generated_time_1, generated_areal_radius_1 = TestCompactObject.primary_compact_object.apparent_horizon_areal_radius
        ah1_time, ah1_areal_radius = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                                % TestCompactObject.simulation_name, usecols=(1, 27), unpack=True)

        generated_time_2, generated_areal_radius_2 = TestCompactObject.secondary_compact_object.apparent_horizon_areal_radius
        ah2_time, ah2_areal_radius = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                                % TestCompactObject.simulation_name, usecols=(1, 27), unpack=True)

        time_1, generated_indices_1, ah1_indices = np.intersect1d(generated_time_1, ah1_time, return_indices=True)
        time_2, generated_indices_2, ah2_indices = np.intersect1d(generated_time_2, ah2_time, return_indices=True)

        generated_areal_radius_1 = generated_areal_radius_1[generated_indices_1]
        generated_areal_radius_2 = generated_areal_radius_2[generated_indices_2]
        ah1_areal_radius = ah1_areal_radius[ah1_indices]
        ah2_areal_radius = ah2_areal_radius[ah2_indices]

        self.assertTrue(np.all(generated_areal_radius_1 == ah1_areal_radius))
        self.assertTrue(np.all(generated_areal_radius_2 == ah2_areal_radius))

    def test_apparent_horizon_expansion_theta_l(self):
        generated_time_1, generated_theta_l_1 = TestCompactObject.primary_compact_object.apparent_horizon_expansion_theta_l
        ah1_time, ah1_theta_l = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                           % TestCompactObject.simulation_name, usecols=(1, 28), unpack=True)

        generated_time_2, generated_theta_l_2 = TestCompactObject.secondary_compact_object.apparent_horizon_expansion_theta_l
        ah2_time, ah2_theta_l = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                           % TestCompactObject.simulation_name, usecols=(1, 28), unpack=True)

        time_1, generated_indices_1, ah1_indices = np.intersect1d(generated_time_1, ah1_time, return_indices=True)
        time_2, generated_indices_2, ah2_indices = np.intersect1d(generated_time_2, ah2_time, return_indices=True)

        generated_theta_l_1 = generated_theta_l_1[generated_indices_1]
        generated_theta_l_2 = generated_theta_l_2[generated_indices_2]
        ah1_theta_l = ah1_theta_l[ah1_indices]
        ah2_theta_l = ah2_theta_l[ah2_indices]

        self.assertTrue(np.all(generated_theta_l_1 == ah1_theta_l))
        self.assertTrue(np.all(generated_theta_l_2 == ah2_theta_l))

    def test_apparent_horizon_inner_expansion_theta_n(self):
        generated_time_1, generated_theta_n_1 = TestCompactObject.primary_compact_object.apparent_horizon_inner_expansion_theta_n
        ah1_time, ah1_theta_n = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                           % TestCompactObject.simulation_name, usecols=(1, 29), unpack=True)

        generated_time_2, generated_theta_n_2 = TestCompactObject.secondary_compact_object.apparent_horizon_inner_expansion_theta_n
        ah2_time, ah2_theta_n = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                           % TestCompactObject.simulation_name, usecols=(1, 29), unpack=True)

        time_1, generated_indices_1, ah1_indices = np.intersect1d(generated_time_1, ah1_time, return_indices=True)
        time_2, generated_indices_2, ah2_indices = np.intersect1d(generated_time_2, ah2_time, return_indices=True)

        generated_theta_n_1 = generated_theta_n_1[generated_indices_1]
        generated_theta_n_2 = generated_theta_n_2[generated_indices_2]
        ah1_theta_n = ah1_theta_n[ah1_indices]
        ah2_theta_n = ah2_theta_n[ah2_indices]

        self.assertTrue(np.all(generated_theta_n_1 == ah1_theta_n))
        self.assertTrue(np.all(generated_theta_n_2 == ah2_theta_n))

    def test_apparent_horizon_minimum_radius(self):
        generated_time_1, generated_min_radius_1 = TestCompactObject.primary_compact_object.apparent_horizon_minimum_radius
        ah1_time, ah1_min_radius = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                              % TestCompactObject.simulation_name, usecols=(1, 5), unpack=True)

        generated_time_2, generated_min_radius_2 = TestCompactObject.secondary_compact_object.apparent_horizon_minimum_radius
        ah2_time, ah2_min_radius = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                           "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                              % TestCompactObject.simulation_name, usecols=(1, 5), unpack=True)

        ah1_time, generated_indices_1, ah1_indices = np.intersect1d(generated_time_1, ah1_time, return_indices=True)
        ah2_time, generated_indices_2, ah2_indices = np.intersect1d(generated_time_2, ah2_time, return_indices=True)

        generated_min_radius_1 = generated_min_radius_1[generated_indices_1]
        generated_min_radius_2 = generated_min_radius_2[generated_indices_2]
        ah1_min_radius = ah1_min_radius[ah1_indices]
        ah2_min_radius = ah2_min_radius[ah2_indices]

        self.assertTrue(np.all(generated_min_radius_1 == ah1_min_radius))
        self.assertTrue(np.all(generated_min_radius_2 == ah2_min_radius))

    def test_apparent_horizon_maximum_radius(self):
        generated_time_1, generated_max_radius_1 = TestCompactObject.primary_compact_object.apparent_horizon_maximum_radius
        ah1_time, ah1_max_radius = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                 "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                              % TestCompactObject.simulation_name, usecols=(1, 6), unpack=True)

        generated_time_2, generated_max_radius_2 = TestCompactObject.secondary_compact_object.apparent_horizon_maximum_radius
        ah2_time, ah2_max_radius = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                 "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                              % TestCompactObject.simulation_name, usecols=(1, 6), unpack=True)

        ah1_time, generated_indices_1, ah1_indices = np.intersect1d(generated_time_1, ah1_time, return_indices=True)
        ah2_time, generated_indices_2, ah2_indices = np.intersect1d(generated_time_2, ah2_time, return_indices=True)

        generated_max_radius_1 = generated_max_radius_1[generated_indices_1]
        generated_max_radius_2 = generated_max_radius_2[generated_indices_2]
        ah1_max_radius = ah1_max_radius[ah1_indices]
        ah2_max_radius = ah2_max_radius[ah2_indices]

        self.assertTrue(np.all(generated_max_radius_1 == ah1_max_radius))
        self.assertTrue(np.all(generated_max_radius_2 == ah2_max_radius))

    def test_apparent_horizon_quadrupoles(self):
        generated_time_1, generated_xx_1, generated_xy_1, generated_xz_1, \
            generated_yy_1, generated_yz_1, generated_zz_1 = \
            TestCompactObject.primary_compact_object.apparent_horizon_quadrupoles

        ah1_time, ah1_xx, ah1_xy, ah1_xz, ah1_yy, ah1_yz, ah1_zz = \
            np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
            "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
            % TestCompactObject.simulation_name, usecols=(1, 8, 9, 10, 11, 12, 13),
            unpack=True)

        generated_time_2, generated_xx_2, generated_xy_2, generated_xz_2, \
            generated_yy_2, generated_yz_2, generated_zz_2 = \
            TestCompactObject.secondary_compact_object.apparent_horizon_quadrupoles

        ah2_time, ah2_xx, ah2_xy, ah2_xz, ah2_yy, ah2_yz, ah2_zz = \
             np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
            "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
            % TestCompactObject.simulation_name, usecols=(1, 8, 9, 10, 11, 12, 13),
            unpack=True)

        ah1_time, generated_indices_1, ah1_indices = np.intersect1d(
            generated_time_1, ah1_time, return_indices=True)
        ah2_time, generated_indices_2, ah2_indices = np.intersect1d(
            generated_time_2, ah2_time, return_indices=True)

        generated_xx_1 = generated_xx_1[generated_indices_1]
        generated_xy_1 = generated_xy_1[generated_indices_1]
        generated_xz_1 = generated_xz_1[generated_indices_1]
        generated_yy_1 = generated_yy_1[generated_indices_1]
        generated_yz_1 = generated_yz_1[generated_indices_1]
        generated_zz_1 = generated_zz_1[generated_indices_1]

        generated_xx_2 = generated_xx_2[generated_indices_2]
        generated_xy_2 = generated_xy_2[generated_indices_2]
        generated_xz_2 = generated_xz_2[generated_indices_2]
        generated_yy_2 = generated_yy_2[generated_indices_2]
        generated_yz_2 = generated_yz_2[generated_indices_2]
        generated_zz_2 = generated_zz_2[generated_indices_2]

        ah1_xx = ah1_xx[ah1_indices]
        ah1_xy = ah1_xy[ah1_indices]
        ah1_xz = ah1_xz[ah1_indices]
        ah1_yy = ah1_yy[ah1_indices]
        ah1_yz = ah1_yz[ah1_indices]
        ah1_zz = ah1_zz[ah1_indices]

        ah2_xx = ah2_xx[ah2_indices]
        ah2_xy = ah2_xy[ah2_indices]
        ah2_xz = ah2_xz[ah2_indices]
        ah2_yy = ah2_yy[ah2_indices]
        ah2_yz = ah2_yz[ah2_indices]
        ah2_zz = ah2_zz[ah2_indices]

        self.assertTrue(np.all(generated_xx_1 == ah1_xx))
        self.assertTrue(np.all(generated_xy_1 == ah1_xy))
        self.assertTrue(np.all(generated_xz_1 == ah1_xz))
        self.assertTrue(np.all(generated_yy_1 == ah1_yy))
        self.assertTrue(np.all(generated_yz_1 == ah1_yz))
        self.assertTrue(np.all(generated_zz_1 == ah1_zz))

        self.assertTrue(np.all(generated_xx_2 == ah2_xx))
        self.assertTrue(np.all(generated_xy_2 == ah2_xy))
        self.assertTrue(np.all(generated_xz_2 == ah2_xz))
        self.assertTrue(np.all(generated_yy_2 == ah2_yy))
        self.assertTrue(np.all(generated_yz_2 == ah2_yz))
        self.assertTrue(np.all(generated_zz_2 == ah2_zz))

    def test_apparent_horizon_mean_radius(self):
        generated_time_1, generated_mean_radius_1 = TestCompactObject.primary_compact_object.apparent_horizon_mean_radius
        ah1_time, ah1_mean_radius = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                 "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp")
                                               % TestCompactObject.simulation_name, usecols=(1, 7), unpack=True)

        generated_time_2, generated_mean_radius_2 = TestCompactObject.secondary_compact_object.apparent_horizon_mean_radius
        ah2_time, ah2_mean_radius = np.loadtxt(os.path.join(TestCompactObject.CURR_DIR,
                                                 "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp")
                                               % TestCompactObject.simulation_name, usecols=(1, 7), unpack=True)

        ah1_time, generated_indices_1, ah1_indices = np.intersect1d(generated_time_1, ah1_time, return_indices=True)
        ah2_time, generated_indices_2, ah2_indices = np.intersect1d(generated_time_2, ah2_time, return_indices=True)

        generated_mean_radius_1 = generated_mean_radius_1[generated_indices_1]
        generated_mean_radius_2 = generated_mean_radius_2[generated_indices_2]
        ah1_mean_radius = ah1_mean_radius[ah1_indices]
        ah2_mean_radius = ah2_mean_radius[ah2_indices]

        self.assertTrue(np.all(generated_mean_radius_1 == ah1_mean_radius))
        self.assertTrue(np.all(generated_mean_radius_2 == ah2_mean_radius))

    def test_last_available_spin_data_time(self):
        last_spin_data_time = TestCompactObject.primary_compact_object.last_available_spin_data_time
        self.assertTrue(np.isclose(2.775000000000000000e+01, last_spin_data_time, atol=1e-4))

        last_spin_data_time = TestCompactObject.secondary_compact_object.last_available_spin_data_time
        self.assertTrue(np.isclose(2.775000000000000000e+01, last_spin_data_time, atol=1e-4))

        last_spin_data_time = TestCompactObject.final_compact_object.last_available_spin_data_time
        self.assertTrue(np.isclose(3.000000000000000000e+02, last_spin_data_time, atol=1e-4))

    def test_dimensional_spin_at_time(self):
        last_available_spin_time = TestCompactObject.primary_compact_object.last_available_spin_data_time

        spin_data = TestCompactObject.primary_compact_object.dimensional_spin_at_time(last_available_spin_time + 5)
        self.assertIsNone(spin_data)

        dimensional_spin = TestCompactObject.final_compact_object.dimensional_spin_at_time(0)
        self.assertIsNone(dimensional_spin)

        expected_dimensional_spin_0_10M = [-7.072372678127300107e-11, 1.804307766674618845e-10,
                                           -6.407273338403585757e-04]
        generated_dimensional_spin_0_10M = TestCompactObject.primary_compact_object.dimensional_spin_at_time(10)
        expected_dimensional_spin_0_20M = [-1.632439070645936855e-09, 1.139439490235757341e-08,
                                           -4.491838961758458197e-03]
        generated_dimensional_spin_0_20M = TestCompactObject.primary_compact_object.dimensional_spin_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_dimensional_spin_0_10M, generated_dimensional_spin_0_10M, atol=1e-3)))
        self.assertTrue(
            np.all(np.isclose(expected_dimensional_spin_0_20M, generated_dimensional_spin_0_20M, atol=1e-3)))

        expected_dimensional_spin_1_10M = [-1.000842083725944768e-10, 1.649309155827778844e-10,
                                           -7.455281376387049800e-04]
        generated_dimensional_spin_1_10M = TestCompactObject.secondary_compact_object.dimensional_spin_at_time(10)
        expected_dimensional_spin_1_20M = [-1.632439141093456147e-09, 1.139439494544503929e-08,
                                           -4.491838932933941639e-03]
        generated_dimensional_spin_1_20M = TestCompactObject.secondary_compact_object.dimensional_spin_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_dimensional_spin_1_10M, generated_dimensional_spin_1_10M, atol=1e-3)))
        self.assertTrue(
            np.all(np.isclose(expected_dimensional_spin_1_20M, generated_dimensional_spin_1_20M, atol=1e-3)))

    def test_irreducible_mass_at_time(self):
        expected_irreducible_mass_0_10M = 5.172565147000000030e-01
        generated_irreducible_mass_0_10M = TestCompactObject.primary_compact_object.irreducible_mass_at_time(10)
        expected_irreducible_mass_0_20M = 5.179290616999999708e-01
        generated_irreducible_mass_0_20M = TestCompactObject.primary_compact_object.irreducible_mass_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_irreducible_mass_0_10M, generated_irreducible_mass_0_10M, atol=1e-4)))
        self.assertTrue(
            np.all(np.isclose(expected_irreducible_mass_0_20M, generated_irreducible_mass_0_20M, atol=1e-4)))

        expected_irreducible_mass_1_10M = 5.172565147000000030e-01
        generated_irreducible_mass_1_10M = TestCompactObject.secondary_compact_object.irreducible_mass_at_time(10)
        expected_irreducible_mass_1_20M = 5.179290616999999708e-01
        generated_irreducible_mass_1_20M = TestCompactObject.secondary_compact_object.irreducible_mass_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_irreducible_mass_1_10M, generated_irreducible_mass_1_10M, atol=1e-4)))
        self.assertTrue(
            np.all(np.isclose(expected_irreducible_mass_1_20M, generated_irreducible_mass_1_20M, atol=1e-4)))

    def test_horizon_mass_at_time(self):
        expected_horizon_mass_0_10M = 5.172565147000000030e-01
        generated_horizon_mass_0_10M = TestCompactObject.primary_compact_object.horizon_mass_at_time(10)
        expected_horizon_mass_0_20M = 5.179290616999999708e-01
        generated_horizon_mass_0_20M = TestCompactObject.primary_compact_object.horizon_mass_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_horizon_mass_0_10M, generated_horizon_mass_0_10M, atol=1e-4)))
        self.assertTrue(
            np.all(np.isclose(expected_horizon_mass_0_20M, generated_horizon_mass_0_20M, atol=1e-4)))

        expected_horizon_mass_1_10M = 5.172565147000000030e-01
        generated_horizon_mass_1_10M = TestCompactObject.secondary_compact_object.horizon_mass_at_time(10)
        expected_horizon_mass_1_20M = 5.179290616999999708e-01
        generated_horizon_mass_1_20M = TestCompactObject.secondary_compact_object.horizon_mass_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_horizon_mass_1_10M, generated_horizon_mass_1_10M, atol=1e-4)))
        self.assertTrue(
            np.all(np.isclose(expected_horizon_mass_1_20M, generated_horizon_mass_1_20M, atol=1e-4)))

    def test_dimensionless_spin_at_time(self):
        expected_dimensionless_spin_0_10M = [0, 0, 0]
        generated_dimensionless_spin_0_10M = TestCompactObject.primary_compact_object.dimensionless_spin_at_time(10)
        expected_dimensionless_spin_0_20M = [0, 0, 0]
        generated_dimensionless_spin_0_20M = TestCompactObject.primary_compact_object.dimensionless_spin_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_dimensionless_spin_0_10M, generated_dimensionless_spin_0_10M, atol=2e-2)))
        self.assertTrue(
            np.all(np.isclose(expected_dimensionless_spin_0_20M, generated_dimensionless_spin_0_20M, atol=2e-2)))

        expected_dimensionless_spin_1_10M = [0, 0, 0]
        generated_dimensionless_spin_1_10M = TestCompactObject.secondary_compact_object.dimensionless_spin_at_time(10)
        expected_dimensionless_spin_1_20M = [0, 0, 0]
        generated_dimensionless_spin_1_20M = TestCompactObject.secondary_compact_object.dimensionless_spin_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_dimensionless_spin_1_10M, generated_dimensionless_spin_1_10M, atol=2e-2)))
        self.assertTrue(
            np.all(np.isclose(expected_dimensionless_spin_1_20M, generated_dimensionless_spin_1_20M, atol=2e-2)))

        test_coalescence = Coalescence(os.path.join(TestCompactObject.CURR_DIR,
                                                    'resources/format_test/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_format_3.h5'))
        expected_dimensionless_spin_1_10M = [0, 0, 0.4]
        generated_dimensionless_spin_1_10M = test_coalescence.secondary_compact_object.dimensionless_spin_at_time(10)
        expected_dimensionless_spin_1_20M = [0, 0, 0.4]
        generated_dimensionless_spin_1_20M = test_coalescence.secondary_compact_object.dimensionless_spin_at_time(20)
        self.assertTrue(
            np.all(np.isclose(expected_dimensionless_spin_1_10M, generated_dimensionless_spin_1_10M, atol=2e-2)))
        self.assertTrue(
            np.all(np.isclose(expected_dimensionless_spin_1_20M, generated_dimensionless_spin_1_20M, atol=2e-2)))

    def test_data_array(self):
        test_compactobject = CompactObject(np.array([[np.nan, 2, 3, 4], [5, 6, np.nan, 8], [8, 7, 6, 5], [4, 3, 2, 1]]),
                                           ['a', 'b', 'c', 'd'],
                                           0,
                                           "BH",
                                           0,
                                           0,
                                           np.array([0, 0, 0]),
                                           np.array([0, 0, 0]))

        expected_data_array = np.array([[np.nan, 2, 3, 4], [5, 6, np.nan, 8], [8, 7, 6, 5], [4, 3, 2, 1]])
        actual_data_array = test_compactobject.data_array

        self.assertTrue(np.all(np.isnan(expected_data_array) == np.isnan(actual_data_array)))
        self.assertTrue(np.all(expected_data_array[np.where(~np.isnan(expected_data_array))] ==
                               actual_data_array[np.where(~np.isnan(actual_data_array))]))


