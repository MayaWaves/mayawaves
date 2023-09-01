from unittest import TestCase
import numpy as  np
import mayawaves.utils.postnewtonianutils as pnutils


class TestPostNewtonianUtils(TestCase):

    def test_orbital_frequency_from_separation(self):
        separation = 11
        mass_ratio = 3
        primary_dimensionless_spin = np.array([0, 0, 0.4])
        secondary_dimensionless_spin = np.array([0, 0, 0.4])
        expected_orb_freq = 0.024
        actual_orb_freq = pnutils.orbital_frequency_from_separation(separation, mass_ratio, primary_dimensionless_spin,
                                                                    secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_orb_freq, actual_orb_freq, atol=2e-3))

        separation = 9
        mass_ratio = 1.7
        primary_dimensionless_spin = np.array([0, 0, 0])
        secondary_dimensionless_spin = np.array([0, 0, 0])
        expected_orb_freq = 0.033
        actual_orb_freq = pnutils.orbital_frequency_from_separation(separation, mass_ratio, primary_dimensionless_spin,
                                                                    secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_orb_freq, actual_orb_freq, atol=2e-3))

    def test_separation_from_orbital_frequency(self):
        orb_freq = 0.024
        mass_ratio = 3
        primary_dimensionless_spin = np.array([0, 0, 0.4])
        secondary_dimensionless_spin = np.array([0, 0, 0.4])
        expected_sep = 11
        actual_sep = pnutils.separation_from_orbital_frequency(orb_freq, mass_ratio, primary_dimensionless_spin,
                                                                    secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_sep, actual_sep, atol=5e-1))

        orb_freq = 0.033
        mass_ratio = 1.7
        primary_dimensionless_spin = np.array([0, 0, 0])
        secondary_dimensionless_spin = np.array([0, 0, 0])
        expected_sep = 9
        actual_sep = pnutils.separation_from_orbital_frequency(orb_freq, mass_ratio, primary_dimensionless_spin,
                                                                    secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_sep, actual_sep, atol=5e-1))

    def test_tangential_momentum_from_separation(self):
        separation = 11
        mass_ratio = 3
        primary_dimensionless_spin = np.array([0, 0, 0.4])
        secondary_dimensionless_spin = np.array([0, 0, 0.4])
        expected_tangential_momentum = 0.066017
        actual_tangential_momentum = pnutils.tangential_momentum_from_separation(separation, mass_ratio, primary_dimensionless_spin,
                                                               secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_tangential_momentum, actual_tangential_momentum, atol=1e-3))

        separation = 9
        mass_ratio = 1.7
        primary_dimensionless_spin = np.array([0, 0, 0])
        secondary_dimensionless_spin = np.array([0, 0, 0])
        expected_tangential_momentum = 0.096590675294125
        actual_tangential_momentum = pnutils.tangential_momentum_from_separation(separation, mass_ratio, primary_dimensionless_spin,
                                                               secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_tangential_momentum, actual_tangential_momentum, atol=1e-3))

    def test_radial_momentum_from_separation(self):
        separation = 11
        mass_ratio = 3
        primary_dimensionless_spin = np.array([0, 0, 0.4])
        secondary_dimensionless_spin = np.array([0, 0, 0.4])
        expected_radial_momentum = 0.000425
        actual_radial_momentum = pnutils.radial_momentum_from_separation(separation, mass_ratio,
                                                                                 primary_dimensionless_spin,
                                                                                 secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_radial_momentum, actual_radial_momentum, atol=1e-4))

        separation = 9
        mass_ratio = 1.7
        primary_dimensionless_spin = np.array([0, 0, 0])
        secondary_dimensionless_spin = np.array([0, 0, 0])
        expected_radial_momentum = 0.001244520445218
        actual_radial_momentum = pnutils.radial_momentum_from_separation(separation, mass_ratio,
                                                                                 primary_dimensionless_spin,
                                                                                 secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_radial_momentum, actual_radial_momentum, atol=5e-4))

    def test_dH_dr_from_separation(self):
        separation = 11
        mass_ratio = 3
        primary_dimensionless_spin = np.array([0, 0, 0.4])
        secondary_dimensionless_spin = np.array([0, 0, 0.4])
        expected_dH_dr = 0.000568
        actual_dH_dr = pnutils.dH_dr_from_separation(separation, mass_ratio,
                                                                         primary_dimensionless_spin,
                                                                         secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_dH_dr, actual_dH_dr, atol=1e-5))

        separation = 9
        mass_ratio = 1.7
        primary_dimensionless_spin = np.array([0, 0, 0])
        secondary_dimensionless_spin = np.array([0, 0, 0])
        expected_dH_dr = 0.000889
        actual_dH_dr = pnutils.dH_dr_from_separation(separation, mass_ratio,
                                                                         primary_dimensionless_spin,
                                                                         secondary_dimensionless_spin)
        self.assertTrue(np.isclose(expected_dH_dr, actual_dH_dr, atol=1e-5))
