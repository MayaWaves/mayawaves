import numpy as np


def orbital_frequency_from_separation(separation: float, mass_ratio: float, primary_dimensionless_spin: np.ndarray,
                                      secondary_dimensionless_spin: np.ndarray) -> float:
    """The orbital frequency for a binary at a given separation.

    For a quasi-circular binary with the provided parameters, this returns the orbital frequency at the requested
    separation. Uses equations from https://arxiv.org/pdf/1702.00872.pdf.

    Args:
        separation (float): the separation between the two objects
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        primary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the larger object
        secondary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the smaller object

    Returns:
        float: orbital frequency for a quasi-circular binary at the given separation

    """
    a1x = primary_dimensionless_spin[0]
    a1y = primary_dimensionless_spin[1]
    a1z = primary_dimensionless_spin[2]

    a2x = secondary_dimensionless_spin[0]
    a2y = secondary_dimensionless_spin[1]
    a2z = secondary_dimensionless_spin[2]

    r = separation
    q = mass_ratio

    M = 1

    M_times_omega = (M / r) ** (3 / 2) * \
                    (
                            1
                            - (1 / 2) * (3 * q ** 2 + 5 * q + 3) / ((1 + q) ** 2) * (M / r)
                            + (- (1 / 4) * ((3 + 4 * q) * q * a1z) / (((1 + q) ** 2)) - (1 / 4) * (
                            ((3 * q + 4) * a2z) / ((1 + q) ** 2))) * (M / r) ** (3 / 2)
                            + (
                                    - (3 / 2) * (((a1x ** 2) * (q ** 2)) / ((1 + q) ** 2))
                                    - 3 * ((a1x * a2x * q) / ((1 + q) ** 2))
                                    + (3 / 4) * (((a1y ** 2) * (q ** 2)) / ((1 + q) ** 2))
                                    + (3 / 2) * ((a1y * a2y * q) / ((1 + q) ** 2))
                                    + (3 / 4) * (((a1z ** 2) * (q ** 2)) / ((1 + q) ** 2))
                                    + (3 / 2) * ((a1z * a2z * q) / ((1 + q) ** 2))
                                    - (3 / 2) * ((a2x ** 2) / ((1 + q) ** 2))
                                    + (3 / 4) * ((a2y ** 2) / ((1 + q) ** 2))
                                    + (3 / 4) * ((a2z ** 2) / ((1 + q) ** 2))
                                    + (1 / 16) * ((24 * (q ** 4) + 103 * (q ** 3) + 164 * (q ** 2) + 103 * q + 24) / (
                                    (1 + q) ** 4))
                            ) * (M / r) ** 2
                            + (
                                    (3 / 16) * (
                                    (q * (16 * (q ** 3) + 30 * (q ** 2) + 34 * q + 13) * a1z) / ((1 + q) ** 4))
                                    + (3 / 16) * (
                                            ((13 * (q ** 3) + 34 * (q ** 2) + 30 * q + 16) * a2z) / ((1 + q) ** 4))
                            ) * (M / r) ** (5 / 2)
                            + (
                                    (1 / 16) * (
                                    ((76 * (q ** 2) + 180 * q + 155) * (q ** 2) * (a1x ** 2)) / ((1 + q) ** 4))
                                    + (1 / 8) * (((120 * (q ** 2) + 187 * q + 120) * q * a2x * a1x) / ((1 + q) ** 4))
                                    - (1 / 8) * (((43 * (q ** 2) + 85 * q + 55) * (q ** 2) * (a1y ** 2)) / (
                                    (1 + q) ** 4))
                                    - (1 / 4) * (((54 * (q ** 2) + 95 * q + 54) * q * a2y * a1y) / ((1 + q) ** 4))
                                    - (1 / 32) * (
                                            ((2 * q + 5) * (14 * q + 27) * (q ** 2) * (a1z ** 2)) / ((1 + q) ** 4))
                                    - (1 / 16) * (((96 * (q ** 2) + 127 * q + 96) * q * a2z * a1z) / ((1 + q) ** 4))
                                    + (1 / 16) * (((155 * (q ** 2) + 180 * q + 76) * (a2x ** 2)) / ((1 + q) ** 4))
                                    - (1 / 8) * (((55 * (q ** 2) + 85 * q + 43) * (a2y ** 2)) / ((1 + q) ** 4))
                                    - (1 / 32) * (((27 * q + 14) * (5 * q + 2) * (a2z ** 2)) / ((1 + q) ** 4))
                                    + ((167 * (np.pi ** 2) * q) / (128 * ((1 + q) ** 2)))
                                    - ((120 * (q ** 6) + 2744 * (q ** 5) + 10049 * (q ** 4) + 14820 * (
                                    q ** 3) + 10049 * (
                                                q ** 2) + 2744 * q + 120) / (96 * ((1 + q) ** 6)))
                            ) * ((M / r) ** 3)
                    )

    omega = M_times_omega / M

    return omega


def separation_from_orbital_frequency(orbital_frequency: float, mass_ratio: float,
                                      primary_dimensionless_spin: np.ndarray, secondary_dimensionless_spin: np.ndarray
                                      ) -> float:
    """The separation at which a binary will have the given orbital frequency.

    For a quasi-circular binary with the provided parameters, this returns the separation in M at which the binary will
    have the given orbital frequency. Uses equations from https://arxiv.org/pdf/1702.00872.pdf.

    Args:
        orbital_frequency (float): the orbital frequency of the binary
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        primary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the larger object
        secondary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the smaller object

    Returns:
        float: the separation at which a quasi-circular binary will have the provided orbital frequency

    """
    a1x = primary_dimensionless_spin[0]
    a1y = primary_dimensionless_spin[1]
    a1z = primary_dimensionless_spin[2]

    a2x = secondary_dimensionless_spin[0]
    a2y = secondary_dimensionless_spin[1]
    a2z = secondary_dimensionless_spin[2]

    omega = orbital_frequency
    q = mass_ratio

    M = 1

    r_over_m = (M * omega) ** (-2 / 3) \
               - (1 / 3) * ((3 * (q ** 2) + 5 * q + 3) / ((1 + q) ** 2)) \
               + (
                       -(1 / 6) * (((3 + 4 * q) * q * a1z) / ((1 + q) ** 2))
                       - (1 / 6) * (((3 * q + 4) * a2z) / ((1 + q) ** 2))
               ) * (M * omega) ** (1 / 3) \
               + (
                       -((a1x ** 2 * (q ** 2)) / ((1 + q) ** 2))
                       - 2 * ((a1x * a2x * q) / ((1 + q) ** 2))
                       + (1 / 2) * ((a1y ** 2 * (q ** 2)) / ((1 + q) ** 2))
                       + ((a1y * a2y * q) / ((1 + q) ** 2))
                       + (1 / 2) * ((a1z ** 2 * (q ** 2)) / ((1 + q) ** 2))
                       + ((a1z * a2z * q) / ((1 + q) ** 2))
                       - ((a2x ** 2) / ((1 + q) ** 2))
                       + (1 / 2) * ((a2y ** 2) / ((1 + q) ** 2))
                       + (1 / 2) * ((a2z ** 2) / ((1 + q) ** 2))
                       - ((18 * (q ** 4) - 9 * (q ** 3) - 62 * (q ** 2) - 9 * q + 18) / (72 * (1 + q) ** 4))
               ) * (M * omega) ** (2 / 3) \
               + (
                       - (1 / 24) * ((q * (26 * (q ** 2) + 6 * q - 3) * a1z) / ((1 + q) ** 4))
                       + (1 / 24) * ((q * (3 * (q ** 2) - 6 * q - 26) * a2z) / ((1 + q) ** 4))
               ) * M * omega \
               + (
                       - (1 / 24) * (((q ** 2) * (8 * (q ** 2) - 40 * q - 71) * a1x ** 2) / ((1 + q) ** 4))
                       + (1 / 12) * ((q * (36 * (q ** 2) + 47 * q + 36) * a2x * a1x) / ((1 + q) ** 4))
                       - (1 / 6) * (((q ** 2) * (11 * (q ** 2) + 25 * q + 17) * a1y ** 2) / ((1 + q) ** 4))
                       - (1 / 2) * ((q * (11 * (q ** 2) + 20 * q + 11) * a2y * a1y) / ((1 + q) ** 4))
                       + (1 / 18) * (((q ** 2) * (7 * (q ** 2) - 15 * q - 27) * a1z ** 2) / ((1 + q) ** 4))
                       - (1 / 9) * ((q * (15 * (q ** 2) + 17 * q + 15) * a2z * a1z) / ((1 + q) ** 4))
                       + (1 / 24) * (((71 * (q ** 2) + 40 * q - 8) * a2x ** 2) / ((1 + q) ** 4))
                       - (1 / 6) * (((17 * (q ** 2) + 25 * q + 11) * a2y ** 2) / ((1 + q) ** 4))
                       - (1 / 18) * (((27 * (q ** 2) + 15 * q - 7) * a2z ** 2) / ((1 + q) ** 4))
                       + ((167 * np.pi ** 2 * q) / (192 * (1 + q) ** 2))
                       - ((324 * (q ** 6) + 16569 * (q ** 5) + 65304 * (q ** 4) + 98086 * (q ** 3) + 65304 * (
                       q ** 2) + 16569 * q + 324) / (1296 * (1 + q) ** 6))
               ) * (M * omega) ** (4 / 3)

    separation = r_over_m * M
    return separation


def tangential_momentum_from_separation(separation: float, mass_ratio: float, primary_dimensionless_spin: np.ndarray,
                                        secondary_dimensionless_spin: np.ndarray) -> float:
    """The tangential momentum for a quasi-circular binary at the given separation.

    For a binary of the provided parameters, this gives the tangential momentum to obtain a quasi-circular orbit. Uses
    equations from https://arxiv.org/pdf/1702.00872.pdf.

    Args:
        separation (float): the separation between the two objects
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        primary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the larger object
        secondary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the smaller object

    Returns:
        float: the tangential momentum of a quasi-circular orbit for the given binary

    """
    a1x = primary_dimensionless_spin[0]
    a1y = primary_dimensionless_spin[1]
    a1z = primary_dimensionless_spin[2]

    a2x = secondary_dimensionless_spin[0]
    a2y = secondary_dimensionless_spin[1]
    a2z = secondary_dimensionless_spin[2]

    r = separation
    q = mass_ratio

    M = 1

    p_t_over_m = q / ((1 + q) ** 2) * np.sqrt((M / r)) \
                 * (
                         1
                         + 2 * (M / r)
                         + (
                                 -(3 / 4) * (((3 + 4 * q) * q * a1z) / ((1 + q) ** 2))
                                 - (3 / 4) * (((3 * q + 4) * a2z) / ((1 + q) ** 2))
                         ) * ((M / r)) ** (3 / 2)
                         + (
                                 -(3 / 2) * ((a1x ** 2 * q ** 2) / ((1 + q) ** 2))
                                 - 3 * ((a1x * a2x * q) / ((1 + q) ** 2))
                                 + (3 / 4) * ((a1y ** 2 * q ** 2) / ((1 + q) ** 2))
                                 + (3 / 2) * ((a1y * a2y * q) / ((1 + q) ** 2))
                                 + (3 / 4) * ((a1z ** 2 * q ** 2) / ((1 + q) ** 2))
                                 + (3 / 2) * ((a1z * a2z * q) / ((1 + q) ** 2))
                                 - (3 / 2) * ((a2x ** 2) / ((1 + q) ** 2))
                                 + (3 / 4) * ((a2y ** 2) / ((1 + q) ** 2))
                                 + (3 / 4) * ((a2z ** 2) / ((1 + q) ** 2))
                                 + (1 / 16) * ((42 * q ** 2 + 41 * q + 42) / ((1 + q) ** 2))
                         ) * (M / r) ** 2
                         + (
                                 -(1 / 16) * ((q * (72 * q ** 3 + 116 * q ** 2 + 60 * q + 13) * a1z) / ((1 + q) ** 4))
                                 - (1 / 16) * (((13 * q ** 3 + 60 * q ** 2 + 116 * q + 72) * a2z) / ((1 + q) ** 4))
                         ) * (M / r) ** (5 / 2)
                         + (
                                 -(1 / 16) * ((q ** 2 * (80 * q ** 2 - 59) * a1x ** 2) / ((1 + q) ** 4))
                                 + (1 / 8) * ((q * (12 * q ** 2 + 35 * q + 12) * a2x * a1x) / ((1 + q) ** 4))
                                 - (1 / 2) * ((q ** 2 * (q ** 2 + 10 * q + 8) * a1y ** 2) / ((1 + q) ** 4))
                                 - (1 / 4) * ((q * (27 * q ** 2 + 58 * q + 27) * a2y * a1y) / ((1 + q) ** 4))
                                 + (1 / 32) * ((q ** 2 * (128 * q ** 2 + 56 * q - 27) * a1z ** 2) / ((1 + q) ** 4))
                                 + (1 / 16) * ((q * (60 * q ** 2 + 133 * q + 60) * a2z * a1z) / ((1 + q) ** 4))
                                 + (1 / 16) * (((59 * q ** 2 - 80) * a2x ** 2) / ((1 + q) ** 4))
                                 - (1 / 2) * (((8 * q ** 2 + 10 * q + 1) * a2y ** 2) / ((1 + q) ** 4))
                                 - (1 / 32) * (((27 * q ** 2 - 56 * q - 128) * a2z ** 2) / ((1 + q) ** 4))
                                 + ((163 * np.pi ** 2 * q) / (128 * (1 + q) ** 2))
                                 + (1 / 32) * ((120 * q ** 4 - 659 * q ** 3 - 1532 * q ** 2 - 659 * q + 120) / (
                                 (1 + q) ** 4))
                         ) * (M / r) ** 3
                 )

    p_t = p_t_over_m * M
    return p_t

def radial_momentum_from_separation(separation: float, mass_ratio: float, primary_dimensionless_spin: np.ndarray,
                                    secondary_dimensionless_spin: np.ndarray):
    """The radial momentum for a quasi-circular binary at the given separation.

    For a binary of the provided parameters, this gives the radial momentum to obtain a quasi-circular orbit. Uses
    equations from https://arxiv.org/pdf/1702.00872.pdf and https://arxiv.org/pdf/1810.00036.pdf.

    Args:
        separation (float): the separation between the two objects
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        primary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the larger object
        secondary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the smaller object

    Returns:
        float: the radial momentum of a quasi-circular orbit for the given binary

    """
    a1x = primary_dimensionless_spin[0]
    a1y = primary_dimensionless_spin[1]
    a1z = primary_dimensionless_spin[2]

    a2x = secondary_dimensionless_spin[0]
    a2y = secondary_dimensionless_spin[1]
    a2z = secondary_dimensionless_spin[2]

    chi_s_vec = (primary_dimensionless_spin + secondary_dimensionless_spin)/2
    chi_a_vec = (primary_dimensionless_spin - secondary_dimensionless_spin)/2

    chi_s = chi_s_vec[2]
    chi_a = chi_a_vec[2]

    r = separation
    q = mass_ratio

    M = 1
    m1 = q / (q+1)
    m2 = 1 / (q+1)

    eta = q/((1+q)**2)

    delta = (m1 - m2)/M

    omega = orbital_frequency_from_separation(separation=separation, mass_ratio=q,
                                              primary_dimensionless_spin=primary_dimensionless_spin,
                                              secondary_dimensionless_spin=secondary_dimensionless_spin)

    v = (M * omega) ** (1/3)

    dM_dt = (32 / 5) * v ** 10 * eta ** 2 * (
            -((v ** 5) / 4) *
            (
                    (1 - 3 * eta) * chi_s * (1 + 3 * chi_s ** 2 + 9 * chi_a ** 2)
                    + (1 - eta) * delta * chi_a * (1 + 3 * chi_a ** 2 + 9 * chi_s ** 2)
            )
    )

    S_l = m1**2 * a1z + m2**2 * a2z
    sigma_l = m2*a2z - m1*a1z
    gamma = 1

    d_E_gw_dt = ((32 / 5) * eta ** 2 * omega ** (10 / 3)) * \
                (1
                 + ((35 * eta) / (12) - (1247 / 336)) * omega ** (2 / 3)
                 + (4 * np.pi - ((5 * delta) / 4) * sigma_l - 4 * S_l) * omega
                 + (
                         ((65 * eta ** 2) / 18)
                         + ((9271 * eta) / 504)
                         - 44711 / 9072
                         - ((89 * delta * chi_a * chi_s) / 48)
                         + ((287 * delta * chi_a * chi_s) / 48)
                         + ((33 / 16) - 8 * eta) * chi_a ** 2
                         - ((33 / 16) - (eta / 4)) * chi_s ** 2
                 ) * omega ** (4 / 3) \
                 + (
                         np.pi * ((583 * eta / 24) - (8191 / 672))
                         + (43 * delta * eta * sigma_l / 4)
                         - (13 * delta * sigma_l / 16)
                         + (272 * eta * S_l / 9)
                         - (9 * S_l / 2)
                 ) * omega ** (5 / 3) \
                 + omega ** 2 * (
                         -(4843497781 / 69854400)
                         - (775 * eta ** 3 / 324)
                         - (94403 * eta ** 2 / 3024)
                         + (8009293 / 54432 - (41 * np.pi ** 2 / 64)) * eta
                         + ((287 * np.pi ** 2) / 192)
                         + (1712 / 105) * (- gamma + ((35 * np.pi ** 2) / 107) - (1 / 2) * np.log(
                     16 * omega ** (2 / 3)))  # check if this is correct base
                         - ((31 * np.pi * delta * sigma_l) / 6)
                         - 16 * np.pi * S_l
                         + delta * (((611 / 252) - (809 * eta)) / 18) * chi_a * chi_s
                         + (43 * eta ** 2 - ((8345 * eta) / 504) + (611 / 504)) * chi_a ** 2
                         + (((173 * eta ** 2) / 18) - ((2393 * eta) / 72) + (611 / 504)) * chi_s ** 2
                 ) \
                 + (-((31 * np.pi * delta * sigma_l) / 6) - 16 * np.pi * S_l) * omega ** 3 \
                 + (
                         np.pi * (((193385 * eta ** 2) / 3024) + ((214745 * eta) / 1728) - (16285 / 504)
                                  - (1501 / 36) * delta * eta ** 2 * sigma_l
                                  + ((1849 * delta * eta * sigma_l) / 126)
                                  + ((9535 * delta * sigma_l) / 336)
                                  - ((2810 * eta ** 2 * S_l) / 27)
                                  + ((6172 * eta * S_l) / 189)
                                  + ((476645 * S_l) / 6804)
                                  + delta * (((34 * eta) / 3) - (71 / 24)) * chi_a ** 3
                                  + delta * (((109 * eta) / 6) - (71 / 8)) * chi_a * chi_s ** 2
                                  + (-((104 * eta ** 2) / 3) + ((263 * eta) / 6) - (71 / 8)) * chi_a ** 2 * chi_s
                                  + (-((2 * eta ** 2) / 3) + (
                                 (28 * eta) / 3)  # originally was 28 * nu but I don't think that was right
                                     - (71 / 24)) * chi_s ** 3
                                  ) * omega ** (7 / 3)
                         + (
                                 ((130583 * np.pi * delta * eta * sigma_l) / 2016)
                                 - ((7163 * np.pi * delta * sigma_l) / 672)
                                 + ((13879 * np.pi * eta * S_l) / 72)
                                 - ((3485 * np.pi * S_l) / 96)
                         ) * omega ** (8 / 3)
                 )
                 )

    dh_dr = dH_dr_from_separation(separation=separation, mass_ratio=q,
                                  primary_dimensionless_spin=primary_dimensionless_spin,
                                  secondary_dimensionless_spin=secondary_dimensionless_spin)

    dr_dt = (d_E_gw_dt + dM_dt) / dh_dr

    p_r_over_m = (
                 dr_dt
                 - (
                                 (1 / 2) * ((q ** 2 * a1x * a1y) / ((1 + q) ** 4))
                                 - (1 / 4) * ((q ** 2 * a2y * a1x) / ((1 + q) ** 4))
                                 - (1 / 4) * ((q ** 2 * a2x * a1y) / ((1 + q) ** 4))
                                 + (1 / 2) * ((q ** 2 * a2y * a2x) / ((1 + q) ** 4))
                         ) * ((M / r)) ** (7 / 2)
                 ) / (
                         (((1 + q) ** 2) / q)
                         - (1 / 2) * (((7 * q ** 2 + 15 * q + 7) / q) * (M / r))
                         + (1 / 8) * (((47 * q ** 4 + 229 * q ** 3 + 363 * q ** 2 + 229 * q + 47)) / (
                             q * r ** 2 * (1 + q))) * (M / r) ** 2
                         + (
                                 (1 / 4) * (((12 * q ** 2 + 11 * q + 4) * a1z) / ((1 + q)))
                                 + (1 / 4) * (((4 * q ** 2 + 11 * q + 12) * a2z) / ((1 + q) * q))
                         ) * (M / r) ** (5 / 2)
                         + (
                                 - (1 / 16) * (np.pi ** 2)
                                 - (1 / 48) * (
                                         ((363 * q ** 6
                                           + 2608 * q ** 5
                                           + 7324 * q ** 4
                                           + 10161 * q ** 3
                                           + 7324 * q ** 2
                                           + 2608 * q
                                           + 363))
                                         / (q * (1 + q) ** 4)
                                 )
                                 + (1 / 4) * (((18 * q ** 2 + 6 * q + 5) * q * a1x ** 2) / ((1 + q) ** 2))
                                 + (((3 * q ** 2 - q + 3) * a2x * a1x) / ((1 + q) ** 2))
                                 - (3 / 4) * (((3 * q ** 2 + q + 1) * q * a1y ** 2) / ((1 + q) ** 2))
                                 - (1 / 2) * (((3 * q ** 2 - 2 * q + 3) * a2y * a1y) / ((1 + q) ** 2))
                                 - (3 / 4) * (((3 * q ** 2 + q + 1) * q * a1z ** 2) / ((1 + q) ** 2))
                                 - (1 / 2) * (((3 * q ** 2 - 2 * q + 3) * a2z * a1z) / ((1 + q) ** 2))
                                 + (1 / 4) * (((5 * q ** 2 + 6 * q + 18) * a2x ** 2) / (q * (1 + q) ** 2))
                                 - (3 / 4) * (((q ** 2 + q + 3) * a2y ** 2) / (q * (1 + q) ** 2))
                                 - (3 / 4) * (((q ** 2 + q + 3) * a2z ** 2) / (q * (1 + q) ** 2))
                         ) * (M / r) ** 3
                 )

    p_r = p_r_over_m * M
    return p_r

def dH_dr_from_separation(separation: float, mass_ratio: float, primary_dimensionless_spin: np.ndarray,
                          secondary_dimensionless_spin: np.ndarray) -> float:
    """Derivative of the hamiltonian with respect to separation.

    Derivative of the hamiltonian with respect to separation at the given separation for a binary with the provided
    parameters.

    Args:
        separation (float): the separation between the two objects
        mass_ratio (float): the ratio of the masses of the two objects, :math:`q = m_1 / m_2 > 1`
        primary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the larger object
        secondary_dimensionless_spin (numpy.ndarray): the dimensionless spin of the smaller object

    Returns:
        float: derivative of the hamiltonian with respect to separation

    """
    a1x = primary_dimensionless_spin[0]
    a1y = primary_dimensionless_spin[1]
    a1z = primary_dimensionless_spin[2]

    a2x = secondary_dimensionless_spin[0]
    a2y = secondary_dimensionless_spin[1]
    a2z = secondary_dimensionless_spin[2]

    r = separation
    q = mass_ratio

    dh_dr = (q / (2 * r**2 * (q + 1) ** 2)) \
            - (2 / (r ** 3)) * ((q * (7 * q ** 2 + 13 * q + 7)) / (8 * (q + 1) ** 4)) \
            - (5 / (2 * r ** (7 / 2))) * (
                    -(((4 * q + 3) * q ** 2 * a2z) / (4 * (q + 1) ** 4))
                    - (((3 * q + 4) * q * a1z) / (4 * (q + 1) ** 4))
            ) \
            - (3 / (r ** 4)) * (
                    -((q ** 3 * a2x ** 2) / (2 * (q + 1) ** 4))
                    + ((q ** 3 * a2z ** 2) / (4 * (q + 1) ** 4))
                    - ((q ** 2 * a1x * a2x) / ((q + 1) ** 4))
                    + ((q ** 2 * a1y * a2y) / (2 * (q + 1) ** 4))
                    + ((q ** 2 * a1z * a2z) / (2 * (q + 1) ** 4))
                    + (((9 * q ** 4 + 16 * q ** 3 + 13 * q ** 2 + 16 * q + 9) * q) / (16 * (q + 1) ** 6))
                    - ((q * a1x ** 2) / (2 * (q + 1) ** 4))
                    + ((q * a1y ** 2) / (4 * (q + 1) ** 4))
                    + ((q * a1z ** 2) / (4 * (q + 1) ** 4))
            ) \
            - (7 / (2 * r ** (9 / 2))) * (
                    -(((q ** 3 + 14 * q ** 2 + 42 * q + 32) * q * a1z) / (16 * (q + 1) ** 6))
                    - (((32 * q ** 3 + 42 * q ** 2 + 14 * q + 1) * q ** 2 * a2z) / (16 * (q + 1) ** 6))
            ) \
            - (4 / (r ** 5)) * (
                    ((179 * q ** 7) / (128 * (q + 1) ** 8))
                    - ((3497 * q ** 6) / (384 * (q + 1) ** 8))
                    - ((18707 * q ** 5) / (384 * (q + 1) ** 8))
                    - ((9787 * q ** 4) / (128 * (q + 1) ** 8))
                    + ((9 * q ** 3 * a1x * a2x) / (8 * (q + 1) ** 6))
                    - ((18707 * q ** 3) / (384 * (q + 1) ** 8))
                    + (((25 * q ** 2 - 12 * q - 52) * q * a1x ** 2) / (16 * (q + 1) ** 6))
                    - ((3 * (4 * q ** 2 + 9 * q + 4) * q ** 2 * a1y * a2y) / (4 * (q + 1) ** 6))
                    + ((3 * (10 * q ** 2 + 21 * q + 10) * q ** 2 * a1z * a2z) / (8 * (q + 1) ** 6)) # typo in the paper
                    + (((3 * q ** 2 + 38 * q + 50) * q * a1z ** 2) / (16 * (q + 1) ** 6)) # typo in paper
                    - ((3497 * q ** 2) / (384 * (q + 1) ** 8))
                    - (((52 * q ** 2 + 12 * q - 25) * q ** 3 * a2x ** 2) / (16 * (q + 1) ** 6))
                    + (((q ** 2 - 17 * q - 15) * q ** 3 * a2y ** 2) / (8 * (q + 1) ** 6))
                    + (((-15 * q ** 3 - 17 * q ** 2 + q) * a1y ** 2) / (8 * (q + 1) ** 6)) # typo in paper
                    + (((50 * q ** 2 + 38 * q + 3) * q ** 3 * a2z ** 2) / (16 * (q + 1) ** 6))
                    + np.pi ** 2 * (((81 * q ** 6) / (128 * (q + 1) ** 8))
                                    + ((81 * q ** 5) / (32 * (q + 1) ** 8))
                                    + ((243 * q ** 4) / (64 * (q + 1) ** 8))
                                    + ((81 * q ** 3) / (32 * (q + 1) ** 8))
                                    + ((81 * q ** 2) / (128 * (q + 1) ** 8))
                                    )
                    + ((179 * q) / (128 * (q + 1) ** 8))
            ) \
            - (9 / (2 * r ** (11 / 2))) * (
                    ((3 * (20 * q + 7) * q ** 4 * a2x ** 2 * a2z) / (8 * (q + 1) ** 6))
                    - ((3 * (12 * q + 5) * q ** 4 * a2y ** 2 * a2z) / (16 * (q + 1) ** 6))
                    - ((3 * (12 * q + 5) * q ** 4 * a2z ** 3) / (16 * (q + 1) ** 6))
                    + a1x ** 2 * (
                            ((3 * (22 * q + 15) * q ** 2 * a2z) / (8 * (q + 1) ** 6))
                            + ((3 * (7 * q + 20) * q * a1z) / (8 * (q + 1) ** 6))
                    )
                    + a1y ** 2 * (-((3 * (28 * q + 15) * q ** 2 * a2z) / (16 * (q + 1) ** 6))
                                  - ((3 * (5 * q + 12) * q * a1z) / (16 * (q + 1) ** 6))) # typo in the paper
                    - ((3 * (22 * q + 23) * q ** 2 * a1z ** 2 * a2z) / (16 * (q + 1) ** 6))
                    + a1x * (
                            ((3 * (5 * q + 3) * q ** 3 * a2x * a2z) / (2 * (q + 1) ** 6))
                            + ((3 * (3 * q + 5) * q ** 2 * a2x * a1z) / (2 * (q + 1) ** 6))
                    )
                    + a1y * (((3 * (3 - 4 * q) * q ** 3 * a2y * a2z) / (8 * (q + 1) ** 6))
                             + ((3 * (3 * q - 4) * q ** 2 * a2y * a1z) / (8 * (q + 1) ** 6))) # typo in the paper
                    + a1z * (
                            ((3 * (15 * q + 22) * q ** 3 * a2x ** 2) / (8 * (q + 1) ** 6))
                            - ((3 * (15 * q + 28) * q ** 3 * a2y ** 2) / (16 * (q + 1) ** 6))
                            - ((3 * (23 * q + 22) * q ** 3 * a2z ** 2) / (16 * (q + 1) ** 6))
                            - (((128 * q ** 5 + 181 * q ** 4 - 88 * q ** 3 + 81 * q ** 2 + 544 * q + 312) * q) / (
                                64 * (q + 1) ** 8))
                    )
                    - (((312 * q ** 5 + 544 * q ** 4 + 81 * q ** 3 - 88 * q ** 2 + 181 * q + 128) * q ** 2 * a2z) / (
                        64 * (q + 1) ** 8))
                    - ((3 * (5 * q + 12) * q * a1z ** 3) / (16 * (q + 1) ** 6)) # typo in the paper
            )

    return dh_dr
