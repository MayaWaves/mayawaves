import h5py
import numpy as np
import warnings
from enum import Enum
from functools import total_ordering


class CompactObject:
    """Class containing all information pertaining to a compact object that is part of a coalescence."""

    @total_ordering
    class Column(Enum):
        """Enum for easily referencing columns of desired data."""

        def __new__(cls, *args, **kwds):
            value = len(cls.__members__) + 1
            obj = object.__new__(cls)
            obj._value_ = value
            return obj

        def __init__(self, header_text):
            self.header_text = header_text

        ITT = "itt"
        TIME = "time"
        X = "x"
        Y = "y"
        Z = "z"
        VX = "vx"
        VY = "vy"
        VZ = "vz"
        AX = "ax"
        AY = "ay"
        AZ = "az"
        SX = "Sx"
        SY = "Sy"
        SZ = "Sz"
        PX = "Px"
        PY = "Py"
        PZ = "Pz"
        MIN_RADIUS = "min radius"
        MAX_RADIUS = "max radius"
        MEAN_RADIUS = "mean radius"
        QUADRUPOLE_XX = "quadrupole_xx"
        QUADRUPOLE_XY = "quadrupole_xy"
        QUADRUPOLE_XZ = "quadrupole_xz"
        QUADRUPOLE_YY = "quadrupole_yy"
        QUADRUPOLE_YZ = "quadrupole_yz"
        QUADRUPOLE_ZZ = "quadrupole_zz"
        MIN_X = "min x"
        MAX_X = "max x"
        MIN_Y = "min y"
        MAX_Y = "max y"
        MIN_Z = "min z"
        MAX_Z = "max z"
        XY_PLANE_CIRCUMFERENCE = "xy-plane circumference"
        XZ_PLANE_CIRCUMFERENCE = "xz-plane circumference"
        YZ_PLANE_CIRCUMFERENCE = "yz-plane circumference"
        RATIO_OF_XZ_XY_PLANE_CIRCUMFERENCES = "ratio of xz/xy-plane circumferences"
        RATIO_OF_YZ_XY_PLANE_CIRCUMFERENCES = "ratio of yz/xy-plane circumferences"
        AREA = "area"
        M_IRREDUCIBLE = "m_irreducible"
        AREAL_RADIUS = "areal radius"
        EXPANSION_THETA_L = "expansion Theta_(l)"
        INNER_EXPANSION_THETA_N = "inner expansion Theta_(n)"
        PRODUCT_OF_THE_EXPANSIONS = "product of the expansions"
        MEAN_CURVATURE = "mean curvature"
        GRADIENT_OF_THE_AREAL_RADIUS = "gradient of the areal radius"
        GRADIENT_OF_THE_EXPANSION_THETA_L = "gradient of the expansion Theta_(l)"
        GRADIENT_OF_THE_INNER_EXPANSION_THETA_N = "gradient of the inner expansion Theta_(n)"
        GRADIENT_OF_THE_PRODUCT_OF_THE_EXPANSIONS = "gradient of the product of the expansions"
        GRADIENT_OF_THE_MEAN_CURVATURE = "gradient of the mean curvature"
        MINIMUM_OF_THE_MEAN_CURVATURE = "minimum  of the mean curvature"
        MAXIMUM_OF_THE_MEAN_CURVATURE = "maximum  of the mean curvature"
        INTEGRAL_OF_THE_MEAN_CURVATURE = "integral of the mean curvature"
        EQUATORIAL_CIRCUMFERENCE = "qlm_equatorial_circumference"
        POLAR_CIRCUMFERENCE_0 = "qlm_polar_circumference_0"
        POLAR_CIRCUMFERENCE_PI_2 = "qlm_polar_circumference_pi_2"
        QLM_AREA = 'qlm_area'
        SPIN_GUESS = "qlm_spin_guess"
        MASS_GUESS = "qlm_mass_guess"
        KILLING_EIGENVALUE_REAL = "qlm_killing_eigenvalue_re"
        KILLING_EIGENVALUE_IMAG = "qlm_killing_eigenvalue_im"
        SPIN_MAGNITUDE = "qlm_spin"
        NPSPIN = "qlm_npspin"
        WSSPIN = "qlm_wsspin"
        SPIN_FROM_PHI_COORDINATE_VECTOR = "qlm_cvspin"
        HORIZON_MASS = "qlm_mass"
        ADM_ENERGY = "qlm_adm_energy"
        ADM_MOMENTUM_X = "qlm_adm_momentum_x"
        ADM_MOMENTUM_Y = "qlm_adm_momentum_y"
        ADM_MOMENTUM_Z = "qlm_adm_momentum_z"
        ADM_ANGULAR_MOMENTUM_X = "qlm_adm_angular_momentum_x"
        ADM_ANGULAR_MOMENTUM_Y = "qlm_adm_angular_momentum_y"
        ADM_ANGULAR_MOMENTUM_Z = "qlm_adm_angular_momentum_z"
        WEINBERG_ENERGY = "qlm_w_energy"
        WEINBERG_MOMENTUM_X = "qlm_w_momentum_x"
        WEINBERG_MOMENTUM_Y = "qlm_w_momentum_y"
        WEINBERG_MOMENTUM_Z = "qlm_w_momentum_z"
        WEINBERG_ANGULAR_MOMENTUM_X = "qlm_w_angular_momentum_x"
        WEINBERG_ANGULAR_MOMENTUM_Y = "qlm_w_angular_momentum_y"
        WEINBERG_ANGULAR_MOMENTUM_Z = "qlm_w_angular_momentum_z"

        def __lt__(self, other):
            if self.__class__ is other.__class__:
                return self.header_text < other.header_text
            return NotImplemented

    def __init__(self, data_array: h5py.Dataset, header_list: list, identifier: int, object_type: str,
                 initial_irreducible_mass: float, initial_horizon_mass: float, initial_dimensionless_spin: np.ndarray,
                 initial_dimensional_spin: np.ndarray):
        self.__object_type = object_type
        self.__identifier = identifier
        self.__data_array = data_array
        self.__header_list = header_list

        if initial_irreducible_mass is None or np.isnan(initial_irreducible_mass):
            self.__initial_irreducible_mass = None
        else:
            self.__initial_irreducible_mass = initial_irreducible_mass

        if initial_horizon_mass is None or np.isnan(initial_horizon_mass):
            self.__initial_horizon_mass = None
        else:
            self.__initial_horizon_mass = initial_horizon_mass

        if initial_dimensionless_spin is None or np.any(np.isnan(initial_dimensionless_spin)):
            self.__initial_dimensionless_spin = None
        else:
            self.__initial_dimensionless_spin = initial_dimensionless_spin

        if initial_dimensional_spin is None or np.any(np.isnan(initial_dimensional_spin)):
            self.__initial_dimensional_spin = None
        else:
            self.__initial_dimensional_spin = initial_dimensional_spin

    def get_data_from_columns(self, column_names: list) -> np.ndarray:
        """Get the data in the requested columns (properties).

        Given a list of columns, returns the data at iterations where all columns have values.

        Args:
            column_names (list): list of desired columns, in the form of enums

        Returns:
            np.ndarray: the data at the requested columns

        """

        try:
            columns = [self.__header_list.index(column.header_text) for column in column_names]
        except ValueError:
            return None

        temp_data_array = self.__data_array[()]
        nan_columns = np.isnan(temp_data_array[:, columns]).any(axis=1)
        if len(columns) == 1:
            data = temp_data_array[~nan_columns][:, columns[0]]
        else:
            data = temp_data_array[~nan_columns][:, columns]
        return data

    @property
    def available_data_columns(self) -> list:
        """All the Columns included for this compact object."""
        return [column for column in CompactObject.Column if column.header_text in self.__header_list]

    @property
    def initial_dimensionless_spin(self) -> np.ndarray:
        """The dimensionless spin (:math:`\pmb{a}` or :math:`\pmb{\chi} = \pmb{J}/M^2`) at the beginning of the
        simulation."""
        if self.__initial_dimensionless_spin is not None:
            return self.__initial_dimensionless_spin
        else:
            time, dimensionless_spin_vector = self.dimensionless_spin_vector
            if dimensionless_spin_vector is not None and len(dimensionless_spin_vector) > 0:
                return dimensionless_spin_vector[0]
            else:
                return None

    @property
    def initial_dimensional_spin(self) -> np.ndarray:
        """The dimensional spin (:math:`\pmb{S}` or :math:`\pmb{J}`) at the beginning of the simulation."""
        if self.__initial_dimensional_spin is not None:
            return self.__initial_dimensional_spin
        else:
            time, dimensional_spin_vector = self.dimensional_spin_vector
            if dimensional_spin_vector is not None and len(dimensional_spin_vector) > 0:
                return dimensional_spin_vector[0]
            else:
                return None

    @property
    def initial_irreducible_mass(self) -> float:
        """The irreducible mass at the beginning of the simulation."""
        if self.__initial_irreducible_mass is not None:
            return self.__initial_irreducible_mass
        else:
            time, irreducible_mass = self.irreducible_mass
            if irreducible_mass is not None and len(irreducible_mass) > 0:
                return irreducible_mass[0]
            else:
                return None

    @property
    def initial_horizon_mass(self) -> float:
        """The horizon mass at the beginning of the simulation."""
        if self.__initial_horizon_mass is not None:
            return self.__initial_horizon_mass
        else:
            time, horizon_mass = self.horizon_mass
            if horizon_mass is not None and len(horizon_mass) > 0:
                return horizon_mass[0]
            else:
                return None

    @property
    def final_dimensionless_spin(self) -> np.ndarray:
        """The last available dimensionless spin (:math:`\pmb{a}` or :math:`\pmb{\chi} = \pmb{J}/M^2`) data"""
        time, dimensionless_spin = self.dimensionless_spin_vector
        if time is None or len(time) == 0:
            return None
        return dimensionless_spin[-1]

    @property
    def final_dimensional_spin(self) -> np.ndarray:
        """The last available dimensional spin (:math:`\pmb{S}` or :math:`\pmb{J}`) data"""
        time, dimensional_spin = self.dimensional_spin_vector
        if time is None or len(time) == 0:
            return None
        return dimensional_spin[-1]

    @property
    def final_irreducible_mass(self) -> float:
        """The last available irreducible mass data"""
        time, irreducible_mass = self.irreducible_mass
        if time is None or len(time) == 0:
            return None
        return irreducible_mass[-1]

    @property
    def final_horizon_mass(self) -> float:
        """The last available horizon mass data"""
        time, horizon_mass = self.horizon_mass
        if time is None or len(time) == 0:
            return None
        return horizon_mass[-1]

    @property
    def position_vector(self) -> tuple:
        """The time and position vector over time."""
        position_data = self.get_data_from_columns([self.Column.TIME, self.Column.X, self.Column.Y, self.Column.Z])

        if position_data is None:
            return None, None

        time = position_data[:, 0]
        position_vector = position_data[:, 1:]

        return time, position_vector

    @property
    def velocity_vector(self) -> tuple:
        """The time and velocity vector over time."""
        velocity_data = self.get_data_from_columns([self.Column.TIME, self.Column.VX, self.Column.VY, self.Column.VZ])

        if velocity_data is None:
            position_data = self.get_data_from_columns([self.Column.TIME, self.Column.X, self.Column.Y, self.Column.Z])
            if position_data.shape[0] > 0:
                time = position_data[:, 0]
                poisition_vector = position_data[:, 1:]
                velocity_vector = np.gradient(poisition_vector, axis=1) / np.gradient(time)[:, None]
                return time, velocity_vector

        time = velocity_data[:, 0]
        velocity_vector = velocity_data[:, 1:]

        if len(time) == 0:
            position_data = self.get_data_from_columns([self.Column.TIME, self.Column.X, self.Column.Y, self.Column.Z])
            if position_data is None:
                return None, None

            if position_data.shape[0] > 0:
                time = position_data[:, 0]
                poisition_vector = position_data[:, 1:]
                velocity_vector = np.gradient(poisition_vector, axis=1) / np.gradient(time)[:, None]

        return time, velocity_vector

    @property
    def momentum_vector(self) -> tuple:
        """The time and momentum vector over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.PX, self.Column.PY, self.Column.PZ])

        if data is None:
            return None, None

        time = data[:, 0]
        momentum_vector = data[:, 1:]

        return time, momentum_vector

    @property
    def dimensional_spin_vector(self) -> tuple:
        """The time and dimensional spin vector (:math:`\pmb{S}` or :math:`\pmb{J}`) over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.SX, self.Column.SY,
                                           self.Column.SZ])

        if data is None:
            return None, None

        time = data[:, 0]
        dimensional_spin_vector = data[:, 1:]

        return time, dimensional_spin_vector

    @property
    def dimensionless_spin_vector(self) -> tuple:
        """The time and dimensionless spin vector (:math:`\pmb{a}` or :math:`\pmb{\chi} = \pmb{J}/M^2`) over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.SX, self.Column.SY,
                                           self.Column.SZ, self.Column.M_IRREDUCIBLE])

        if data is None:
            return None, None

        time = data[:, 0]
        dimensional_spin_vector = data[:, 1:4]
        irreducible_mass = data[:, 4]

        spin_magnitude = np.linalg.norm(dimensional_spin_vector, axis=1)
        horizon_mass = np.sqrt(irreducible_mass ** 2 + spin_magnitude ** 2 / (4 * irreducible_mass ** 2))

        horizon_mass = horizon_mass.reshape((len(horizon_mass), 1))

        dimensionless_spin_vector = np.divide(dimensional_spin_vector, horizon_mass * horizon_mass)

        return time, dimensionless_spin_vector

    @property
    def horizon_mass(self) -> tuple:
        """The time and horizon mass over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.SX, self.Column.SY,
                                           self.Column.SZ, self.Column.M_IRREDUCIBLE])

        if data is None:
            return None, None

        time = data[:, 0]
        dimensional_spin_vector = data[:, 1:4]
        irreducible_mass = data[:, 4]

        spin_magnitude = np.linalg.norm(dimensional_spin_vector, axis=1)
        horizon_mass = np.sqrt(irreducible_mass ** 2 + spin_magnitude ** 2 / (4 * irreducible_mass ** 2))

        return time, horizon_mass

    @property
    def apparent_horizon_area(self) -> tuple:
        """The time and the area of the apparent horizon over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.AREA])

        if data is None:
            return None, None

        time = data[:, 0]
        apparent_horizon_area = data[:, 1]

        return time, apparent_horizon_area

    @property
    def irreducible_mass(self) -> tuple:
        """The time and irreducible mass over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.M_IRREDUCIBLE])

        if data is None:
            return None, None

        time = data[:, 0]
        irreducible_mass = data[:, 1]

        return time, irreducible_mass

    @property
    def apparent_horizon_mean_curvature(self) -> tuple:
        """The time and mean curvature of the apparent horizon over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.MEAN_CURVATURE])

        if data is None:
            return None, None

        time = data[:, 0]
        apparent_horizon_mean_curvature = data[:, 1]

        return time, apparent_horizon_mean_curvature

    @property
    def apparent_horizon_areal_radius(self) -> tuple:
        """The time and areal radius of the apparent horizon over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.AREAL_RADIUS])

        if data is None:
            return None, None

        time = data[:, 0]
        apparent_horizon_areal_radius = data[:, 1]

        return time, apparent_horizon_areal_radius

    @property
    def apparent_horizon_expansion_theta_l(self) -> tuple:
        """The time and the expansion Theta_(l) over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.EXPANSION_THETA_L])

        if data is None:
            return None, None

        time = data[:, 0]
        apparent_horizon_expansion_theta_l = data[:, 1]

        return time, apparent_horizon_expansion_theta_l

    @property
    def apparent_horizon_inner_expansion_theta_n(self) -> tuple:
        """The time and the inner expansion Theta_(n) on the apparent horizon over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.INNER_EXPANSION_THETA_N])

        if data is None:
            return None, None

        time = data[:, 0]
        apparent_horizon_inner_expansion_theta_n = data[:, 1]

        return time, apparent_horizon_inner_expansion_theta_n

    @property
    def apparent_horizon_minimum_radius(self) -> tuple:
        """The time and apparent horizon minimum radius over time."""

        data = self.get_data_from_columns([self.Column.TIME, self.Column.MIN_RADIUS])

        if data is None:
            return None, None

        time = data[:, 0]
        apparent_horizon_minimum_radius = data[:, 1]

        return time, apparent_horizon_minimum_radius

    @property
    def apparent_horizon_maximum_radius(self) -> tuple:
        """The time and maximum apparent horizon radius over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.MAX_RADIUS])

        if data is None:
            return None, None

        time = data[:, 0]
        apparent_horizon_maximum_radius = data[:, 1]

        return time, apparent_horizon_maximum_radius

    @property
    def apparent_horizon_circumferences(self) -> tuple:
        """The time and apparent horizon circumferences in the XY, XZ, and YZ planes respectively over time."""
        circumferences_data = self.get_data_from_columns(
            [self.Column.TIME, self.Column.XY_PLANE_CIRCUMFERENCE,
             self.Column.XZ_PLANE_CIRCUMFERENCE,
             self.Column.YZ_PLANE_CIRCUMFERENCE])

        if circumferences_data is None:
            return None, None, None, None

        time = circumferences_data[:, 0]
        xy_plane_circumference = circumferences_data[:, 1]
        xz_plane_circumference = circumferences_data[:, 2]
        yz_plane_circumference = circumferences_data[:, 3]

        return time, xy_plane_circumference, xz_plane_circumference, yz_plane_circumference

    @property
    def apparent_horizon_mean_radius(self) -> tuple:
        """The time and mean apparent horizon radius over time."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.MEAN_RADIUS])

        if data is None:
            return None, None

        time = data[:, 0]
        apparent_horizon_mean_radius = data[:, 1]

        return time, apparent_horizon_mean_radius

    @property
    def apparent_horizon_quadrupoles(self) -> tuple:
        """The time and apparent horizon quadrupoles over time.

        Returns a tuple including the time, xx quadrupole, xy quadrupole, xz quadrupole, yy quadrupole,
        yz quadrupole, zz quadrupole.
        """
        quadrupoles_data = self.get_data_from_columns(
            [self.Column.TIME, self.Column.QUADRUPOLE_XX, self.Column.QUADRUPOLE_XY,
             self.Column.QUADRUPOLE_XZ, self.Column.QUADRUPOLE_YY, self.Column.QUADRUPOLE_YZ,
             self.Column.QUADRUPOLE_ZZ])

        if quadrupoles_data is None:
            return None, None, None, None, None, None, None

        time = quadrupoles_data[:, 0]
        quadrupole_xx = quadrupoles_data[:, 1]
        quadrupole_xy = quadrupoles_data[:, 2]
        quadrupole_xz = quadrupoles_data[:, 3]
        quadrupole_yy = quadrupoles_data[:, 4]
        quadrupole_yz = quadrupoles_data[:, 5]
        quadrupole_zz = quadrupoles_data[:, 6]

        return time, quadrupole_xx, quadrupole_xy, quadrupole_xz, \
            quadrupole_yy, quadrupole_yz, quadrupole_zz

    @property
    def last_available_spin_data_time(self) -> float:
        """The last time at which spin data is available."""
        data = self.get_data_from_columns([self.Column.TIME, self.Column.SX])

        if data is None:
            return 0

        time_data = data[:, 0]

        if len(time_data) == 0:
            return 0

        return time_data[-1]

    @property
    def data_array(self) -> np.ndarray:
        """Function to be used internally to return the full data array for this compact object."""
        return self.__data_array[()]

    def dimensional_spin_at_time(self, desired_time: float) -> np.ndarray:
        """Dimensional spin vector (:math:`\pmb{S}` or :math:`\pmb{J}`) at a given time.

        Args:
            desired_time (float): the time at which to return the spin

        Returns:
            np.ndarray: the dimensional spin vector at the desired time

        """

        spin_data = self.get_data_from_columns([self.Column.TIME, self.Column.SX, self.Column.SY,
                                                self.Column.SZ])

        if spin_data is None or len(spin_data) == 0:
            warnings.warn("Spin data not available at that time.")
            return None

        time = spin_data[:, 0]

        if desired_time > time[-1] or desired_time < time[0]:
            warnings.warn("Spin data not available at that time.")
            return None

        cut_index = np.argmax(time > desired_time)
        if time[cut_index] > (desired_time + 1):
            warnings.warn("Closest spin data is more than 1M away.")

        dimensional_spin = spin_data[cut_index, 1:]
        return dimensional_spin

    def irreducible_mass_at_time(self, desired_time: float) -> float:
        """Irreducible mass at a given time.

        Args:
            desired_time: the time at which to return the mass

        Returns:
            float: the irreducible mass at the desired time

        """
        irreducible_mass_data = self.get_data_from_columns([self.Column.TIME, self.Column.M_IRREDUCIBLE])

        if irreducible_mass_data is None or len(irreducible_mass_data) == 0:
            warnings.warn("Can't obtain mass data at that time. Returning the initial irreducible mass.")
            return self.initial_irreducible_mass

        time = irreducible_mass_data[:, 0]

        cut_index = np.argmax(time > desired_time)
        if time[cut_index] > (desired_time + 1):
            warnings.warn("Can't obtain mass data at that time")
            return None

        irreducible_mass = irreducible_mass_data[cut_index, 1]
        return irreducible_mass

    def horizon_mass_at_time(self, desired_time: float) -> float:
        """Horizon mass at a given time.

        Args:
            desired_time: the time at which to return the mass

        Returns:
            float: the horizon mass at the desired time

        """
        irreducible_mass = self.irreducible_mass_at_time(desired_time)
        dimensional_spin = self.dimensional_spin_at_time(desired_time)

        if irreducible_mass is None:
            return None

        if dimensional_spin is None:
            return None

        spin_magnitude = np.linalg.norm(dimensional_spin)

        horizon_mass = np.sqrt(irreducible_mass ** 2 + spin_magnitude ** 2 / (4 * irreducible_mass ** 2))
        return horizon_mass

    def dimensionless_spin_at_time(self, desired_time: float) -> np.ndarray:
        """Dimensionless spin vector (:math:`\pmb{a}` or :math:`\pmb{\chi} = \pmb{J}/M^2`) at a giventime.

        Args:
            desired_time (float): the time at which to return the spin

        Returns:
            np.ndarray: the dimensionless spin vector at the desired time
        """
        dimensional_spin = self.dimensional_spin_at_time(desired_time)
        horizon_mass = self.horizon_mass_at_time(desired_time)

        if dimensional_spin is None:
            return None

        if horizon_mass is None:
            return None

        dimensionless_spin = dimensional_spin / (horizon_mass * horizon_mass)
        return dimensionless_spin
