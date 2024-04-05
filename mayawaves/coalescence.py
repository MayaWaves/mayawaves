import numpy as np
import h5py
import warnings
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from mayawaves.compactobject import CompactObject
from mayawaves.radiation import RadiationBundle
from mayawaves.utils import postnewtonianutils as pn
import pandas as pd

class Coalescence:
    """Fundamental class for interacting with all data about a simulation of a compact object coalescence."""

    def __init__(self, h5_filepath: str):
        self.__compact_objects = []
        self.__h5_filepath = h5_filepath
        self.__h5_file = h5py.File(h5_filepath, 'r')
        self.__primary_compact_object_index = None
        self.__secondary_compact_object_index = None
        self.__final_compact_object_index = None

        # Create CompactObjects
        object_numbers = [int(object_group[7:]) for object_group in self.__h5_file["compact_object"].keys()]
        for object_num in object_numbers:
            if object_num == 0 or object_num == 1:
                irreducible_mass = self.__h5_file.attrs["irreducible mass %d" % object_num]
                horizon_mass = self.__h5_file.attrs["horizon mass %d" % object_num]
                dimensionless_spin = self.__h5_file.attrs["dimensionless spin %d" % object_num]
                dimensional_spin = self.__h5_file.attrs["dimensional spin %d" % object_num]
            else:
                irreducible_mass = None
                horizon_mass = None
                dimensionless_spin = None
                dimensional_spin = None

            self.__compact_objects.append(
                CompactObject(self.__h5_file["compact_object"]["object=%d" % object_num],
                              self.__h5_file["compact_object"]["object=%d" % object_num].attrs["header"].tolist(),
                              object_num,
                              "BH",
                              irreducible_mass,
                              horizon_mass,
                              dimensionless_spin,
                              dimensional_spin)
            )

        if len(self.__compact_objects) > 0:
            self.__primary_compact_object_index = 0
        if len(self.__compact_objects) > 1:
            self.__secondary_compact_object_index = 1
        if len(self.__compact_objects) > 2:
            self.__final_compact_object_index = 2

        if 'radiative' in self.__h5_file:
            self.__radiation_mode_bundle = RadiationBundle.create_radiation_bundle(self.__h5_file["radiative"])

    @property
    def name(self) -> str:
        """The name of the coalescence. This is what the simulation was named when it was performed."""
        return self.__h5_file.attrs["name"]

    @property
    def catalog_id(self) -> str:
        """The id of the simulation if it exists within a catalog."""
        if 'catalog id' in self.__h5_file.attrs:
            return self.__h5_file.attrs['catalog id']
        print('This simulation is not part of a catalog and does not have a catalog id.')
        return None

    @property
    def parameter_files(self) -> dict:
        """Dictionary containing the text of the parameter files (.rpar and .par) used to create the simulation

        The keys are "par" and "rpar"

        """
        if 'parfile' not in self.__h5_file:
            return {}
        parfile_dict = {}
        parfile_group = self.__h5_file["parfile"]
        if 'rpar_content' in parfile_group.attrs:
            parfile_dict['rpar'] = parfile_group.attrs['rpar_content']
        if 'par_content' in parfile_group.attrs:
            parfile_dict['par'] = parfile_group.attrs['par_content']
        return parfile_dict

    @property
    def h5_filepath(self) -> str:
        """The path to the h5 file containing the coalescence data."""
        return self.__h5_filepath

    @property
    def primary_compact_object(self) -> CompactObject:
        """A reference to the primary compact object. This is the larger of the two initial compact objects."""
        if self.__primary_compact_object_index is not None:
            return self.compact_objects[self.__primary_compact_object_index]
        else:
            return None

    @property
    def secondary_compact_object(self) -> CompactObject:
        """A reference to the secondary compact object. This is the smaller of the two initial compact objects."""
        if self.__secondary_compact_object_index is not None:
            return self.compact_objects[self.__secondary_compact_object_index]
        else:
            return None

    @property
    def final_compact_object(self) -> CompactObject:
        """A reference to the final compact object. This is the remnant that is formed after the merger."""
        if self.__final_compact_object_index is not None:
            return self.compact_objects[self.__final_compact_object_index]
        else:
            return None

    @property
    def compact_objects(self) -> list:
        """List of all initial and final compact objects."""
        return self.__compact_objects

    @property
    def radiationbundle(self) -> RadiationBundle:
        """A reference to the RadiationBundle object."""
        return self.__radiation_mode_bundle

    @property
    def mass_ratio(self) -> float:
        """Mass ratio of the coalescence. Always greater than 1. Defined as the horizon mass of the larger compact
        object divided by the horizon mass of the smaller compact object. :math:`q = m_1/m_2 > 1`"""
        return self.__h5_file.attrs["mass ratio"]

    @property
    def symmetric_mass_ratio(self) -> float:
        """Symmetric mass ratio of the coalescence. Computed using the horizon masses as
        :math:`\eta = m_1 m_2/(m_1+m_2)^2`"""
        mass0 = self.primary_compact_object.initial_horizon_mass
        mass1 = self.secondary_compact_object.initial_horizon_mass

        if mass0 is None or mass1 is None:
            return None

        eta = mass0 * mass1 / ((mass0 + mass1) ** 2)
        eta = np.round(eta, decimals=6)
        return eta

    @property
    def spin_configuration(self) -> str:
        """Spin configuration. Can be non-spinning, aligned-spins, or precessing. Non-spinning means neither black hole
        has a spin. Aligned-spins means the spin of each black hole is either aligned or anti-aligned with the orbital
        angular momentum. Precessing means at least one compact object has a spin not aligned or anti-aligned with the
        orbital angular momentum."""
        return self.__h5_file.attrs["spin configuration"]

    @property
    def initial_separation(self) -> float:
        """Initial separation of the coalesence. Computed as the coordinate separation between the centers of the
        primary and secondary compact objects."""
        return self.__h5_file.attrs["initial separation"]

    @property
    def initial_orbital_frequency(self) -> float:
        """Inital orbital frequency of the coalescence. Computed from the separation vector between the primary and
        secondary compact objects."""
        return self.__h5_file.attrs["initial orbital frequency"]

    @property
    def separation_vector(self) -> tuple:
        """Coordinate separation vector between the primary and secondary compact objects. Returns the time and the
        separation vector in a tuple. Points from the primary object to the secondary object."""
        time_0, position_0 = self.primary_compact_object.position_vector
        time_1, position_1 = self.secondary_compact_object.position_vector

        time, time_indices_0, time_indices_1 = np.intersect1d(time_0, time_1, assume_unique=True, return_indices=True)
        position_0 = position_0[time_indices_0]
        position_1 = position_1[time_indices_1]

        separation_vector = position_1 - position_0

        return time, separation_vector

    @property
    def orbital_angular_momentum_unit_vector(self) -> tuple:
        """Time and orbital angular momentum unit vector over time.

        Unit vector for the orbital angular momentum computed from the coordinate positions of the primary and secondary
        compact objects.
        """
        time, separation_vector = self.separation_vector

        nhat = separation_vector / np.linalg.norm(separation_vector, axis=1).reshape(separation_vector.shape[0], 1)
        dnhat_dt = np.gradient(nhat, axis=0) / (np.gradient(time).reshape(nhat.shape[0], 1))
        orbital_frequency = np.cross(nhat, dnhat_dt, axis=1)
        mag_orbital_frequency = np.linalg.norm(orbital_frequency, axis=1)

        orbital_angular_momentum_unit_vector = orbital_frequency / mag_orbital_frequency.reshape(
            orbital_frequency.shape[0], 1)

        return time, orbital_angular_momentum_unit_vector

    @property
    def orbital_phase_in_xy_plane(self) -> tuple:
        """Time and orbital phase over time in the xy plane computed from the coordinate positions."""
        time, separation_vector = self.separation_vector

        phase = np.unwrap(np.arctan2(separation_vector[:, 1], separation_vector[:, 0]))

        return time, phase

    @property
    def orbital_frequency(self) -> tuple:
        """Time and orbital frequency over time.

        Compute and return the orbital frequency as a function of time, calculated from the coordinate separation
        between the primary and secondary compact objects.
        """
        time, separation_vector = self.separation_vector

        nhat = separation_vector / np.linalg.norm(separation_vector, axis=1).reshape(separation_vector.shape[0], 1)
        dnhat_dt = np.gradient(nhat, axis=0) / (np.gradient(time).reshape(nhat.shape[0], 1))
        orbital_frequency = np.cross(nhat, dnhat_dt, axis=1)
        mag_orbital_frequency = np.linalg.norm(orbital_frequency, axis=1)

        return time, mag_orbital_frequency

    @property
    def average_run_speed(self) -> float:
        """Average speed of the simulation in M/hr."""
        if "runstats" not in self.__h5_file:
            warnings.warn('Runstats information is not available for this coalescence.')
            return None
        speed_column = self.__h5_file["runstats"].attrs["header"].tolist().index("speed (hours^-1)")
        return float(np.mean(self.__h5_file["runstats"][:, speed_column], axis=0))

    @property
    def l_max(self) -> int:
        """Maximum l mode included."""
        return self.__radiation_mode_bundle.l_max

    @property
    def included_modes(self) -> list:
        """:math:`\Psi_4` is decomposed into spherical harmonics labeled by (l, m). This provides a list of all (l,
        m) modes included."""
        return self.__radiation_mode_bundle.included_modes

    @property
    def included_extraction_radii(self) -> list:
        """List of all extraction radii included."""
        return self.__radiation_mode_bundle.included_radii

    @property
    def object_numbers(self) -> list:
        """List of all object ids included."""
        object_numbers = [int(object_group[7:]) for object_group in self.__h5_file["compact_object"].keys()]
        return object_numbers

    @property
    def runstats_data(self) -> np.ndarray:
        """All runstats data stored as a pandas dataset."""
        if "runstats" in self.__h5_file:
            column_names = self.__h5_file["runstats"].attrs["header"]
            runstats_dataframe = pd.DataFrame(data=self.__h5_file["runstats"][()], columns=column_names)
            return runstats_dataframe
        else:
            return None

    @property
    def psi4_source(self) -> str:
        """Source of the :math:`\Psi_4` data. Could by ascii or h5 forms of WeylScal or Multipole."""
        source = self.__h5_file["radiative"]["psi4"].attrs["source"]
        # older versions used "Ylm_WEYLSCAL4" ascii files, newer ones are more explicit
        if source == "Ylm_WEYLSCAL4":
            source = "YLM_WEYLSCAL4_ASC"
        return source

    @property
    def merge_time(self) -> float:
        """Time at which the coordinate separation goes below 1e-2"""
        time, separation_vector = self.separation_vector
        separation_mag = np.linalg.norm(separation_vector, axis=1)
        return time[np.argmax(separation_mag < 1e-2)]

    @property
    def radiation_frame(self) -> str:
        """The current frame of the radiation data"""
        return self.radiationbundle.frame

    def set_radiation_frame(self, center_of_mass_corrected: bool = False):
        """Set the frame for the radiation mode decomposition.

        Options are center of mass drift corrected or the original, raw frame

        Args:
            center_of_mass_corrected (:obj:`bool`, optional): Whether to correct for center of mass drift. Default False. If false, the frame is set back to the original, raw frame.
        """
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            warnings.warn('Python version too recent to be compatible with Scri package. Unable to change the radiation frame.')
            raise ImportError('Unable to change the radiation frame. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

        from mayawaves.radiation import Frame

        if center_of_mass_corrected:
            com_time, center_of_mass = self.center_of_mass
            if com_time is None or center_of_mass is None:
                warnings.warn('Unable to compute center of mass and therefore unable to correct for it. Staying in the raw frame.')
                return
            self.radiationbundle.set_frame(Frame.COM_CORRECTED, time=com_time, center_of_mass=center_of_mass)
        else:
            self.radiationbundle.set_frame(Frame.RAW)

    @property
    def center_of_mass(self) -> tuple:
        """Center of mass of the primary and secondary compact object. Computed with the coordinate positions and
        horizon mass."""
        time_pos_0, position_0 = self.primary_compact_object.position_vector
        time_pos_1, position_1 = self.secondary_compact_object.position_vector

        if time_pos_0 is None or time_pos_1 is None:
            return None, None

        time_pos, time_indices_0, time_indices_1 = np.intersect1d(time_pos_0, time_pos_1, assume_unique=True,
                                                                  return_indices=True)
        position_0 = position_0[time_indices_0]
        position_1 = position_1[time_indices_1]

        time_mass_0, horizon_mass_0 = self.primary_compact_object.horizon_mass
        time_mass_1, horizon_mass_1 = self.secondary_compact_object.horizon_mass

        if time_mass_0 is None or time_mass_1 is None:
            initial_horizon_mass_0 = self.primary_compact_object.initial_horizon_mass
            initial_horizon_mass_1 = self.secondary_compact_object.initial_horizon_mass

            if initial_horizon_mass_0 is None or initial_horizon_mass_1 is None:
                return None, None

            center_of_mass = position_0 * initial_horizon_mass_0 + position_1 * initial_horizon_mass_1

            return time_pos, center_of_mass

        time_mass, time_indices_0, time_indices_1 = np.intersect1d(time_mass_0, time_mass_1, assume_unique=True,
                                                                   return_indices=True)
        horizon_mass_0 = horizon_mass_0[time_indices_0]
        horizon_mass_1 = horizon_mass_1[time_indices_1]

        if len(time_mass) > 0 and np.isclose(time_mass[-1], time_pos[-1], atol=10):
            time, time_indices_pos, time_indices_mass = np.intersect1d(time_pos, time_mass, assume_unique=True,
                                                                       return_indices=True)
            position_0 = position_0[time_indices_pos]
            position_1 = position_1[time_indices_pos]
            horizon_mass_0 = horizon_mass_0[time_indices_mass]
            horizon_mass_1 = horizon_mass_1[time_indices_mass]

            center_of_mass = position_0 * horizon_mass_0.reshape(len(horizon_mass_0),
                                                                 1) + position_1 * horizon_mass_1.reshape(
                len(horizon_mass_1), 1)

            return time, center_of_mass

        else:
            initial_horizon_mass_0 = self.primary_compact_object.initial_horizon_mass
            initial_horizon_mass_1 = self.secondary_compact_object.initial_horizon_mass

            if initial_horizon_mass_0 is None or initial_horizon_mass_1 is None:
                return None, None
            
            center_of_mass = position_0 * initial_horizon_mass_0 + position_1 * initial_horizon_mass_1

            return time_pos, center_of_mass

    @property
    def radius_for_extrapolation(self):
        """Radius to use when extrapolating data to infinite radius."""
        return self.radiationbundle.radius_for_extrapolation

    @radius_for_extrapolation.setter
    def radius_for_extrapolation(self, extraction_radius: float):
        """Set the extraction radius to use when extrapolating to infinite radius

        Args:
            extraction_radius (float): if extrapolating from infinite radius, this provides the
                radius to extrapolate from.

        """
        self.radiationbundle.radius_for_extrapolation = extraction_radius

    def recoil_velocity(self, km_per_sec: bool = False) -> np.ndarray:
        """Kick vector of the final object

        Computed from the momentum radiated in gravitational waves.

        Args:
            km_per_sec (:obj:`bool`, optional): whether to report the velocity in km/s. Defaults to reporting as a fraction of c.

        Returns:
            np.ndarray: the recoil velocity

        """
        time, radiated_momentum = self.linear_momentum_radiated()
        total_radiated_momentum = radiated_momentum[-1]
        final_compact_object = self.final_compact_object
        if final_compact_object is None:
            return None
        final_horizon_mass = final_compact_object.final_horizon_mass
        if final_horizon_mass is None:
            return None
        kick_vector = -1 * total_radiated_momentum / final_horizon_mass
        if km_per_sec:
            speed_of_light_km_per_s = 299792.458
            kick_vector = kick_vector * speed_of_light_km_per_s
        return kick_vector

    def recoil_speed(self, km_per_sec: bool = False) -> float:
        """Magnitude of the recoil of the final object.

        Computed from the momentum radiated in gravitational waves.

        Args:
            km_per_sec (:obj:`bool`, optional): whether to report the speed in km/s. Defaults to reporting as a fraction of c.

        Returns:
            float: the speed of the recoil

        """
        kick_vector = self.recoil_velocity(km_per_sec=km_per_sec)
        if kick_vector is None:
            return None
        kick_magnitude = np.linalg.norm(kick_vector)
        return kick_magnitude

    def compact_object_by_id(self, id: int) -> CompactObject:
        """Compact object associated with a given object id.

            Args:
                id (int): ID for the desired object

            Returns:
                CompactObject: Compact object associated with the provided id

            """
        return self.compact_objects[id]

    def separation_at_time(self, desired_time: float) -> float:
        """Separation at a given time.

            Computes and returns the coordinate separation between the primary and secondary compact object centers at
            the desired time. Points from the primary object to the secondary object.

            Args:
                 desired_time (float): time at which to return the separation

            Returns:
                float: separation at the desired time

        """
        time, separation_vector = self.separation_vector
        separation_magnitude = np.linalg.norm(separation_vector, axis=1)
        if desired_time > time[-1]:
            warnings.warn("Unable to find the separation at that time")
            return None
        if desired_time < time[0]:
            warnings.warn("Returning the separation at the earliest available time: t={t}".format(t=time[0]))
            return separation_magnitude[0]
        return separation_magnitude[np.argmax(time > desired_time)]

    def orbital_frequency_at_time(self, desired_time: float) -> float:
        """Orbital frequency at given time.

        Compute and return the orbital frequency at the desired time, calculated from the coordinate separation between
        the primary and secondary compact objects.

        Args:
            desired_time (float): time at which to return the orbital separation

        Returns:
            float: orbital frequency at the desired time

        """
        time, orbital_frequency = self.orbital_frequency
        if desired_time > time[-1]:
            warnings.warn("Unable to find the orbital frequency at that time")
            return None
        if desired_time < time[0]:
            warnings.warn("Returning the orbital frequency at the earliest available time: t={t}".format(t=time[0]))
            return orbital_frequency[0]

        return orbital_frequency[np.argmax(time > desired_time)]

    def orbital_angular_momentum_unit_vector_at_time(self, desired_time: float) -> np.ndarray:
        """Orbital angular momentum unit vector at desired time.

        Unit vector for the orbital angular momentum computed from the coordinate positions of the primary and secondary
        compact objects.

        Args:
            desired_time (float): time at which to return the orbital separation

        Returns:
            numpy.ndarray: unit vector of the orbital angular momentum at the desired time

        """
        time, orbital_angular_momentum_unit_vector = self.orbital_angular_momentum_unit_vector
        if desired_time > time[-1]:
            warnings.warn("Unable to find the orbital angular momentum unit vector at that time")
            return None
        if desired_time < time[0]:
            warnings.warn(
                "Returning the orbital angular momentum unit vector at the earliest available time: t={t}".format(
                    t=time[0]))
            return orbital_angular_momentum_unit_vector[0]

        return orbital_angular_momentum_unit_vector[np.argmax(time > desired_time)]

    def separation_unit_vector_at_time(self, desired_time: float) -> np.ndarray:
        """Separation unit vector at desired time.

        Unit vector for the separation computed from the coordinate positions of the primary and secondary compact
        objects. Points from the primary object to the secondary object.

        Args:
            desired_time (float): time at which to return the separation unit vector

        Returns:
            numpy.ndarray: unit vector of the separation at the desired time

        """
        time, separation_vector = self.separation_vector

        cut_index = np.argmax(time > desired_time)
        if time[cut_index] > (desired_time + 1):
            raise IOError("Can't obtain separation data at that time")

        nhat = separation_vector / np.linalg.norm(separation_vector, axis=1).reshape(separation_vector.shape[0], 1)
        return nhat[cut_index]

    def _crop_to_three_or_four_orbits(self, start_time: float, time: np.ndarray, orbital_phase: np.ndarray,
                                      separation_magnitude: np.ndarray, data: np.ndarray) -> tuple:
        """Crop a given timeseries to three or four orbits, preferring four.

        Compute and return four orbits worth of a given timeseries, beginning at the specified start time. If there are
        not four orbits, it crops to three orbits. If there aren't three orbits, it returns None. The number of orbits
        is determined by the orbital phase.

        Args:
            start_time (float): time at which to begin crop
            time (numpy.ndarray): time stamps associated with data
            orbital_phase (numpy.ndarray): orbital phase associated with data
            separation_magnitude (numpy.ndarray): magnitude of the separation associated with data
            data (numpy.ndarray): timeseries to be cropped

        Returns:
            tuple: time and data cropped down to four orbits

        """
        merge_time = time[np.argmax(separation_magnitude < 0.1)]

        if merge_time <= 0:
            merge_time = time[-1]

        start_index = np.argmax(time > start_time)

        three_orbits_time = time[np.argmax(orbital_phase > (6 * np.pi + orbital_phase[start_index]))]
        four_orbits_time = time[np.argmax(orbital_phase > (8 * np.pi + orbital_phase[start_index]))]

        end_time = four_orbits_time
        if end_time <= 0 or end_time > merge_time - 50:
            end_time = three_orbits_time
        if end_time <= 0 or end_time > merge_time - 50:
            print("There is not enough data to crop to four orbits")
            return None, None

        end_index = np.argmax(time >= end_time)

        time = time[start_index: end_index]
        data = data[start_index: end_index]

        if len(time) < 1:
            print("There is not enough data to crop to four orbits")
            return None, None

        return time, data

    def _anomaly_from_apsis_times(self, periapsis_times: np.ndarray, apoapsis_times: np.ndarray,
                                  desired_time: float) -> float:
        """Mean anomaly based on periapsis and apoapsis times

        Compute and return the mean anomaly at the desired time using
        :math:`2 \pi * \frac{t - T_{prev}}{T_{next} - T_{prev}}` where :math:`T_{next}' and :math:`T_{prev}' are
        periapsis times.

        Args:
            periapsis_times (numpy.ndarray): times of periapsis
            apoapsis_times (numpy.ndarray): times of apoapsis
            desired_time (float): time at which mean anomaly should be reported

        Returns:
            float: mean anomaly at desired time

        """
        if len(periapsis_times) > 1:
            period = periapsis_times[1] - periapsis_times[0]
        elif len(apoapsis_times) > 1:
            period = apoapsis_times[1] - apoapsis_times[0]
        else:
            warnings.warn("Unable to compute mean anomaly_from_eccentric_timeseries. Setting to -1.")
            mean_anomaly = -1
            return mean_anomaly

        if desired_time >= periapsis_times[0]:
            reversed_periapsis_times = np.flip(periapsis_times)
            previous_periapsis_time = reversed_periapsis_times[np.argmax(reversed_periapsis_times <= desired_time)]
            mean_anomaly = 2 * np.pi * (desired_time - previous_periapsis_time) / period
        else:
            mean_anomaly = 2 * np.pi * (period - (periapsis_times[0] - desired_time)) / period

        return mean_anomaly

    def _anomaly_from_eccentric_timeseries(self, time: np.ndarray, eccentric_timeseries: np.ndarray,
                                           desired_time: float) -> float:
        """Mean anomaly computed from an eccentricity timeseries

        Find the periapsis and apoapsis times based on the provided eccentric timeseries and call
        anomaly_from_apsis_times to compute the mean anomaly at the desired time.

        Args:
            time (numpy.ndarray): time stamps associated with data
            eccentric_timeseries (numpy.ndarray): time series of eccentricity oscillations
            desired_time (float): time at which the mean anomaly should be reported

        Returns:
            float: mean anomaly at desired time

        """
        minima = argrelextrema(eccentric_timeseries, np.less)
        maxima = argrelextrema(eccentric_timeseries, np.greater)

        periapsis_times = time[minima]
        apoapsis_times = time[maxima]

        mean_anomaly = self._anomaly_from_apsis_times(periapsis_times, apoapsis_times, desired_time)

        return mean_anomaly

    def eccentricity_and_mean_anomaly_at_time(self, start_time, desired_time) -> tuple:
        """Eccentricity and mean anomaly at desired time.

        Compute the eccentricity and the mean anomaly using the orbital frequency as described in https://arxiv.org/abs/1810.00036.
        Computes the eccentricity averaged over four orbits. If it is unable to fit four orbits of data successfully, it returns
        an estimate of the eccentricity based on the initial momentum. Mean anomaly is defined as
        :math:`2 \pi * \frac{t - T_{prev}}{T_{next} - T_{prev}}` where :math:`T_{next}' and :math:`T_{prev}' are
        periapsis times.

        Args:
            start_time (float): time from which to begin fitting eccentricity
            desired_time (float): time at which to return the eccentricity

        Returns:
            tuple: averaged eccentricity over first four orbits and the mean anomaly at the desired time

        """

        time, orbital_phase = self.orbital_phase_in_xy_plane
        _, orbital_frequency = self.orbital_frequency
        _, separation_vector = self.separation_vector
        separation_magnitude = np.linalg.norm(separation_vector, axis=1)
        tmax = self.psi4_max_time_for_mode(l=2, m=2)
        eta = self.symmetric_mass_ratio
        mass_ratio = self.mass_ratio

        if self.primary_compact_object is None or self.secondary_compact_object is None:
            return None, None

        primary_dimensionless_spin = self.primary_compact_object.initial_dimensionless_spin
        secondary_dimensionless_spin = self.secondary_compact_object.initial_dimensionless_spin
        primary_dimensional_spin = self.primary_compact_object.initial_dimensional_spin

        if primary_dimensionless_spin is None or secondary_dimensionless_spin is None:
            return None, None

        # rough estimate with initial velocity
        time_momentum, momentum_vector = self.primary_compact_object.momentum_vector
        if momentum_vector is None:
            return -1, -1

        initial_momentum = momentum_vector[0]
        tangential_initial_momentum = initial_momentum[1]
        qc_tangential_momentum = pn.tangential_momentum_from_separation(separation_magnitude[0], mass_ratio,
                                                                        primary_dimensionless_spin,
                                                                        secondary_dimensionless_spin)

        eps = 1 - tangential_initial_momentum / qc_tangential_momentum

        estimated_eccentricity = abs(2 * eps - eps ** 2)

        if estimated_eccentricity > 0.2:
            warnings.warn("Estimated eccentricity is higher than 0.2 so the orbital frequncy fit won't be valid. Returning the eccentricity estimated from the initial momentum.")
            return estimated_eccentricity, -1

        time_inspiral, orbital_frequency_inspiral = self._crop_to_three_or_four_orbits(start_time=start_time, time=time,
                                                                              orbital_phase=orbital_phase,
                                                                              separation_magnitude=separation_magnitude,
                                                                              data=orbital_frequency)

        if time_inspiral is None:
            warnings.warn('Unable to fit for the eccentricity. Returning the eccentricity based on newtonian estimations from initial momentum.')
            return estimated_eccentricity, -1

        chi_1x = primary_dimensionless_spin[0]
        chi_1y = primary_dimensionless_spin[1]
        chi_1z = primary_dimensionless_spin[2]
        chi_2x = secondary_dimensionless_spin[0]
        chi_2y = secondary_dimensionless_spin[1]
        chi_2z = secondary_dimensionless_spin[2]
        s1z = primary_dimensional_spin[2]

        def quasicircular_ansatz(t, a, t_0):
            theta = np.power(((eta / 5) * abs(tmax * t_0 - t)), -1.0 / 8.0)
            c = 1
            gamma = c ** -2
            sqrt_1_minus_4_eta = np.sqrt(1 - 4 * np.round(eta, decimals=10))

            b1 = (11 * eta) / 32 + 743 / 2688
            b2 = (1 / 320) * (
                    -113 * ((sqrt_1_minus_4_eta - 1) * chi_1z - (sqrt_1_minus_4_eta + 1) * chi_2z) - 96 * np.pi) - (
                         19 / 80) * eta * (chi_1z + chi_2z)
            b3 = (371 * eta ** 2 / 2048) + (
                    eta * (61236 * s1z ** 2 - 119448 * chi_1z * chi_2z + 61236 * chi_2z ** 2 + 56975)) / 258048 + (
                         1 / 14450688) * (1714608 * (sqrt_1_minus_4_eta - 1) * chi_1z ** 2 - 1714608 * (
                    sqrt_1_minus_4_eta + 1) * chi_2z ** 2 + 1855099)
            b4 = - (1 / 128) * 17 * eta ** 2 * (chi_1z + chi_2z) + ((eta * (117 * np.pi - 2 * (
                    (63 * sqrt_1_minus_4_eta + 1213) * chi_1z + (
                    1213 - 63 * sqrt_1_minus_4_eta) * chi_2z))) / 2304) + ((-146597 * (
                    (sqrt_1_minus_4_eta - 1) * chi_1z - (
                    sqrt_1_minus_4_eta + 1) * chi_2z) - 46374 * np.pi) / 129024)
            b5 = ((235925 * eta ** 3) / 1769472) + eta ** 2 * (
                    ((335129 * chi_1z ** 2) / 2457600) - ((488071 * s1z * chi_2z) / 1228800) + (
                    (335129 * chi_2z ** 2) / 2457600) - (30913 / 1835008)) + eta * (
                         (((23281001 - 6352738 * sqrt_1_minus_4_eta) * chi_1z ** 2) / 68812800) + chi_1z * (
                         ((1051 * np.pi) / 3200) - ((377345 * chi_2z) / 1376256)) + (
                                 ((6352738 * sqrt_1_minus_4_eta + 23281001) * chi_2z ** 2) / 68812800) + (
                                 (1051 * np.pi * chi_2z) / 3200) - ((451 * np.pi ** 2) / 2048) + (
                                 25302017977 / 4161798144)) + (
                         (6127 * np.pi * (sqrt_1_minus_4_eta - 1) * chi_1z) / 12800) - (
                         (16928263 * (sqrt_1_minus_4_eta + 1) * chi_2z ** 2) / 137625600) + (
                         (16928263 * (sqrt_1_minus_4_eta - 1) * chi_1z ** 2) / 137625600) - (
                         (6127 * np.pi * (sqrt_1_minus_4_eta + 1) * chi_2z) / 12800) + ((53 * np.pi ** 2) / 200) - (
                         720817631400877 / 288412611379200) + (107 / 280) * (gamma + np.log(2 * theta))

            A = (a * theta ** 3 / 8) * (
                    1 + b1 * theta ** 2 + b2 * theta ** 3 + b3 * theta ** 4 + b4 * theta ** 5 + b5 * theta ** 6)

            return A

        try:
            qc_fit_values, qc_pcov = curve_fit(quasicircular_ansatz, time_inspiral, orbital_frequency_inspiral,
                                               bounds=([0, -np.inf], [3, np.inf]))

            qc_fit = quasicircular_ansatz(time_inspiral, qc_fit_values[0], qc_fit_values[1])

            eccentricity_from_qc = max(np.abs(np.divide((orbital_frequency_inspiral - qc_fit), 2 * qc_fit)))
            
        except RuntimeError as e:
            print(f"Runtime error encountered when trying to fit the orbital frequency: {e}")
            return estimated_eccentricity, -1

        r = self.separation_at_time(desired_time)
        q = self.mass_ratio

        omega_0 = (1 / (r ** (3 / 2))) * (
                1 - (1 / r) * (
                (3 * q ** 2 + 5 * q + 3) / (2 * (q + 1) ** 2))
                + (1 / (r ** (3 / 2))) * (
                        -(((3 * q + 4) * chi_1z) / (4 * (q + 1) ** 2)) - (
                        (q * (4 * q + 3) * chi_2z) / (4 * (q + 1) ** 2)))
                + (1 / (r ** 2)) * (
                        -((3 * q ** 2 * chi_2x ** 2) / (2 * (q + 1) ** 2))
                        + ((3 * q ** 2 * chi_2y ** 2) / (4 * (q + 1) ** 2))
                        + ((3 * q ** 2 * chi_2z ** 2) / (4 * (q + 1) ** 2))
                        + ((24 * q ** 4 + 103 * q ** 3 + 164 * q ** 2 + 103 * q + 24) / (16 * (q + 1) ** 4))
                        - ((3 * chi_1x ** 2) / (2 * (q + 1) ** 2))
                        - ((3 * q * chi_1x * chi_2x) / ((q + 1) ** 2))
                        + ((3 * chi_1y ** 2) / (4 * (q + 1) ** 2))
                        + ((3 * q * chi_1y * chi_2y) / (2 * (q + 1) ** 2))
                        + ((3 * chi_1z ** 2) / (4 * (q + 1) ** 2))
                        + ((3 * q * chi_1z * chi_2z) / (2 * (q + 1) ** 2)))
                + (1 / (r ** (5 / 2))) * (
                        ((3 * (13 * q ** 3 + 34 * q ** 2 + 30 * q + 16) * chi_1z) / (16 * (q + 1) ** 4))
                        + ((3 * q * (16 * q ** 3 + 30 * q ** 2 + 34 * q + 13) * chi_2z) / (16 * (q + 1) ** 4)))
                + (1 / (r ** 3)) * (
                        (((155 * q ** 2 + 180 * q + 76) * chi_1x ** 2) / (16 * (q + 1) ** 4 * r ** 3))
                        + ((q * (120 * q ** 2 + 187 * q + 120) * chi_1x * chi_2x) / (8 * (q + 1) ** 4 * r ** 3))
                        - (((55 * q ** 2 + 85 * q + 43) * chi_1y ** 2) / (8 * (q + 1) ** 4 * r ** 3))
                        - ((q * (54 * q ** 2 + 95 * q + 54) * chi_1y * chi_2y) / (4 * (q + 1) ** 4 * r ** 3))
                        - ((q * (96 * q ** 2 + 127 * q + 96) * chi_1z * chi_2z) / (16 * (q + 1) ** 4 * r ** 3))
                        + ((q ** 2 * (76 * q ** 2 + 180 * q + 155) * chi_2x ** 2) / (16 * (q + 1) ** 4 * r ** 3))
                        - ((q ** 2 * (43 * q ** 2 + 85 * q + 55) * chi_2y ** 2) / (8 * (q + 1) ** 4 * r ** 3))
                        - ((q ** 2 * (2 * q + 5) * (14 * q + 27) * chi_2z ** 2) / (32 * (q + 1) ** 4 * r ** 3))
                        - (((5 * q + 2) * (27 * q + 14) * chi_1z ** 2) / (32 * (q + 1) ** 4 * r ** 3))
                        + ((501 * np.pi ** 2 * q * (q + 1) ** 4 - 4 * (
                        120 * q ** 6 + 2744 * q ** 5 + 10049 * q ** 4 + 14820 * q ** 3 + 10049 * q ** 2 + 2744 * q + 120)) / (
                                   384 * (q + 1) ** 6 * r ** 3)))
                + (1 / (r ** (7 / 2))) * (
                        ((3 * (4 * q + 1) * q ** 3 * chi_2x ** 2 * chi_2z) / (2 * (q + 1) ** 4))
                        - ((3 * (4 * q + 1) * q ** 3 * chi_2y ** 2 * chi_2z) / (8 * (q + 1) ** 4))
                        - ((3 * (4 * q + 1) * q ** 3 * chi_2z ** 3) / (8 * (q + 1) ** 4))
                        + chi_1x * (((9 * (2 * q + 1) * q ** 2 * chi_2x * chi_2z) / (4 * (q + 1) ** 4)) + (
                        (9 * (q + 2) * q * chi_2x * chi_2z) / (4 * (q + 1) ** 4)))
                        + chi_1y * (((9 * q ** 2 * chi_2y * chi_1z) / (4 * (q + 1) ** 4)) + (
                        (9 * q ** 2 * chi_2y * chi_2z) / (4 * (q + 1) ** 4)))
                        + chi_1z * (((9 * q ** 2 * (2 * q + 3) * chi_2x ** 2) / (4 * (q + 1) ** 4)) - (
                        (9 * q ** 2 * (q + 2) * chi_2y ** 2) / (4 * (q + 1) ** 4)) - (
                                            (9 * q ** 2 * chi_2z ** 2) / (4 * (q + 1) ** 3)) - ((
                                                                                                        135 * q ** 5 + 385 * q ** 4 + 363 * q ** 3 + 377 * q ** 2 + 387 * q + 168) / (
                                                                                                        32 * (
                                                                                                        q + 1) ** 6)))
                        - (((
                                    168 * q ** 5 + 387 * q ** 4 + 377 * q ** 3 + 363 * q ** 2 + 385 * q + 135) * q * chi_2z) / (
                                   32 * (q + 1) ** 6))
                        + chi_1x ** 2 * (((3 * (q + 4) * chi_1z) / (2 * (q + 1) ** 4)) + (
                        (9 * q * (3 * q + 2) * chi_2z) / (4 * (q + 1) ** 4)))
                        + chi_1y ** 2 * (-((3 * (q + 4) * chi_1z) / (8 * (q + 1) ** 4)) - (
                        (9 * q * (2 * q + 1) * chi_2z) / (4 * (q + 1) ** 4)))
                        - ((9 * q * chi_1z ** 2 * chi_2z) / (4 * (q + 1) ** 3))
                        - ((3 * (q + 4) * chi_1z ** 3) / (8 * (q + 1) ** 4))))

        def eccentric_ansatz(t, a, t_0, e, omega_1, t_1):
            A = quasicircular_ansatz(t, a, t_0) + e * np.cos(omega_0 * omega_1 * t + t_1)
            return A

        def eccentric_part_only(t, e, omega_1, t_1):
            A = e * np.cos(omega_0 * omega_1 * t + t_1)
            return A

        try:
            eccentric_fit_values, _ = curve_fit(eccentric_ansatz, time_inspiral,
                                                             orbital_frequency_inspiral)


            # compute eccentricity using eccentric ansatz
            e = eccentric_fit_values[2]
            eccentricity_from_eccentric_fit = e / (2 * qc_fit[0])

            eccentricity = eccentricity_from_eccentric_fit

            eccentric_ansatz_fit_line = eccentric_ansatz(time_inspiral, eccentric_fit_values[0], eccentric_fit_values[1], eccentric_fit_values[2], eccentric_fit_values[3], eccentric_fit_values[4])

            error = np.sqrt(np.mean(((eccentric_ansatz_fit_line - orbital_frequency_inspiral)) ** 2)) / np.max(np.abs(orbital_frequency_inspiral - qc_fit))
            if error > 0.3:
                raise RuntimeError('Eccentric fit not close enough to data.')
            
            # flip the sign to match apoapsis and periapsis to separation
            eccentricity_only_timeseries = -1 * eccentric_part_only(time_inspiral, eccentric_fit_values[2],
                                                                    eccentric_fit_values[3],
                                                                    eccentric_fit_values[4])
            # mean anomaly_from_eccentric_timeseries
            if abs(eccentricity) < 1e-3:
                mean_anomaly = -1
            else:
                mean_anomaly = self._anomaly_from_eccentric_timeseries(time_inspiral, eccentricity_only_timeseries, desired_time)
                
            return abs(eccentricity), mean_anomaly

        except RuntimeError as e:
            warnings.warn(f"Runtime error encountered when trying to fit the orbital frequency with an eccentric ansatz. Returning the eccentricity computed using the quasi-circular fit.")
            return eccentricity_from_qc, -1

    def extrapolate_psi4_to_infinite_radius(self, order: int = 1, extraction_radius: float = None):
        """Calculate :math:`\Psi_4` at infinite radius by extrapolation.

        Extrapolates from a finite radius to obtain :math:`\Psi_4` at infinite radius using the method described in
        https://arxiv.org/abs/1008.4360 and https://arxiv.org/abs/1108.4421.

        Args:
            order (:obj:`int`, optional): order of extraction. Defaults to 1.
            extraction_radius: radius to extrapolate from

        """
        if extraction_radius is not None:
            self.radius_for_extrapolation = extraction_radius

        self.__radiation_mode_bundle.create_extrapolated_sphere(order=order)

    def psi4_real_imag_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> tuple:
        """Real and imaginary components of :math:`\Psi_4` for a given mode.

        Returns the time and real and imaginary parts of :math:`\Psi_4` for a given mode and extraction radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.

        Returns:
            tuple: time, :math:`\Psi_4` real component, and :math:`\Psi_4` imaginary component for a given mode and extraction radius

        """
        real = self.__radiation_mode_bundle.get_psi4_real_for_mode(l, m, extraction_radius=extraction_radius)
        imag = self.__radiation_mode_bundle.get_psi4_imaginary_for_mode(l, m, extraction_radius=extraction_radius)
        time = self.__radiation_mode_bundle.get_time(extraction_radius)
        return time, real, imag

    def psi4_amp_phase_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> tuple:
        """Amplitude and phase of :math:`\Psi_4` for a given mode.

        Returns the time and the amplitude and phase of :math:`\Psi_4` for a given mode and extraction radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.

        Returns:
            tuple: time, amplitude, and phase of :math:`\Psi_4` for a given mode and extraction radius

        """
        amplitude = self.__radiation_mode_bundle.get_psi4_amplitude_for_mode(l, m, extraction_radius=extraction_radius)
        phase = self.__radiation_mode_bundle.get_psi4_phase_for_mode(l, m, extraction_radius=extraction_radius)
        time = self.__radiation_mode_bundle.get_time(extraction_radius)
        return time, amplitude, phase

    def strain_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> tuple:
        """Real and imaginary components of strain for a given mode.

        Returns the time and the plus and cross components of :math:`rh` for a given mode and extraction radius,
        where r is the extraction radiys. The strain is the second time integral of :math:`\Psi_4` computed using
        fixed-frequency integration.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.

        Returns:
            tuple: time, :math:`rh_+`, and :math:`rh_{\\times}` for a given mode and extraction radius

        """
        plus = self.__radiation_mode_bundle.get_strain_plus_for_mode(l, m, extraction_radius=extraction_radius)
        cross = self.__radiation_mode_bundle.get_strain_cross_for_mode(l, m, extraction_radius=extraction_radius)
        time = self.radiationbundle.get_time(extraction_radius)
        return time, plus, cross

    def strain_recomposed_at_sky_location(self, theta: float, phi: float, extraction_radius: float = 0) -> tuple:
        """Time, plus, and cross components of strain recomposed at a given sky location

        The strain is recomposed by summing up the modes using spin weighted spherical harmonics as
        :math:`h(t,\\theta,\phi) = \sum_{\ell,m} {}_{-2}Y_{\ell,m}(\\theta, \phi) h_{ \ell,m}(t)`

        Args:
            theta (float): :math:`0 \leq \\theta \lt \pi`
            phi (float): :math:`0 \leq \phi \lt 2\pi`
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided, extrapolates to
                infinite radius.

        Returns:
            tuple: time, :math:`rh_+`, and :math:`rh_{\\times}` recomposed at a given sky location and extraction radius
        """
        plus, cross = self.__radiation_mode_bundle.get_strain_recomposed_at_sky_location(theta=theta, phi=phi,
                                                                                  extraction_radius=extraction_radius)
        time = self.__radiation_mode_bundle.get_time(extraction_radius)
        return time, plus, cross

    def strain_amp_phase_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> tuple:
        """Amplitude and phase of strain for a given mode.

        Returns the time and amplitude and phase of the strain for a given mode and extraction radius. The
        strain is the second time integral of :math:`\Psi_4` computed using fixed-frequency integration.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.

        Returns:
            tuple: time, :math:`rh_+`, and :math:`rh_{\\times}` for a given mode and extraction radius

        """
        amp = self.__radiation_mode_bundle.get_strain_amplitude_for_mode(l, m, extraction_radius=extraction_radius)
        phase = self.__radiation_mode_bundle.get_strain_phase_for_mode(l, m, extraction_radius=extraction_radius)
        time = self.radiationbundle.get_time(extraction_radius)
        return time, amp, phase

    def psi4_max_time_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> float:
        """Time of maximum :math:`\Psi_4` amplitude for a given mode.

        The time at which the amplitude of :math:`\Psi_4` reaches its peak.

        Args:
            l (int): specific l value at which to compute and sum
            m (int): specific m value to specify a single mode
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided, extrapolates to
                infinite radius.

        Returns:
            float: time of :math:`\Psi_4` max for given mode and extraction radius

        """
        return self.__radiation_mode_bundle.get_psi4_max_time_for_mode(l, m, extraction_radius)

    def dEnergy_dt_radiated(self, lmin: int = None, lmax: int = None, l: int = None, m: int = None,
                            extraction_radius: float = None) -> tuple:
        """Rate at which energy is radiated in gravitational waves

        Computed using the method described in https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided,
        it computes the total sum of all modes.

        Args:
            lmin (:obj:`int`, optional): minumum value of l range
            lmax (:obj:`int`, optional): maximum value of l range
            l (:obj:`int`, optional): specific l value
            m (:obj:`int`, optional): specific m value
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided, extrapolates to
                infinite radius.

        Returns:
            np.array: time
            np.array: dE/dt

        """
        kwargs = {}
        if lmin is not None:
            kwargs['lmin'] = lmin
        if lmax is not None:
            kwargs['lmax'] = lmax
        if l is not None:
            kwargs['l'] = l
        if m is not None:
            kwargs['m'] = m
        if extraction_radius is not None:
            kwargs['extraction_radius'] = extraction_radius

        time, energydot_radiated = self.__radiation_mode_bundle.get_dEnergy_dt_radiated(**kwargs)

        return time, energydot_radiated

    def energy_radiated(self, lmin: int = None, lmax: int = None, l: int = None, m: int = None,
                        extraction_radius: float = None) -> tuple:
        """Cummulative nergy radiated in gravitational waves.

        Returns the cumulative energy radiated computed using the method described in https://arxiv.org/abs/0707.4654.
        If no lmin, lmax, l, or m are provided, it computes the total sum of all modes.

        Args:
            lmin (:obj:`int`, optional): minumum value of l range
            lmax (:obj:`int`, optional): maximum value of l range
            l (:obj:`int`, optional): specific l value
            m (:obj:`int`, optional): specific m value
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided, extrapolates to
                infinite radius.

        Returns:
            np.array: time
            np.array: cumulative energy radiated for a given l range and extraction radius

        """
        kwargs = {}
        if lmin is not None:
            kwargs['lmin'] = lmin
        if lmax is not None:
            kwargs['lmax'] = lmax
        if l is not None:
            kwargs['l'] = l
        if m is not None:
            kwargs['m'] = m
        if extraction_radius is not None:
            kwargs['extraction_radius'] = extraction_radius

        time, energy_radiated = self.__radiation_mode_bundle.get_energy_radiated(**kwargs)

        return time, energy_radiated

    def dP_dt_radiated(self, lmin: int = None, lmax: int = None, l: int = None, m: int = None,
                       extraction_radius: float = None) -> tuple:
        """Derivative of linear momentum radiated.

        The rate of linear momentum being radiated through gravitational waves, computed using the method described in
        https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it computes the total sum of all modes.

        Args:
            lmin (:obj:`int`, optional): minumum value of l range
            lmax (:obj:`int`, optional): maximum value of l range
            l (:obj:`int`, optional): specific l value
            m (:obj:`int`, optional): specific m value
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided, extrapolates to
                infinite radius.

        Returns:
            np.ndarray: time
            np.ndarray: dP/dt

        """
        kwargs = {}
        if lmin is not None:
            kwargs['lmin'] = lmin
        if lmax is not None:
            kwargs['lmax'] = lmax
        if l is not None:
            kwargs['l'] = l
        if m is not None:
            kwargs['m'] = m
        if extraction_radius is not None:
            kwargs['extraction_radius'] = extraction_radius

        time, dP_dt = self.radiationbundle.get_dP_dt_radiated(**kwargs)
        return time, dP_dt

    def linear_momentum_radiated(self, lmin: int = None, lmax: int = None, l: int = None, m: int = None,
                                 extraction_radius: float = None) -> tuple:
        """Cummulative inear momentum radiated.

        Cummulative linear momentum radiated through gravitational waves, computed using the method described in
        https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it computes the total sum of all modes.

        Args:
            lmin (:obj:`int`, optional): minumum value of l range
            lmax (:obj:`int`, optional): maximum value of l range
            l (:obj:`int`, optional): specific l value
            m (:obj:`int`, optional): specific m value
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.

        Returns:
            np.ndarray: time
            np.ndarray: cummulative linear momentum radiated

        """
        kwargs = {}
        if lmin is not None:
            kwargs['lmin'] = lmin
        if lmax is not None:
            kwargs['lmax'] = lmax
        if l is not None:
            kwargs['l'] = l
        if m is not None:
            kwargs['m'] = m
        if extraction_radius is not None:
            kwargs['extraction_radius'] = extraction_radius

        time, linear_momentum = self.radiationbundle.get_linear_momentum_radiated(**kwargs)
        return time, linear_momentum

    def compact_object_data_for_object(self, object_num) -> np.ndarray:
        """Function to be used internally to return all the data for a given compact object."""
        return self.__h5_file["compact_object"]["object=%d" % object_num]

    def compact_object_metadata_dict(self) -> dict:
        """Function to be used internally to retrieve the metadata relating to the compact objects."""
        metadata_dict = {}
        for key in self.__h5_file["compact_object"].attrs:
            metadata_dict[key] = self.__h5_file["compact_object"].attrs[key]
        return metadata_dict

    def close(self):
        """Close the Coalescence object by closing its associated h5 file."""
        self.__h5_file.close()
