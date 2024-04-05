import warnings
from enum import Enum
import h5py
import numpy as np
import scipy.integrate
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt
from scipy.signal.windows import blackmanharris
import math


class Frame(Enum):
    RAW = 1
    COM_CORRECTED = 2


class RadiationBundle:
    """Class for interacting with all radiative information from the simulation."""

    def __init__(self, radiation_spheres: dict):
        self.__radiation_spheres = radiation_spheres
        self.__extrapolated_sphere = None
        self.__radius_for_extrapolation = None
        self.__frame = Frame.RAW

    @staticmethod
    def create_radiation_bundle(radiation_group: h5py.Group):
        """Create a RadiationBundle

        Args:
            radiation_group (h5py.Group): group containing all the data pertaining to :math:`\Psi_4` for all modes
                and all extraction radii.

        Returns:
            RadiationBundle: a RadiationBundle consisting of RadiationSpheres for each extraction radius

        """
        radiation_spheres = {}
        if 'psi4' not in radiation_group:
            warnings.warn('There is no psi4 data to create a radiation bundle from.')
            return None
        psi4_group = radiation_group["psi4"]
        for radius_group_name, radius_group in psi4_group.items():
            if radius_group_name != "extrapolated":
                try:
                    radius = float(radius_group_name[radius_group_name.find('=') + 1:])
                    new_radiation_sphere = RadiationSphere.create_radiation_sphere(radius_group=radius_group,
                                                                                   radius=radius, extrapolated=False)
                    radiation_spheres[radius] = new_radiation_sphere
                except Exception as e:
                    print(e)
                    warnings.warn(f"Unable to process this radius group name: {radius_group_name}")
        if len(radiation_spheres) == 0:
            warnings.warn('There is no data to create a radiation sphere and therefore a radiation bundle from.')
            return None
        return RadiationBundle(radiation_spheres=radiation_spheres)

    @property
    def frame(self) -> Frame:
        """The frame for the radiation modes

        This can be either the raw frame (Frame.RAW) or the frame which corrects for the drift of the center of mass
        (FRAME.COM_CORRECTED)."""
        return self.__frame

    def set_frame(self, new_frame: Frame, time: np.ndarray = None, center_of_mass: np.ndarray = None):
        """Set the frame to use when decomposing the modes.

        Options given by the frame enum and are the raw, original frame or the frame corrected for center of mass drift.

        Args:
            new_frame (Frame): Frame to transform to
            time (:obj:`numpy.ndarray`, optional): Time stamps for center of mass. Only necessary if moving to center of mass corrected frame.
            center_of_mass (:obj:`numpy.ndarray`, optional): Time series of center of mass. Only necessary if moving to center of mass corrected frame.

        """
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            raise ImportError('Unable to change the radiation frame. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

        if type(new_frame) != Frame:
            warnings.warn("You must provide the frame as a Frame enum value")
            return
        for radiation_sphere in self.radiation_spheres.values():
            radiation_sphere.set_frame(new_frame, time=time, center_of_mass=center_of_mass)
        if self.__extrapolated_sphere is not None:
            self.__extrapolated_sphere.set_frame(new_frame, time=time, center_of_mass=center_of_mass)
        self.__frame = new_frame

    @property
    def radiation_spheres(self) -> dict:
        """Dictionary containing all radius -> RadiationSphere pairs"""
        return self.__radiation_spheres

    @property
    def extrapolated_sphere(self):
        """The RadiationSphere with extrapolated radius. All :math:`\Psi_4` data has been extrapolated to infinite
        radius using the method described in https://arxiv.org/abs/1008.4360 and https://arxiv.org/abs/1108.4421."""
        if self.__extrapolated_sphere is None:
            self.create_extrapolated_sphere()
        return self.__extrapolated_sphere

    @property
    def radius_for_extrapolation(self) -> float:
        """The radius from which extrapolation to infinite radius will be computed."""
        if self.__radius_for_extrapolation is None:
            included_radii = self.included_radii
            # find closest to 75, pick lower over higher if equal
            difference_from_75 = [abs(radius - 75) for radius in included_radii]
            self.__radius_for_extrapolation = included_radii[np.argmin(difference_from_75)]
            warnings.warn(
                "Using data extrapolated from {radius}M for radius extrapolation. You can set this manually by "
                "setting radius_for_extrapolation".format(
                    radius=self.__radius_for_extrapolation))
        return self.__radius_for_extrapolation

    @radius_for_extrapolation.setter
    def radius_for_extrapolation(self, radius: float):
        """Set the radius to use when extrapolating to infinity

        Args:
            radius (float): radius to use when extrapolating to infinity

        """
        if radius not in self.radiation_spheres:
            warnings.warn(
                "There is no data at a radiation radius of {radius}M. Cannot use to extrapolate the radius.".format(
                    radius=radius))
            return
        self.__radius_for_extrapolation = radius
        # reset extrapolated sphere so it uses the new radiation radius for extrapolation
        self.__extrapolated_sphere = None

    @property
    def l_max(self) -> int:
        """Maximum l mode included."""
        return max([sphere.l_max for sphere in self.radiation_spheres.values()])

    @property
    def included_modes(self) -> list:
        """:math:`\Psi_4` is decomposed using spherical harmonics labeled by (l, m). This provides a list of all (l,m)
        modes included."""
        return sorted(list({mode for sphere in self.radiation_spheres.values() for mode in sphere.included_modes}))

    @property
    def included_radii(self) -> list:
        """List of all extraction radii included."""
        return sorted(list(self.radiation_spheres.keys()))

    def get_time(self, extraction_radius: float = 0) -> np.ndarray:
        """Time array associated with all radiation timeseries at the given radius.

        Args:
            extraction_radius (:obj:`float`, optional): Extraction radius for which the time data is requested.

        Returns:
            numpy.ndarray: array containing the time data for all radiation information at the given radius

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.time
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        time = self.radiation_spheres[extraction_radius].time
        return time

    def get_psi4_real_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> np.ndarray:
        """Real component of :math:`\Psi_4` for a given mode and extraction radius.

        Returns the real part of :math:`\Psi_4` for a given mode and extraction radius. If the extraction radius is 0,
        the data is extrapolated to infinite radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): extraction radius for :math:`\Psi_4` data. If 0, the data at
                infinity is provided.

        Returns:
            numpy.ndarray: :math:`\Psi_4` real component for a given mode and extraction radius.

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_psi4_real_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_psi4_real_for_mode(l=l, m=m)

    def get_psi4_imaginary_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> np.ndarray:
        """Imaginary component of :math:`\Psi_4` for a given mode and extraction radius.

        Returns the imaginary part of :math:`\Psi_4` for a given mode and extraction radius. If the extraction radius is
        0, the data is extrapolated to infinite radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): extraction radius for :math:`\Psi_4` data. If 0, the data at
                infinity is provided.

        Returns:
            numpy.ndarray: :math:`\Psi_4` imaginary component for a given mode and extraction radius.

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_psi4_imaginary_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_psi4_imaginary_for_mode(l=l, m=m)

    def get_psi4_amplitude_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> np.ndarray:
        """Amplitude of :math:`\Psi_4` for a given mode and extraction radius.

        Returns the amplitude of :math:`\Psi_4` for a given mode and extraction radius. If the extraction radius is 0,
        the data is extrapolated to infinite radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): extraction radius for :math:`\Psi_4` data. If 0, the data at
                infinity is provided.

        Returns:
            numpy.ndarray: :math:`\Psi_4` amplitude for a given mode and extraction radius.

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_psi4_amplitude_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_psi4_amplitude_for_mode(l=l, m=m)

    def get_psi4_phase_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> np.ndarray:
        """Phase of :math:`\Psi_4` for a given mode and extraction radius.

        Returns the phase of :math:`\Psi_4` for a given mode and extraction radius. If the extraction radius is 0,
        the data is extrapolated to infinite radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): extraction radius for :math:`\Psi_4` data. If 0, the data at
                infinity is provided.

        Returns:
            numpy.ndarray: :math:`\Psi_4` phase for a given mode and extraction radius.

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_psi4_phase_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_psi4_phase_for_mode(l=l, m=m)

    def get_strain_plus_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> np.ndarray:
        """Plus component of :math:`rh` for a given mode and extraction radius.

        Returns the plus component of strain for a given mode and extraction radius. The strain is the second time
        integral of :math:`\Psi_4` computed using fixed-frequency integration. If the extraction radius is 0, the data
        is extrapolated to infinite radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): extraction radius for strain data. If 0, the data at infinity is
                provided.

        Returns:
            numpy.ndarray: :math:`rh_+` for a given mode and extraction radius.

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_strain_plus_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_strain_plus_for_mode(l=l, m=m)

    def get_strain_cross_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> np.ndarray:
        """Cross component of :math:`rh` for a given mode and extraction radius.

        Returns the cross component of strain for a given mode and extraction radius. The strain is the second time
        integral of :math:`\Psi_4` computed using fixed-frequency integration. If the extraction radius is 0, the data
        is extrapolated to infinite radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): extraction radius for strain data. If 0, the data at infinity is
                provided.

        Returns:
            numpy.ndarray: :math:`rh_{\\times}` for a given mode and extraction radius.

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_strain_cross_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_strain_cross_for_mode(l=l, m=m)

    def get_strain_recomposed_at_sky_location(self, theta: float, phi: float, extraction_radius: float = 0) -> tuple:
        """Plus and cross components of strain recomposed at a given sky location

        The strain is recomposed by summing up the modes using spin weighted spherical harmonics as
        :math:`h(t,\\theta,\phi) = \sum_{\ell,m} {}_{-2}Y_{\ell,m}(\\theta, \phi) h_{ \ell,m}(t)`. If the extraction
        radius is 0, the data is extrapolated to infinite radius.

        Args:
            theta (float): :math:`0 \leq \\theta \lt \pi`
            phi (float): :math:`0 \leq \phi \lt 2\pi`
            extraction_radius (:obj:`float`, optional): extraction radius for strain data. If 0, the data at infinity is
                provided.

        Returns:
            tuple: :math:`rh_{+}` and :math:`rh_{\\times}` recomposed at a given sky location

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_strain_recomposed_at_sky_location(theta=theta, phi=phi)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_strain_recomposed_at_sky_location(theta=theta, phi=phi)

    def get_strain_amplitude_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> np.ndarray:
        """Amplitude of :math:`rh` for a given mode and extraction radius.

        Returns the amplitude of strain for a given mode and extraction radius. The strain is the second time
        integral of :math:`\Psi_4` computed using fixed-frequency integration. If the extraction radius is 0, the data
        is extrapolated to infinite radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): extraction radius for strain data. If 0, the data at infinity is
                provided.

        Returns:
            numpy.ndarray: amplitude of :math:`rh` for a given mode and extraction radius.

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_strain_amplitude_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_strain_amplitude_for_mode(l=l, m=m)

    def get_strain_phase_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> np.ndarray:
        """Phase of :math:`rh` for a given mode and extraction radius.

        Returns the phase of strain for a given mode and extraction radius. The strain is the second time
        integral of :math:`\Psi_4` computed using fixed-frequency integration. If the extraction radius is 0, the data
        is extrapolated to infinite radius.


        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): extraction radius for strain data. If 0, the data at infinity is
                provided.

        Returns:
            numpy.ndarray: phase of :math:`rh` for a given mode and extraction radius.

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_strain_phase_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_strain_phase_for_mode(l=l, m=m)

    def get_psi4_max_time_for_mode(self, l: int, m: int, extraction_radius: float = 0) -> int:
        """Time of maximum :math:`\Psi_4` amplitude for a given mode and extraction radius.

        The time at which the amplitude of :math:`\Psi_4` reaches its peak. If the extraction radius is 0, the data is
        extrapolated to infinite radius.

        Args:
            l (int): l value of mode
            m (int): m value of mode
            extraction_radius (:obj:`float`, optional): radius at which :math:`\Psi_4` was extracted

        Returns:
            float: time of :math:`\Psi_4` max for given mode and extraction radius

        """
        if extraction_radius == 0:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            return self.extrapolated_sphere.get_psi4_max_time_for_mode(l=l, m=m)
        if extraction_radius not in self.radiation_spheres:
            warnings.warn("There is no data at a radiation radius of {radius}M.".format(radius=extraction_radius))
            return None
        return self.radiation_spheres[extraction_radius].get_psi4_max_time_for_mode(l=l, m=m)

    def get_dEnergy_dt_radiated(self, extraction_radius: float = None, **kwargs) -> tuple:
        """Rate at which energy is radiated, :math:`dE/dt`

        Uses the method described in https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it
        computes the total sum of all modes.

        Args:
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.
            **kwargs : parameters to specify the modes you would like to sum over

                Valid keyword arguments are:

                lmin (:obj:`int`, optional): minimum value of l range

                lmax (:obj:`int`, optional): maximum value of l range

                l (:obj:`int`, optional): specific l value

                m (:obj:`int`, optional): specific m value

        Returns:
            tuple: time, :math:`dE/dt` for the given modes

        """
        if extraction_radius is None:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None
            desired_sphere = self.extrapolated_sphere
        else:
            if extraction_radius not in self.radiation_spheres:
                return None
            desired_sphere = self.radiation_spheres[extraction_radius]

        time, energydot_radiated = desired_sphere.get_dEnergy_dt_radiated(**kwargs)

        return time, energydot_radiated

    def get_energy_radiated(self, extraction_radius: float = None, **kwargs) -> tuple:
        """Cumulative radiated energy :math:`E`

        Uses the method described in https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it
        computes the total sum of all modes.

        Args:
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.
            **kwargs : parameters to specify the modes you would like to sum over

                Valid keyword arguments are:

                lmin (:obj:`int`, optional): minumum value of l range

                lmax (:obj:`int`, optional): maximum value of l range

                l (:obj:`int`, optional): specific l value

                m (:obj:`int`, optional): specific m value

        Returns:
            tuple: time, cumulative energy radiated for the given modes

        """
        if extraction_radius is None:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None, None
            desired_sphere = self.extrapolated_sphere
        else:
            if extraction_radius not in self.radiation_spheres:
                return None, None
            desired_sphere = self.radiation_spheres[extraction_radius]

        time, energy_radiated = desired_sphere.get_energy_radiated(**kwargs)

        return time, energy_radiated

    def get_dP_dt_radiated(self, extraction_radius: float = None, **kwargs) -> tuple:
        """Rate at which linear momentum is radiated

        Rate at which linear momentum is radiated through gravitational waves, computed using the method described in
        https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided,
        it computes the total sum of all modes.

        Args:
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.
            **kwargs : parameters to specify the modes you would like to sum over

                Valid keyword arguments are:

                lmin (:obj:`int`, optional): minumum value of l range

                lmax (:obj:`int`, optional): maximum value of l range

                l (:obj:`int`, optional): specific l value

                m (:obj:`int`, optional): specific m value

        Returns:
            tuple: time, dP/dt for the given modes

        """
        if extraction_radius is None:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None, None
            desired_sphere = self.extrapolated_sphere
        else:
            if extraction_radius not in self.radiation_spheres:
                return None
            desired_sphere = self.radiation_spheres[extraction_radius]

        time, dP_dt_radiated = desired_sphere.get_dP_dt_radiated(**kwargs)

        return time, dP_dt_radiated

    def get_linear_momentum_radiated(self, extraction_radius: float = None, **kwargs) -> tuple:
        """Linear momentum radiated

        Cummulative linear momentum radiated through gravitational waves, computed using the method described in
        https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it
        computes the total sum of all modes.

        Args:
            extraction_radius (:obj:`float`, optional): radius for gravitational wave extraction. If not provided,
                extrapolates to infinite radius.
            **kwargs : parameters to specify the modes you would like to sum over

                Valid keyword arguments are:

                lmin (:obj:`int`, optional): minumum value of l range

                lmax (:obj:`int`, optional): maximum value of l range

                l (:obj:`int`, optional): specific l value

                m (:obj:`int`, optional): specific m value

        Returns:
            tuple: time, cummulative linear momentum radiated for the given modes

        """
        if extraction_radius is None:
            if self.extrapolated_sphere is None:
                warnings.warn("There is no data extrapolated to infinity for that mode")
                return None, None
            desired_sphere = self.extrapolated_sphere
        else:
            if extraction_radius not in self.radiation_spheres:
                return None
            desired_sphere = self.radiation_spheres[extraction_radius]

        time, linear_momentum_radiated = desired_sphere.get_linear_momentum_radiated(**kwargs)

        return time, linear_momentum_radiated

    def create_extrapolated_sphere(self, order: int = 1):
        """Create a RadiationSphere object extrapolated from the RadiationSphere at the provided radius for
        extrapolation.

        Extrapolate the :math:`\Psi_4` of all RadiationModes in the RadiationSphere at the set radius for extrapolation
        to infinite radius and create and store a new RadiationSphere with those extrapolated RadiationModes. Uses
        the extrapolation method described in https://arxiv.org/abs/1008.4360 and https://arxiv.org/abs/1108.4421.

        Args:
            order (:obj:`int`, optional): the extrapolation order. Defaults to 1.

        """
        radiation_sphere = self.radiation_spheres[self.radius_for_extrapolation]
        extrap_sphere = radiation_sphere.get_extrapolated_sphere(order=order)
        if extrap_sphere is None:
            return
        self.__extrapolated_sphere = extrap_sphere


class RadiationSphere:
    """Class for interacting with the radiative data associated with a single extraction sphere."""

    def __init__(self, mode_dict: dict, time: np.ndarray, radius: float,
                 extrapolated: bool = False):
        self.__raw_modes = mode_dict
        self.__com_corrected_modes = None
        self.__time = time
        self.__radius = radius
        self.__extrapolated = extrapolated

        self.__frame = Frame.RAW
        self.__com_corrected_time = None

        self.__alpha = None
        self.__beta = None

        for mode in self.__raw_modes.values():
            mode.radiation_sphere = self

    @staticmethod
    def create_radiation_sphere(radius_group: dict, radius: float, extrapolated: bool = False):
        """Create a RadiationSphere

        Args:
            radius_group (h5py.Group): group containing all the data pertaining to :math:`\Psi_4` for all modes at
                the given extraction radius.
            radius (float): the radius at which all the :math:`\Psi_4` data has been extracted.
            extrapolated (:obj:`bool`, optional): Whether the data has been extrapolated to infinite radius. Defaults
                to false.

        Returns:
            RadiationSphere: a RadiationSphere consisting of all RadiationModes at the given extraction radius

        """
        temp_modes = {}

        if 'time' not in radius_group:
            return None

        time = np.array(radius_group['time'])

        if 'modes' in radius_group:
            modes_group = radius_group["modes"]
            for l_group_name in list(modes_group.keys()):
                l_value = int(l_group_name[l_group_name.find('=') + 1:])
                l_group = modes_group[l_group_name]
                for m_group_name in l_group:
                    m_value = int(m_group_name[m_group_name.find('=') + 1:])
                    m_group = l_group[m_group_name]
                    temp_modes[(l_value, m_value)] = RadiationMode.create_radiation_mode(m_group, l_value,
                                                                                         m_value, radius,
                                                                                         np.array(time))

        return RadiationSphere(mode_dict=temp_modes, time=time, radius=radius, extrapolated=extrapolated)

    @property
    def frame(self) -> Frame:
        """Frame the modes are decomposed in.

        Options are either the raw inertial frame or the frame corrected for center of mass drift.
        """
        return self.__frame

    def set_frame(self, new_frame: Frame, time: np.ndarray = None, center_of_mass: np.ndarray = None,
                  alpha: np.ndarray = None, beta: np.ndarray = None):
        """Set the frame to use when decomposing the modes.

        Options given by the frame enum and are the raw, original frame or corrected for center of mass drift.

        Args:
            new_frame (Frame): Frame to transform to
            time (:obj:`numpy.ndarray`, optional): Time stamps for center of mass. Only necessary if moving to center of mass corrected frame and not providing alpha and beta.
            center_of_mass (:obj:`numpy.ndarray`, optional): Time series of center of mass. Only necessary if moving to center of mass corrected frame and not providing alpha and beta.
            alpha (:obj:`numpy.ndarray`, optional): Offset for center of mass correction. Only necessary if moving to center of mass corrected frame and not providing the center of mass timeseries.
            beta (:obj:`numpy.ndarray`, optional): Boost for center of mass correction. Only necessary if moving to center of mass corrected frame and not providing the center of mass timeseries.
        """
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            raise ImportError(
                'Unable to change the radiation frame. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

        if type(new_frame) != Frame:
            warnings.warn('You must provide the new frame as a Frame enum')
            return

        if alpha is not None and beta is not None:
            self.__alpha = alpha
            self.__beta = beta

        elif time is not None and center_of_mass is not None and (self.__alpha is None or self.__beta is None):
            self._set_alpha_beta_for_com_transformation(time, center_of_mass)

        self.__frame = new_frame

    @property
    def modes(self) -> dict:
        """Dictionary containing all the (l, m) -> RadiationMode objects in the current frame."""
        if self.frame == Frame.RAW:
            return self.raw_modes
        if self.frame == Frame.COM_CORRECTED:
            import sys
            if sys.version_info.major == 3 and sys.version_info.minor > 10:
                raise ImportError(
                    'Unable to return modes in center-of-mass corrected frame. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

            if self.__com_corrected_modes is None:
                self.__frame = Frame.RAW
                self._generate_com_corrected_modes()
                self.__frame = Frame.COM_CORRECTED
            if self.__com_corrected_modes is None:  # if it's Still none after trying to set it
                warnings.warn("Unable to correct for center of mass offset. Remaining in original frame.")
                self.__frame = Frame.RAW
                return self.raw_modes
            return self.__com_corrected_modes

    @property
    def raw_modes(self) -> dict:
        """Dictionary containing all the (l, m) -> RadiationMode objects in their original frame."""
        return self.__raw_modes

    @property
    def included_modes(self) -> list:
        """:math:`\Psi_4` is decomposed into spherical harmonics labeled by (l, m). This provides a list of all (l,
        m) modes included. """
        return sorted(list(self.raw_modes.keys()))

    @property
    def l_max(self) -> int:
        """Maximum l mode included."""
        if self.included_modes is None or self.included_modes == []:
            return 0
        return max([l for l, m in self.included_modes])

    @property
    def time(self) -> np.ndarray:
        """Time array associated with all timeseries provided by this RadiationSphere."""
        if self.frame == Frame.COM_CORRECTED:
            import sys
            if sys.version_info.major == 3 and sys.version_info.minor > 10:
                raise ImportError(
                    'Unable to return time in center-of-mass corrected frame. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

            return self.__com_corrected_time
        return self.__time

    @property
    def radius(self) -> float:
        """Radius at which the :math:`\Psi_4` data was extracted."""
        return self.__radius

    @property
    def extrapolated(self) -> bool:
        """Whether the :math:`\Psi_4` data has been extrapolated to infinite radius."""
        return self.__extrapolated

    def get_mode(self, l: int, m: int):
        """RadiationMode with a given (l, m).

        Returns the RadiationMode extracted on this RadiationSphere with the given spherical harmonic decomposition
        mode.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            RadiationMode: RadiationMode associated with the given (l, m)

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return
        return self.modes[(l, m)]

    def get_psi4_real_for_mode(self, l: int, m: int) -> np.ndarray:
        """Real component of :math:`\Psi_4` for a given mode.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            numpy.ndarray: :math:`\Psi_4` real component for a given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return
        return self.modes[(l, m)].psi4_real

    def get_psi4_imaginary_for_mode(self, l: int, m: int) -> np.ndarray:
        """Imaginary component of :math:`\Psi_4` for a given mode.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            numpy.ndarray: :math:`\Psi_4` imaginary component for a given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return
        return self.modes[(l, m)].psi4_imaginary

    def get_psi4_amplitude_for_mode(self, l: int, m: int) -> np.ndarray:
        """Amplitude of :math:`\Psi_4` for a given mode.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            numpy.ndarray: :math:`\Psi_4` amplitude for a given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return
        return self.modes[(l, m)].psi4_amplitude

    def get_psi4_phase_for_mode(self, l: int, m: int) -> np.ndarray:
        """
        Phase of :math:`\Psi_4` for a given mode.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            numpy.ndarray: :math:`\Psi_4` phase for a given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return
        return self.modes[(l, m)].psi4_phase

    def get_strain_plus_for_mode(self, l: int, m: int) -> np.ndarray:
        """Plus component of :math:`rh` for a given mode.

        The strain is the second time integral of :math:`\Psi_4` computed using fixed-frequency integration.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            numpy.ndarray: :math:`rh_+` for a given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return None

        return self.modes[(l, m)].strain_plus

    def get_strain_cross_for_mode(self, l: int, m: int) -> np.ndarray:
        """Cross component of :math:`rh` for a given mode.

        The strain is the second time integral of :math:`\Psi_4` computed using fixed-frequency integration.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            numpy.ndarray: :math:`rh_{\\times}` for a given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return None

        return self.modes[(l, m)].strain_cross

    def get_strain_amplitude_for_mode(self, l: int, m: int) -> np.ndarray:
        """Amplitude of :math:`rh` for a given mode.

        The strain is the second time integral of :math:`\Psi_4` computed using fixed-frequency integration.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            numpy.ndarray: amplitude of :math:`rh` for a given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return

        return self.modes[(l, m)].strain_amplitude

    def get_strain_phase_for_mode(self, l: int, m: int) -> np.ndarray:
        """Phase of :math:`rh` for a given mode.

        The strain is the second time integral of :math:`\Psi_4` computed using fixed-frequency integration.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            numpy.ndarray: phase of :math:`rh` for a given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return None

        return self.modes[(l, m)].strain_phase

    def get_strain_recomposed_at_sky_location(self, theta: float, phi: float) -> tuple:
        """Plus and cross components of strain recomposed at a given sky location

        The strain is recomposed by summing up the modes using spin weighted spherical harmonics as
        :math:`h(t,\\theta,\phi) = \sum_{\ell,m} {}_{-2}Y_{\ell,m}(\\theta, \phi) h_{ \ell,m}(t)`

        Args:
            theta (float): :math:`0 \leq \\theta \lt \pi`
            phi (float): :math:`0 \leq \phi \lt 2\pi`

        Returns:
            tuple: :math:`rh_+` and :math:`rh_{\\times}` recomposed at a given sky location

        """

        if len(self.modes) == 0:
            warnings.warn('This radiation sphere has no modes.')
            return None

        h_t_shape = list(self.modes.values())[0].strain_plus.shape
        h_t = np.zeros(h_t_shape, dtype=np.complex_)

        for mode in self.modes:
            l, m = mode

            h_lm = self.modes[l, m].strain_plus - 1j * self.modes[l, m].strain_cross

            # get the spherical harmonic coefficients for the mode
            y_lm = RadiationMode.ylm(l, m, theta, phi)

            h_t += y_lm * h_lm

        return np.real(h_t), -1 * np.imag(h_t)

    def get_psi4_max_time_for_mode(self, l: int, m: int) -> np.ndarray:
        """Time of maximum :math:`\Psi_4` amplitude for a given mode.

        The time at which the amplitude of :math:`\Psi_4` reaches its peak.

        Args:
            l (int): l value of mode
            m (int): m value of mode

        Returns:
            float: time of :math:`\Psi_4` max for given mode

        """
        if (l, m) not in self.modes:
            warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
            return
        return self.modes[(l, m)].psi4_max_time

    def get_dEnergy_dt_radiated(self, **kwargs) -> tuple:
        """Rate at which energy is radiated, :math:`dE/dt`

        Uses the method described in https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it
        computes the total sum of all modes.

        Args:
            **kwargs : parameters to specify the modes you would like to sum over

                Valid keyword arguments are:

                lmin (:obj:`int`, optional): minumum value of l range

                lmax (:obj:`int`, optional): maximum value of l range

                l (:obj:`int`, optional): specific l value

                m (:obj:`int`, optional): specific m value

        Returns:
            tuple: time, :math:`dE/dt` for given modes.

        """
        if 'm' in kwargs and 'l' not in kwargs:
            warnings.warn("Unclear input. Only an m requested without a corresponding l")
            return

        if 'l' in kwargs and ('lmin' in kwargs or 'lmax' in kwargs):
            warnings.warn("Unclear input. Asking for both a specific l and an l range. Please request either a "
                          "specific l, an l range, or neither.")
            return

        if 'l' in kwargs and 'm' in kwargs:
            l = kwargs['l']
            m = kwargs['m']
            if (l, m) not in self.modes:
                warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
                return
            return self.time, self.modes[(l, m)].dEnergy_dt_radiated

        lmin = 2
        lmax = self.l_max

        if 'l' in kwargs and 'm' not in kwargs:
            lmin = kwargs['l']
            lmax = kwargs['l']
        if 'lmin' in kwargs:
            lmin = kwargs['lmin']
        if 'lmax' in kwargs:
            lmax = kwargs['lmax']

        time = self.time

        energydot_radiated_total = np.zeros(len(time))

        for l in range(lmin, lmax + 1):
            for m in range(-l, l + 1):
                if (l, m) not in self.modes:
                    warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
                    return
                curr_mode_denergy_radiated = self.modes[(l, m)].dEnergy_dt_radiated
                energydot_radiated_total += curr_mode_denergy_radiated

        return time, energydot_radiated_total

    def get_energy_radiated(self, **kwargs) -> tuple:
        """Cumulative radiated energy :math:`E`

        Uses the method described in https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it
        computes the total sum of all modes.

        Args:
            **kwargs : parameters to specify the modes you would like to sum over

                Valid keyword arguments are:

                lmin (:obj:`int`, optional): minumum value of l range

                lmax (:obj:`int`, optional): maximum value of l range

                l (:obj:`int`, optional): specific l value

                m (:obj:`int`, optional): specific m value

        Returns:
            tuple: time, cumulative energy radiated for given modes.

        """
        if 'm' in kwargs and 'l' not in kwargs:
            warnings.warn("Unclear input. Only an m requested without a corresponding l")
            return None, None

        if 'l' in kwargs and ('lmin' in kwargs or 'lmax' in kwargs):
            warnings.warn("Unclear input. Asking for both a specific l and an l range. Please request either a "
                          "specific l, an l range, or neither.")
            return None, None

        if 'l' in kwargs and 'm' in kwargs:
            l = kwargs['l']
            m = kwargs['m']
            if (l, m) not in self.modes:
                warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
                return None, None
            return self.time, self.modes[(l, m)].energy_radiated

        lmin = 2
        lmax = self.l_max

        if 'l' in kwargs and 'm' not in kwargs:
            lmin = kwargs['l']
            lmax = kwargs['l']
        if 'lmin' in kwargs:
            lmin = kwargs['lmin']
        if 'lmax' in kwargs:
            lmax = kwargs['lmax']

        time = self.time

        energy_radiated_total = np.zeros(len(time))

        for l in range(lmin, lmax + 1):
            for m in range(-l, l + 1):
                if (l, m) not in self.modes:
                    warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
                    return None, None
                curr_mode_energy_radiated = self.modes[(l, m)].energy_radiated
                energy_radiated_total += curr_mode_energy_radiated

        return time, energy_radiated_total

    def get_dP_dt_radiated(self, **kwargs) -> tuple:
        """Rate at which linear momentum is radiated

        Rate at which linear momentum is radiated through gravitational waves, computed using the method described in
        https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it computes the total sum of all modes.

        Args:
            **kwargs : parameters to specify the modes you would like to sum over

                Valid keyword arguments are:

                lmin (:obj:`int`, optional): minumum value of l range

                lmax (:obj:`int`, optional): maximum value of l range

                l (:obj:`int`, optional): specific l value

                m (:obj:`int`, optional): specific m value

        Returns:
            tuple: time, :math:`dP/dt` for given modes.

        """
        if 'm' in kwargs and 'l' not in kwargs:
            warnings.warn("Unclear input. Only an m requested without a corresponding l")
            return

        if 'l' in kwargs and ('lmin' in kwargs or 'lmax' in kwargs):
            warnings.warn("Unclear input. Asking for both a specific l and an l range. Please request either a "
                          "specific l, an l range, or neither.")
            return

        if 'l' in kwargs and 'm' in kwargs:
            l = kwargs['l']
            m = kwargs['m']
            if (l, m) not in self.modes:
                warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
                return
            return self.time, self.modes[(l, m)].dP_dt_radiated

        lmin = 2
        lmax = self.l_max

        if 'l' in kwargs and 'm' not in kwargs:
            lmin = kwargs['l']
            lmax = kwargs['l']
        if 'lmin' in kwargs:
            lmin = kwargs['lmin']
        if 'lmax' in kwargs:
            lmax = kwargs['lmax']

        dP_dt_total = np.zeros((len(self.time), 3))  # + 1j * np.zeros((len(self.time), 3))
        for l in range(lmin, lmax + 1):
            for m in range(-l, l + 1):
                dP_dt_total += self.modes[(l, m)].dP_dt_radiated

        return self.time, dP_dt_total

    def get_linear_momentum_radiated(self, **kwargs) -> tuple:
        """Linear momentum radiated

        Linear momentum radiated through gravitational waves computed using the method described in
        https://arxiv.org/abs/0707.4654. If no lmin, lmax, l, or m are provided, it computes the total sum of all modes.

        Args:
            **kwargs : parameters to specify the modes you would like to sum over

                Valid keyword arguments are:

                lmin (:obj:`int`, optional): minumum value of l range

                lmax (:obj:`int`, optional): maximum value of l range

                l (:obj:`int`, optional): specific l value

                m (:obj:`int`, optional): specific m value

        Returns:
            tuple: time, cumulative linear momentum radiated for given modes

        """
        if 'm' in kwargs and 'l' not in kwargs:
            warnings.warn("Unclear input. Only an m requested without a corresponding l")
            return

        if 'l' in kwargs and ('lmin' in kwargs or 'lmax' in kwargs):
            warnings.warn("Unclear input. Asking for both a specific l and an l range. Please request either a "
                          "specific l, an l range, or neither.")
            return

        if 'l' in kwargs and 'm' in kwargs:
            l = kwargs['l']
            m = kwargs['m']
            if (l, m) not in self.modes:
                warnings.warn("There is no l={l}, m={m} mode for this radiation sphere".format(l=l, m=m))
                return
            return self.time, self.modes[(l, m)].linear_momentum_radiated

        lmin = 2
        lmax = self.l_max

        if 'l' in kwargs and 'm' not in kwargs:
            lmin = kwargs['l']
            lmax = kwargs['l']
        if 'lmin' in kwargs:
            lmin = kwargs['lmin']
        if 'lmax' in kwargs:
            lmax = kwargs['lmax']

        linear_momentum_total = np.zeros((len(self.time), 3))  # + 1j * np.zeros((len(self.time), 3))
        for l in range(lmin, lmax + 1):
            for m in range(-l, l + 1):
                if (l, m) not in self.modes:
                    warnings.warn(f'({l}, {m}) information not available')
                    continue
                linear_momentum_total += self.modes[(l, m)].linear_momentum_radiated

        return self.time, linear_momentum_total

    def get_extrapolated_sphere(self, order: int = 1):
        """RadiationSphere object extrapolated from this RadiationSphere to infinite radius.

        Extrapolate the :math:`\Psi_4` of all RadiationModes in this RadiationSphere to infinite radius and create and
        return a new RadiationSphere with those extrapolated RadiationModes. Uses the method described in
        https://arxiv.org/abs/1008.4360 and https://arxiv.org/abs/1108.4421.

        Args:
            order (:obj:`int`, optional): the extrapolation order. Defaults to 1.

        Returns:
            RadiationSphere: A RadiationSphere with the same properties as this one but with :math:`\Psi_4` extrapolated
            to infinite radius.

        """
        if self.extrapolated:
            return self

        temp_modes = {}
        for mode_tuple, radiation_mode in self.raw_modes.items():
            extrap_mode = radiation_mode.get_mode_with_extrapolated_radius(order=order)
            if extrap_mode is not None:
                temp_modes[mode_tuple] = extrap_mode

        if len(temp_modes.keys()) == 0:
            return None
        extrapolated_sphere = RadiationSphere(mode_dict=temp_modes, time=np.array(self.__time), radius=self.radius,
                                              extrapolated=True)
        if self.frame != Frame.RAW:
            import sys
            if sys.version_info.major == 3 and sys.version_info.minor > 10:
                raise ImportError(
                    'Unable to set center-of-mass corrected frame. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

            extrapolated_sphere.set_frame(self.frame, alpha=self.__alpha, beta=self.__beta)

        return extrapolated_sphere

    def _scri_waveform_modes_object(self):
        """Create and return a Scri WaveformModes object containing the data for this extraction sphere.

        For more information on scri objects, refer to https://scri.readthedocs.io.

        Returns: A Scri WaveformModes object

        """
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            raise ImportError(
                'Unable to create scri waveform modes object. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

        from scri import WaveformModes
        from scri import h as scri_h
        from scri import Inertial

        l_min = 2
        l_max = self.l_max

        for l in range(l_min, l_max + 1):
            for m in range(-l, l + 1):
                if (l, m) not in self.raw_modes:
                    return None

        def mode_data(l, m):
            mode = self.raw_modes[(l, m)]
            mode_content = np.array(mode.strain_plus - 1j * mode.strain_cross)
            return mode_content

        data = np.array([mode_data(l, m) for l in range(l_min, l_max + 1) for m in range(-l, l + 1)]).T

        cut_index = np.argmax(self.__time > 150)

        h = WaveformModes(
            t=self.__time[cut_index:],
            data=data[cut_index:],
            ell_min=l_min,
            ell_max=l_max,
            dataType=scri_h,
            frameType=Inertial,
            r_is_scaled_out=True,
            m_is_scaled_out=True
        )
        return h

    def _set_alpha_beta_for_com_transformation(self, com_time, center_of_mass):
        """Set the offset and boost for center of mass drift correction

        Args:
            com_time (np.ndarray): time stamps for the center of mass data
            center_of_mass (np.ndarray): timeseries of center of mass

        """
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            raise ImportError(
                'Unable to convert to center-of-mass corrected frame. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

        com_time = com_time + self.radius
        t_max = np.max(com_time)
        ti = 0.1 * t_max
        tf = 0.9 * t_max

        idx_min = np.argmax(com_time >= ti)
        idx_max = np.argmax(com_time >= tf)
        t_com_maya_int, com_maya_int = com_time[idx_min:idx_max + 1], center_of_mass[idx_min: idx_max + 1, :]

        x0 = np.zeros(3)
        xt0 = np.zeros(3)
        for i in range(3):
            x0[i] = scipy.integrate.trapz(com_maya_int[:, i], t_com_maya_int) / (tf - ti)
            xt0[i] = scipy.integrate.trapz(com_maya_int[:, i] * t_com_maya_int, t_com_maya_int) / (tf - ti)

        # calculate alpha and beta with the Newtonian approach
        self.__alpha = (4 * (tf ** 2 + tf * ti + ti * 2) * x0 - 6 * (tf + ti) * xt0) / (tf - ti) ** 2
        self.__beta = (12 * xt0 - 6 * (tf + ti) * x0) / (tf - ti) ** 2

    def _generate_com_corrected_modes(self):
        """Create a dictionary of RadiationModes in a frame which has corrected for center of mass drift.

        Uses scri to perform the transformation. For more information on scri, refer to https://scri.readthedocs.io.
        """
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            raise ImportError(
                'Unable to return modes in center-of-mass corrected frame. Python version too recent to be compatible with Scri package. If you would like the ability to move to center-of-mass corrected frame, use python <= 3.10.')

        from spherical_functions import LM_index

        scri_waveform_object = self._scri_waveform_modes_object()
        if scri_waveform_object is None:
            return

        scri_waveform_object = scri_waveform_object.transform(space_translation=self.__alpha,
                                                              boost_velocity=self.__beta)
        time = scri_waveform_object.t
        self.__com_corrected_time = time

        self.__com_corrected_modes = {}

        for l in range(2, self.l_max + 1):
            for m in range(-l, l + 1):
                data_lm = scri_waveform_object.data[:, LM_index(l, m, scri_waveform_object.ell_min)]
                strain_plus = np.real(data_lm)
                strain_cross = -1 * np.imag(data_lm)
                self.__com_corrected_modes[(l, m)] = RadiationMode(strain_plus=strain_plus, strain_cross=strain_cross,
                                                                   l=l,
                                                                   m=m,
                                                                   rad=self.radius, time=np.array(time),
                                                                   extrapolated=self.extrapolated,
                                                                   center_of_mass_corrected=True)


class RadiationMode:
    """Class for interacting with a single mode of radiation information when expressed in terms of spin weighted
    spherical harmonics."""

    def __init__(self, l: int, m: int, rad: float, time: np.ndarray, psi4_group: h5py.Group = None,
                 extrapolated: bool = False, radiation_sphere: RadiationSphere = None, psi4_real: np.ndarray = None,
                 psi4_imaginary: np.ndarray = None, strain_plus: np.ndarray = None, strain_cross: np.ndarray = None,
                 center_of_mass_corrected: bool = False):
        self.__l_value = l
        self.__m_value = m
        self.__radius = rad
        self.__time = time
        self.__extrapolated = extrapolated
        self.__psi4_group = psi4_group
        self.__radiation_sphere = radiation_sphere
        self.__psi4_amplitude = None
        self.__psi4_phase = None
        self.__strain_amplitude = None
        self.__strain_phase = None
        self.__strain_plus = strain_plus
        self.__strain_cross = strain_cross
        self.__psi4_real = psi4_real
        self.__psi4_imaginary = psi4_imaginary
        self.__center_of_mass_corrected = center_of_mass_corrected

    @staticmethod
    def create_radiation_mode(psi4_group: h5py.Group = None, l: int = 0, m: int = 0, rad: float = 0,
                              time: np.ndarray = None, extrapolated: bool = False,
                              radiation_sphere: RadiationSphere = None):
        """Create a RadiationMode from psi4 group.

        Args:
            psi4_group (:obj:`h5py.Group`, optional): group containing all the data pertaining to :math:`\Psi_4` for the
                given mode
            l (:obj:`int`, optional): value of l for the spherical harmonic mode. Defaults to 0.
            m (:obj:`int`, optional): value of m for the spherical harmonic mode. Defaults to 0.
            rad (:obj:`float`, optional): radius at which the given mode was extracted. Defaults to 0.
            time (:obj:`numpy.ndarray`, optional): array containing the time value associated with each data point of
                :math:`\Psi_4`. Defaults to empty array.
            extrapolated (:obj:`bool`, optional): whether the data has been extrapolated to infinite radius. Defaults to
                false.
            radiation_sphere (:obj:`RadiationSphere`, optional): the RadiationSphere this mode is associated with.
                Defaults to None.

        Returns:
            RadiationMode: The constructed RadiationMode

        """

        if time is None:
            return None

        if psi4_group is None:
            return None

        if 'imaginary' not in psi4_group or 'real' not in psi4_group:
            return None

        return RadiationMode(psi4_group=psi4_group, l=l, m=m, rad=rad, time=time,
                             extrapolated=extrapolated, radiation_sphere=radiation_sphere)

    @property
    def l_value(self) -> int:
        """l value in the (l, m) pair that denotes a mode of the spherical harmonic decomposition."""
        return self.__l_value

    @property
    def m_value(self) -> int:
        """m value in the (l, m) pair that denotes a mode of the spherical harmonic decomposition."""
        return self.__m_value

    @property
    def radius(self) -> float:
        """Radius at which :math:`\Psi_4` for this mode was extracted."""
        return self.__radius

    @property
    def time(self) -> np.ndarray:
        """Array of time stamps associated with the :math:`\Psi_4` and strain data."""
        return self.__time

    @property
    def extrapolated(self) -> bool:
        """Whether the data within this mode has been extrapolated to infinite radius."""
        return self.__extrapolated

    @property
    def psi4_real(self) -> np.ndarray:
        """The real part of the Weyl Scalar (:math:`\Psi_4`) as a timeseries."""
        if self.__psi4_group is not None:
            return self.__psi4_group["real"][()]
        elif self.__psi4_real is not None:
            return self.__psi4_real
        elif self.__strain_plus is not None:
            # compute psi4
            strain_plus = self.__strain_plus
            time = self.time
            psi4_real = np.gradient(np.gradient(strain_plus, time), time) / self.radius
            self.__psi4_real = psi4_real
            return psi4_real
        else:
            return None

    @property
    def psi4_imaginary(self) -> np.ndarray:
        """The imaginary part of the Weyl Scalar (:math:`\Psi_4`) as a timeseries."""
        if self.__psi4_group is not None:
            return self.__psi4_group["imaginary"][()]
        elif self.__psi4_imaginary is not None:
            return self.__psi4_imaginary
        elif self.__strain_cross is not None:
            # compute psi4
            strain_cross = self.__strain_cross
            time = self.time
            psi4_imaginary = -1 * np.gradient(np.gradient(strain_cross, time), time) / self.radius
            self.__psi4_imaginary = psi4_imaginary
            return psi4_imaginary
        else:
            return None

    @property
    def psi4_amplitude(self) -> np.ndarray:
        """The amplitude of the Weyl Scalar (:math:`\Psi_4`) as a timeseries, computed from the real and imaginary
        parts as :math:`\sqrt{\mathcal{Re}(\Psi_4)^2 + \mathcal{Im}(\Psi_4)^2}`."""
        if self.__psi4_amplitude is None:
            self.__psi4_amplitude = np.sqrt(np.power(self.psi4_real, 2) + np.power(self.psi4_imaginary, 2))
        return self.__psi4_amplitude

    @property
    def psi4_phase(self) -> np.ndarray:
        """The phase of the Weyl Scalar (:math:`\Psi_4`) as a timeseries, computed from the real and imaginary parts as
        :math:`\\textrm{tan}^{-1}( \mathcal{Im}(\Psi_4) / \mathcal{Re}(\Psi_4) )`."""
        if self.__psi4_phase is None:
            self.__psi4_phase = -1 * np.unwrap(np.arctan2(self.psi4_imaginary, self.psi4_real))
        return self.__psi4_phase

    @property
    def strain_plus(self) -> np.ndarray:
        """The plus component of :math:`rh` as a timeseries. The strain is the second time integral of
        :math:`\Psi_4` computed using fixed-frequency integration."""
        if self.__strain_plus is None:
            self.compute_and_store_strain()
        return self.__strain_plus

    @property
    def strain_cross(self) -> np.ndarray:
        """The cross component of :math:`rh` as a timeseries. The strain is the second time integral of
        :math:`\Psi_4` computed using fixed-frequency integration."""
        if self.__strain_cross is None:
            self.compute_and_store_strain()
        return self.__strain_cross

    @property
    def strain_amplitude(self) -> np.ndarray:
        """The amplitude of :math:`rh` as a timeseries, computed from the real and imaginary parts as
        :math:`\sqrt{\mathcal{Re}(rh)^2 + \mathcal{Im}(rh)^2}`. The strain is the second time integral of
        :math:`\Psi_4` computed using fixed-frequency integration."""
        if self.__strain_amplitude is None:
            self.__strain_amplitude = np.sqrt(np.power(self.strain_plus, 2) + np.power(self.strain_cross, 2))
        return self.__strain_amplitude

    @property
    def strain_phase(self) -> np.ndarray:
        """The phase of :math:`rh` as a timeseries, computed from the real and imaginary parts as
        :math:`\\textrm{tan}^{-1}( \mathcal{Im}(rh) / \mathcal{Re}(rh) )`. The strain is the second time integral of
        :math:`\Psi_4` computed using fixed-frequency integration."""
        if self.__strain_phase is None:
            self.__strain_phase = -1 * np.unwrap(np.arctan2(self.strain_cross, self.strain_plus))
        return self.__strain_phase

    @property
    def radiation_sphere(self) -> RadiationSphere:
        """The RadiationSphere this mode is associated with."""
        return self.__radiation_sphere

    @radiation_sphere.setter
    def radiation_sphere(self, radiation_sphere: RadiationSphere):
        """Set the radiation sphere this mode is associated with."""
        self.__radiation_sphere = radiation_sphere

    @property
    def psi4_omega(self):
        """The frequency of :math:`\Psi_4` computed as the time derivative of the phase."""
        return np.gradient(self.psi4_phase, self.time)

    @property
    def omega_start(self):
        """The starting frequency of :math:`\Psi_4` where the frequency is computed as the time derivative of the
        phase."""
        start_time = self.time[0]
        end_time = start_time + 30
        if end_time > self.time[-1]:
            warnings.warn('There is not sufficient data to compute the starting frequency.')
            return None
        filter_size = np.argmax(self.time > end_time).squeeze()
        omega_rolling_average = uniform_filter1d(self.psi4_omega, size=filter_size)

        desired_time = self.radius + 75
        if desired_time > self.time[-1]:
            desired_time = 0

        return abs(omega_rolling_average[np.argmax(self.time > desired_time)])

    @property
    def psi4_max_time(self):
        """The time at which the amplitude of :math:`\Psi_4` is at a maximum"""
        return self.time[np.argmax(self.psi4_amplitude)]

    @property
    def h_plus_dot(self):
        """The real part of the first time derivative of the strain."""
        return np.gradient(self.strain_plus, self.time)

    @property
    def h_cross_dot(self):
        """The imaginary part of the first time derivative of the strain."""
        return np.gradient(self.strain_cross, self.time)

    @property
    def dEnergy_dt_radiated(self) -> np.ndarray:
        """Time derivative of the energy radiated.

        https://arxiv.org/pdf/0707.4654.pdf
        eq 3.8

        Returns:
            numpy.ndarray: :math:`dE/dt` radiated

        """
        hdot_complex_mag_squared = self.h_plus_dot ** 2 + self.h_cross_dot ** 2
        energydot_radiated = hdot_complex_mag_squared / (16.0 * np.pi)
        return energydot_radiated

    @property
    def energy_radiated(self) -> np.ndarray:
        """Energy radiated, :math:`E`.

        https://arxiv.org/pdf/0707.4654.pdf
        eq 3.8

        Returns:
            numpy.ndarray: cumulative :math:`E` radiated

        """
        energy_dot = self.dEnergy_dt_radiated
        energy_radiated = scipy.integrate.cumulative_trapezoid(energy_dot, self.time, initial=0)
        return energy_radiated

    @property
    def dP_dt_radiated(self) -> np.ndarray:
        """Rate at which linear momentum is radiated as gravitational waves

        https://arxiv.org/pdf/0707.4654.pdf
        eqs 3.14, 3.15

        Returns:
            numpy.ndarray: :math:`dP/dt` radiated

        """

        def a_lm(l, m):
            alm = ((l - m) * (l + m + 1)) ** 0.5 / (l * (l + 1))
            return alm

        def b_lm(l, m):
            blm = 0.5 * (1 / l) * ((l - 2) * (l + 2) * (l + m) * (l + m - 1) * (1 / ((2 * l - 1) * (2 * l + 1)))) ** 0.5
            return blm

        def c_lm(l, m):
            clm = 2.0 * m * (1 / (l * (l + 1)))
            return clm

        def d_lm(l, m):
            dlm = (1 / l) * ((l - 2) * (l + 2) * (l - m) * (l + m) * (1 / ((2 * l - 1) * (2 * l + 1)))) ** 0.5
            return dlm

        A_integral = self.h_plus_dot - 1j * self.h_cross_dot

        def A_integral_conj(new_l, new_m):
            if (new_l, new_m) not in self.radiation_sphere.modes:
                return np.zeros(self.time.shape) - 1j * np.zeros(self.time.shape)
            curr_h_plus_dot = self.radiation_sphere.modes[(new_l, new_m)].h_plus_dot
            curr_h_cross_dot = self.radiation_sphere.modes[(new_l, new_m)].h_cross_dot
            curr_hdot_complex = curr_h_plus_dot - 1j * curr_h_cross_dot
            curr_hdot_conj = np.conj(curr_hdot_complex)
            return curr_hdot_conj

        Pdot_plus_term1 = a_lm(self.l_value, self.m_value) * A_integral_conj(self.l_value, self.m_value + 1)
        Pdot_plus_term2 = b_lm(self.l_value, -self.m_value) * A_integral_conj(self.l_value - 1, self.m_value + 1)
        Pdot_plus_term3 = b_lm(self.l_value + 1, self.m_value + 1) * A_integral_conj(self.l_value + 1, self.m_value + 1)

        # negative one comes from the negative in equation 3.3
        Pdot_plus = -1 * A_integral * (Pdot_plus_term1 + Pdot_plus_term2 - Pdot_plus_term3) / (8 * np.pi)

        Px_dot = np.real(Pdot_plus)
        Py_dot = np.imag(Pdot_plus)

        Pz_dot_term1 = c_lm(self.l_value, self.m_value) * A_integral_conj(self.l_value, self.m_value)
        Pz_dot_term2 = d_lm(self.l_value, self.m_value) * A_integral_conj(self.l_value - 1, self.m_value)
        Pz_dot_term3 = d_lm(self.l_value + 1, self.m_value) * A_integral_conj(self.l_value + 1, self.m_value)

        # negative one comes from the negative in equation 3.3
        Pz_dot = np.real(-1 * A_integral * (Pz_dot_term1 + Pz_dot_term2 + Pz_dot_term3) / (16 * np.pi))

        return np.column_stack([Px_dot, Py_dot, Pz_dot])

    @property
    def linear_momentum_radiated(self) -> np.ndarray:
        """Linear momentum radiated as gravitational waves

        https://arxiv.org/pdf/0707.4654.pdf
        eqs 3.14, 3.15

        Returns:
            numpy.ndarray: linear momentum radiated

        """
        Pdot = self.dP_dt_radiated
        linear_momentum_radiated = scipy.integrate.cumulative_trapezoid(Pdot, self.time, initial=0, axis=0)
        return linear_momentum_radiated

    def compute_and_store_strain(self):
        """Computes and stores the strain data from the :math:`\Psi_4` data.

        Computes the strain using the fixed-frequency integration method to compute the double time integral of
        :math:`\Psi_4` using the fourier transform.

        """
        if self.l_value != 2 or self.m_value != 2:
            if self.radiation_sphere is None:
                warnings.warn(
                    "This mode is not l=2, m=2 and does not have an radiation sphere. Cannot compute necessary \
                    quantities to compute strain.")
                return
            if (2, 2) not in self.radiation_sphere.modes:
                warnings.warn(
                    "The associated radiation sphere does not have an l=2, m=2 mode. Cannot compute necessary \
                    quantities to compute strain.")
                return

        if len(self.time) < 2:
            warnings.warn('Data is not long enough to compute strain')
            return

        if self.l_value == 2 and self.m_value == 2:
            omega_22_start = self.omega_start
            if omega_22_start is None:
                warnings.warn('Unable to compute the starting frequncy and therefore cannot compute the strain.')
                return
            max_22_time = self.psi4_max_time
        else:
            omega_22_start = self.radiation_sphere.modes[(2, 2)].omega_start
            if omega_22_start is None:
                warnings.warn('Unable to compute the starting frequncy and therefore cannot compute the strain.')
                return
            max_22_time = self.radiation_sphere.modes[(2, 2)].psi4_max_time

        timestep = self.time[1] - self.time[0]
        length = len(self.time)

        # compute fmin
        if self.m_value == 0:
            fmin = 0.25 * omega_22_start / (2 * np.pi)
        else:
            fmin = 0.75 * (abs(self.m_value) / 2) * omega_22_start / (2 * np.pi)

        # psi4f from psi4t
        psi4_f_real = np.fft.rfft(self.psi4_real, length)
        psi4_f_imaginary = np.fft.rfft(self.psi4_imaginary, length)
        frequency = np.fft.rfftfreq(length, d=timestep)

        frequency[abs(frequency) < fmin] = fmin

        # hf from psi4_f
        factor = 4 * np.pi ** 2 * np.power(frequency, 2)
        hf_plus = np.divide(psi4_f_real, -1 * factor)
        hf_cross = np.divide(psi4_f_imaginary, factor)

        # ht from hf
        ht_plus = np.fft.irfft(hf_plus, length)
        ht_cross = np.fft.irfft(hf_cross, length)

        if self.extrapolated:
            strain_plus = ht_plus
            strain_cross = ht_cross
        else:
            strain_plus = ht_plus * self.radius
            strain_cross = ht_cross * self.radius

        # window strain
        window_length_in_time = 30
        start_window_after_max = 120

        start_window_time = max_22_time + start_window_after_max
        end_window_time = start_window_time + window_length_in_time

        if self.time[-1] < start_window_time:
            warnings.warn("Not enough time after max to window")
            self.__strain_plus = strain_plus
            self.__strain_cross = strain_cross
            return

        end_window_time = min(end_window_time, self.time[-2])

        start_window_index = np.argmax(self.time > start_window_time)
        end_window_index = np.argmax(self.time > end_window_time)

        window = np.zeros(self.time.shape)

        # set to 1's before the windowing
        window[:start_window_index] = np.ones(start_window_index)

        # set the windowing
        window_width = end_window_index - start_window_index
        blackmanharris_window_symmetric = blackmanharris(2 * window_width)
        blackmanharris_window_half = blackmanharris_window_symmetric[window_width:]
        window[start_window_index:end_window_index] = blackmanharris_window_half

        # smooth window with low pass filter
        step_size = self.time[1] - self.time[0]
        fs = 1 / step_size
        cutoff = 1
        Wn = 2 * (cutoff / (2 * np.pi)) / fs

        b, a = butter(4, Wn, analog=False)
        smoothed_window = filtfilt(b, a, window, axis=0)

        strain_plus = strain_plus * smoothed_window
        strain_cross = strain_cross * smoothed_window

        self.__strain_plus = strain_plus
        self.__strain_cross = strain_cross

    def get_mode_with_extrapolated_radius(self, order: int = 1):
        """RadiationMode object extrapolated from this RadiationMode to infinite radius.

        Extrapolate the :math:`\Psi_4` of this RadiationMode to infinite radius and create and return a new
        RadiationMode with that :math:`\Psi_4` data. Uses the method described in
        https://arxiv.org/abs/1008.4360 and https://arxiv.org/abs/1108.4421.

        Args:
            order (:obj:`int`, optional): the extrapolation order. Defaults to 1.

        Returns:
            RadiationMode: A RadiationMode with the same properties as this one but with :math:`\Psi_4` extrapolated to
            infinite radius.

        """
        if self.extrapolated:
            return self

        M_adm = 1

        # multiple psi4 by radius
        psi4_real_times_radius = self.psi4_real * self.radius
        psi4_imaginary_times_radius = self.psi4_imaginary * self.radius

        factor = 1

        psi4_complex = psi4_real_times_radius + 1j * psi4_imaginary_times_radius
        if self.strain_plus is None or self.strain_cross is None:
            warnings.warn('Cannot create extrapolated sphere due to inability to compute strain.')
            return

        strain_complex = self.strain_plus - 1j * self.strain_cross
        hdot_complex = self.h_plus_dot - 1j * self.h_cross_dot

        psi4_inf = self.radius * psi4_complex - (self.l_value - 1) * (self.l_value + 2) / (2 * self.radius) * np.conj(
            hdot_complex)

        if order == 2:
            psi4_inf = psi4_inf + (self.l_value - 1) * (self.l_value + 2) * (self.l_value ** 2 + self.l_value - 4) / (
                    8 * self.radius ** 2) * np.conj(strain_complex) - 3 * M_adm / (
                               2 * self.radius ** 2) * np.conj(hdot_complex)

        psi4_inf = factor * psi4_inf

        psi4_inf_real = psi4_inf.real / self.radius
        psi4_inf_imaginary = psi4_inf.imag / self.radius

        return RadiationMode(psi4_real=psi4_inf_real, psi4_imaginary=psi4_inf_imaginary, l=self.l_value, m=self.m_value,
                             rad=self.radius, time=np.array(self.time), extrapolated=True)

    @staticmethod
    def ylm(l, m, theta, phi) -> float:
        """Spherical harmonic for l, m at given angles theta and phi
        
        Args:
            l: l value of mode
            m: m value of mode
            theta: angle defined from the +z-axis
            phi: angle defined in the x-y plane moving counterclockwise from the +x-axis

        Returns:
            float: The spherical harmonic value at the given l, m, theta, phi

        """
        k1 = max(0, m - 2)
        k2 = min(l + m, l - 2)
        d_lm = 0 + 0j
        for k in range(k1, k2 + 1):
            cos_theta_over_2 = np.cos(theta / 2)
            sin_theta_over_2 = np.sin(theta / 2)
            cos_exponent = 2 * l + m - 2 * k - 2
            sin_exponent = 2 * k - m + 2

            if cos_theta_over_2 == 0 and cos_exponent == 0:
                cos_term = 1
            else:
                cos_term = cos_theta_over_2 ** cos_exponent
            if sin_theta_over_2 == 0 and sin_exponent == 0:
                sin_term = 1
            else:
                sin_term = sin_theta_over_2 ** sin_exponent

            first_part = (-1) ** k / (math.factorial(k))
            second_part_numerator = np.sqrt(
                float(math.factorial(l + m) * math.factorial(l - m) * math.factorial(l + 2) * math.factorial(l - 2)))
            second_part_denominator = math.factorial(k - m + 2) * math.factorial(l + m - k) * math.factorial(l - k - 2)
            second_part = second_part_numerator / second_part_denominator
            third_part = cos_term * sin_term
            d_lm_k = first_part * second_part * third_part
            d_lm += d_lm_k

        Y_lm = np.sqrt((2 * l + 1) / (4 * np.pi)) * d_lm * np.exp(1j * m * phi)

        return Y_lm
