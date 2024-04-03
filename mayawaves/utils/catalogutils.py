import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
import os
import wget
import datetime as dt


class Parameter(Enum):
    """Enum for easily referencing parameters of the simulations."""

    NAME = 'name'
    MASS_1 = 'm1'
    MASS_2 = 'm2'
    IRREDUCIBLE_MASS_1 = 'm1_irr'
    IRREDUCIBLE_MASS_2 = 'm2_irr'
    MASS_RATIO = 'q'
    SYMMETRIC_MASS_RATIO = 'eta'
    DIMENSIONLESS_SPIN_1 = 'a1'
    DIMENSIONLESS_SPIN_X_1 = 'a1x'
    DIMENSIONLESS_SPIN_Y_1 = 'a1y'
    DIMENSIONLESS_SPIN_Z_1 = 'a1z'
    DIMENSIONLESS_SPIN_2 = 'a2'
    DIMENSIONLESS_SPIN_X_2 = 'a2x'
    DIMENSIONLESS_SPIN_Y_2 = 'a2y'
    DIMENSIONLESS_SPIN_Z_2 = 'a2z'
    CHI_EFF = 'chi_eff'
    CHI_P = 'chi_p'
    F_LOWER_AT_1MSUN = 'f_lower_at_1MSUN'  # after junk
    SEPARATION = 'separation'
    ECCENTRICITY = 'eccentricity'
    MEAN_ANOMALY = 'mean_anomaly'
    MERGE_TIME = 'merge_time'
    MAYA_SIZE_GB = 'maya file size (GB)'
    LVCNR_SIZE_GB = 'lvcnr file size (GB)'


class Catalog:
    """Class for interacting with the MAYA catalog."""

    def __init__(self):
        # Finding the path to metadata file, file name is hardcoded!
        CURR_DIR = os.path.dirname(__file__)
        path = os.path.join(CURR_DIR, "MAYAmetadata.pkl")

        if not os.path.exists(path):

            print(
                "MAYAmetadata.pkl does not exist in this directory. Attempting to download it.")
            try:
                wget.download('https://cgpstorage.ph.utexas.edu/MAYAmetadata.pkl', out=CURR_DIR)
                print("Downloaded MAYAmetadata.pkl")
            except:
                print("Unable to download MAYAmetadata.pkl. Exiting.")
                return

        modified_time = dt.datetime.fromtimestamp(os.path.getmtime(path))
        current_time = dt.datetime.now()

        if (current_time - dt.timedelta(weeks=1)) > modified_time:
            print(
                "MAYAmetadata.pkl is more than a week old. Attempting to download an updated version.")
            try:
                os.rename(os.path.join(CURR_DIR, 'MAYAmetadata.pkl'), os.path.join(CURR_DIR, 'MAYAmetadata_old.pkl'))
                wget.download('https://cgpstorage.ph.utexas.edu/MAYAmetadata.pkl', out=CURR_DIR)
                os.remove(os.path.join(CURR_DIR, 'MAYAmetadata_old.pkl'))
                print("Downloaded MAYAmetadata.pkl")
            except:
                warnings.warn('Unable to download MAYAmetadata.pkl. Catalog data may be out of date.')
                os.rename(os.path.join(CURR_DIR, 'MAYAmetadata_old.pkl'), os.path.join(CURR_DIR, 'MAYAmetadata.pkl'))

        # read metadata
        self.df = pd.read_pickle(path)
        self.df[Parameter.DIMENSIONLESS_SPIN_1.value] = np.linalg.norm(self.df[[Parameter.DIMENSIONLESS_SPIN_X_1.value,
                                                                                Parameter.DIMENSIONLESS_SPIN_Y_1.value,
                                                                                Parameter.DIMENSIONLESS_SPIN_Z_1.value]],
                                                                       axis=1)
        self.df[Parameter.DIMENSIONLESS_SPIN_2.value] = np.linalg.norm(self.df[[Parameter.DIMENSIONLESS_SPIN_X_2.value,
                                                                                Parameter.DIMENSIONLESS_SPIN_Y_2.value,
                                                                                Parameter.DIMENSIONLESS_SPIN_Z_2.value]],
                                                                       axis=1)
        self.df[Parameter.CHI_EFF.value] = (self.df[Parameter.MASS_RATIO.value] * self.df[
            Parameter.DIMENSIONLESS_SPIN_Z_1.value] + self.df[Parameter.DIMENSIONLESS_SPIN_Z_2.value]) / (
                                                   1 + self.df[Parameter.MASS_RATIO.value])

        xy_mag_spin_1 = np.linalg.norm(self.df[[Parameter.DIMENSIONLESS_SPIN_X_1.value, Parameter.DIMENSIONLESS_SPIN_Y_1.value]], axis=1)
        xy_mag_spin_2 = np.linalg.norm(self.df[[Parameter.DIMENSIONLESS_SPIN_X_2.value, Parameter.DIMENSIONLESS_SPIN_Y_2.value]], axis=1)
        q = self.df[Parameter.MASS_RATIO.value]
        self.df[Parameter.CHI_P.value] = np.maximum(xy_mag_spin_1, ((4 / q + 3) / (4 * q + 3)) * xy_mag_spin_2)

        self.simulations = list(self.df.index)  # available waveforms

    @property
    def nonspinning_simulations(self):
        """List of all simulations/waveforms with no spin."""
        nonspinning_simulations = self.df[(np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_1.value]) < 1e-3) &
                                          (np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_2.value]) < 1e-3)]
        return list(nonspinning_simulations.index)

    @property
    def aligned_spin_simulations(self):
        """List of all simulations/waveforms with spins aligned or anti-aligned with the orbital angular momentum."""
        aligned_spin_simulations = self.df[(np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_X_1.value]) < 1e-3) &
                                           (np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_Y_1.value]) < 1e-3) &
                                           (np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_X_2.value]) < 1e-3) &
                                           (np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_Y_2.value]) < 1e-3) &
                                           ((np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_1.value]) >= 1e-3) | (
                                                   np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_2.value]) >= 1e-3))]
        return list(aligned_spin_simulations.index)

    @property
    def precessing_simulations(self):
        """List of all simulations/waveforms with precessing spins (spins not aligned with the orbital angular
        momentum)."""
        precessing_spin_simulations = self.df[(np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_X_1.value]) >= 1e-3) |
                                              (np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_Y_1.value]) >= 1e-3) |
                                              (np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_X_2.value]) >= 1e-3) |
                                              (np.abs(self.df[Parameter.DIMENSIONLESS_SPIN_Y_2.value]) >= 1e-3)]
        return list(precessing_spin_simulations.index)

    def _check_input(self, param: Parameter, value: float) -> bool:
        """Checks if the value provided for a parameter is valid.

        Args:
            param (Parameter): simulation parameter to check
            value (float): corresponding value

        Returns:
            bool: Whether the value falls within the range of the parameter

        """

        if param == Parameter.MASS_RATIO:
            valid = (value >= 1)
            if not valid:
                warnings.warn("q needs to be greater or equal to 1.")
            return valid

        if param == Parameter.SYMMETRIC_MASS_RATIO:
            valid = (0 < value <= 0.25)
            if not valid:
                warnings.warn("Symmetric mass ratio needs to be between 0 and 0.25.")
            return valid

        if param == Parameter.DIMENSIONLESS_SPIN_X_1 or \
                param == Parameter.DIMENSIONLESS_SPIN_Y_1 or \
                param == Parameter.DIMENSIONLESS_SPIN_Z_1 or \
                param == Parameter.DIMENSIONLESS_SPIN_X_2 or \
                param == Parameter.DIMENSIONLESS_SPIN_Y_2 or \
                param == Parameter.DIMENSIONLESS_SPIN_Z_2:
            valid = (0 <= np.abs(value) <= 1)
            if not valid:
                warnings.warn("Individual spin components need to be between -1 and 1.")
            return valid

        if param == Parameter.MASS_1 or \
                param == Parameter.MASS_2 or \
                param == Parameter.IRREDUCIBLE_MASS_1 or \
                param == Parameter.IRREDUCIBLE_MASS_2:

            valid = (0 <= value <= 1.0)
            if not valid:
                warnings.warn("Christodoulou and irreducible mass for each object need to be between 0 and 1.")
            return valid

        if param == Parameter.F_LOWER_AT_1MSUN or \
                param == Parameter.SEPARATION or \
                param == Parameter.MAYA_SIZE_GB or \
                param == Parameter.LVCNR_SIZE_GB:
            valid = (value >= 0)
            if not valid:
                warnings.warn("Value needs to be greater than 0.")
            return valid

        if param == Parameter.ECCENTRICITY:
            valid = (0 <= value <= 1)
            if not valid:
                warnings.warn("Eccentricity needs to be between 0 and 1.")
            return valid

        return True

    def _print_parameters(self, catalog_id: str):
        """ Prints a simulation/waveform's parameters and their corresponding values.

        Args:
            catalog_id (str): simulation/waveform name

        """
        print(f"Catalog id: {catalog_id}")
        for param in Parameter:
            if type(self.df.loc[catalog_id][param.value]) == str:
                print('{0:20}  {1}'.format(param.value, self.df.loc[catalog_id][param.value]))
            else:
                print('{0:20}  {1}'.format(param.value, format(self.df.loc[catalog_id][param.value], "0.4f")))

    def spin_magnitudes_for_simulation(self, name: str) -> tuple:
        """Calculates spin magnitude for each object in the binary system and returns a tuple with individual spin
        magnitudes.

        Args:
            name (str): simulation/waveform name

        Returns:
            tuple: spin of the primary object, spin of the secondary object

        """

        simulation = self.df.loc[name]
        return simulation[Parameter.DIMENSIONLESS_SPIN_1.value], simulation[Parameter.DIMENSIONLESS_SPIN_2.value]

    def get_simulations_with_parameters(self, params: list, values: list, tol: list = None) -> list:
        """Takes in a list of parameters, their corresponding values and tolerances (optional, 0.0001
        if not provided) and returns all the waveforms with those values within those tolerances.

        Args:
            params (list): list of parameters
            values (list): list of corresponding values
            tol (:obj:`list`, optional): list of tolerances. If not provided, tolerances of 0.0001 will be used.

        Returns:
            list: a list of simulations/waveforms which satisfy the requirements

        """

        for i in range(len(params)):
            valid = (params[i] in Parameter)
            if not valid:
                warnings.warn(f"{params[i]} is not a valid input.")
                return None
        if tol is not None:
            valid = (len(params) == len(values) == len(tol))
            if not valid:
                warnings.warn("Inconsistent input. Make sure the length of each array is the same.")
                return None
            tol = np.abs(tol)
        else:
            valid = (len(params) == len(values))
            if not valid:
                warnings.warn("Inconsistent input. Make sure the length of each array is the same.")
                return None
            tol = np.ones(len(params)) * 0.0001

        sims = self.simulations
        final_sims = []
        for i in range(len(params)):
            valid = self._check_input(params[i], values[i])
            if not valid:
                return None

            for j in sims:
                if values[i] - tol[i] <= self.df.loc[j][params[i].value] <= values[i] + tol[i]:
                    final_sims.append(j)
            sims = final_sims
            final_sims = []
        return sims

    def get_simulations_with_mass_ratio(self, mass_ratio: float, tol: float = 0.0001) -> list:
        """List of all waveforms with a given mass ratio within specified tolerance.

        Mass ratio is defined as :math:`q=m_1 / m_2 \geq 1`

        Args:
            mass_ratio (float): desired mass ratio value
            tol (:obj:`float`, optional): tolerance

        Returns:
            list:  a list of simulations/waveforms which satisfy the requirements.

        """

        tol = np.abs(tol)
        if mass_ratio < 0:
            print("q should be positive, taking absolute value.")
            mass_ratio = np.abs(mass_ratio)
        if 0 < mass_ratio < 1:
            print(
                f"q should be greater or equal to 1 but the provided value of q is less than 1. Taking inverse : "
                f"q = {1 / mass_ratio:0.4f}. Tolerance will assume q>1.")
            mass_ratio = 1 / mass_ratio
        simulations_with_q = []
        for i in self.simulations:
            if mass_ratio - tol <= self.df.loc[i]["q"] <= mass_ratio + tol:
                simulations_with_q.append(i)
        return simulations_with_q

    def get_simulations_with_symmetric_mass_ratio(self, eta: float, tol: float = 0.0001) -> list:
        """List of all waveforms with a given symmetric mass ratio within specified tolerance.

        Symmetric mass ratio defined as :math:`q/(1+q)^2`

        Args:
            eta (float): desired symmetric mass ratio
            tol (:obj:`float`, optional): tolerance

        Returns:
            list: a list of simulations/waveforms which satisfy the requirements.

        """

        tol = np.abs(tol)
        valid = self._check_input(Parameter.SYMMETRIC_MASS_RATIO, eta)
        if not valid:
            return None
        simulations_with_eta = []
        for i in self.simulations:
            if eta - tol <= self.df.loc[i]["eta"] <= eta + tol:
                simulations_with_eta.append(i)
        return simulations_with_eta

    def get_parameters_for_simulation(self, catalog_id: str) -> dict:
        """Outputs all the metadata for a given waveform.

        Args:
            catalog_id (str): simulation/waveform catalog id

        Returns:
            dict: parameter dictionary 

        """
        row = self.df.loc[catalog_id]
        parameter_dict = {p: row[p.value] for p in Parameter}
        return parameter_dict

    def plot_catalog_parameters(self, p1: Parameter, p2: Parameter, save_path: str = False, dpi: int = 300,
                                color: str = "cornflowerblue"):
        """Plots the two parameter against each other as well as a histogram distribution of each parameter for all waveforms
        in the catalog. If a path is provided, figures will be saved in the provided directory. If a path is not provided,
        the figures will be displayed.

        Args:
            p1 (Parameter): one of the parameters to plot
            p2 (Parameter): the other parameter to plot
            save_path (:obj:`str`, optional): path where you want to save the plots
            dpi (:obj:`int`, optional): dpi of the saved plots
            color (:obj:`str`, optional): color to be used for the plots

        """

        # Plots will be displayed if the path isn't provided or doesn't exist
        if not save_path:
            pass
        elif os.path.exists(save_path) is False:
            print("Path provided doesn't exist. Displaying plots instead.")
            save_path = False

        if p1 in Parameter and p2 in Parameter:
            p1_array = self.df[p1.value]
            p2_array = self.df[p2.value]

            # 2-D plot
            plt.figure(1)
            plt.xlabel(p1.value)
            plt.ylabel(p2.value)
            plt.scatter(p1_array, p2_array, color=color)
            if save_path:
                plt.savefig(f"{save_path}/MAYA_{p1.value}_{p2.value}_plot.png", dpi=dpi, bbox_inches='tight')
            else:
                plt.show()
            plt.clf()

            # Histogram - 1
            plt.figure(2)
            plt.xlabel(p1.value)
            plt.ylabel('count')
            plt.hist(p1_array, color=color)
            if save_path:
                plt.savefig(f"{save_path}/MAYA_{p1.value}_hist.png", dpi=dpi, bbox_inches='tight')
            else:
                plt.show()
            plt.clf()

            # Histogram - 2
            plt.figure(3)
            plt.xlabel(p2.value)
            plt.ylabel('count')
            plt.hist(p2_array, color=color)
            if save_path:
                plt.savefig(f"{save_path}/MAYA_{p2.value}_hist.png", dpi=dpi, bbox_inches='tight')
            else:
                plt.show()
            plt.clf()

            # range of the parameters in the argument
            print(f"Minimum {p1.value}: {min(p1_array)}, maximum {p1.value}: {max(p1_array)}")
            print(f"Minimum {p2.value}: {min(p2_array)}, maximum {p2.value}: {max(p2_array)}")

        else:
            print("Only the following parameters can be plotted")
            print(*Parameter, sep="\n")

    def download_waveforms(self, waveforms: list, save_wf_path, safety: bool = True, lvcnr_format: bool = False):
        """Downloads waveforms from the MAYA Catalog given a list of waveform ids. By default, they are downloaded in the
        MAYA format, but they can also be downloaded in the lvc-nr format (https://arxiv.org/abs/1703.01076).

        Args:
            waveforms (list): list of waveforms to download
            save_wf_path (str): where to save the downloaded waveforms
            safety (:obj:`bool`, optional): whether to request verification if the total size of waveforms exceeds 10 GB.
                Defaults to True.
            lvcnr_format (:obj:`bool`, optional): download the lvc-nr format instead of the MAYA format. Defaults to
                False.

        """

        if not lvcnr_format:
            base_website = "https://cgpstorage.ph.utexas.edu/maya_format"
        else:
            base_website = "https://cgpstorage.ph.utexas.edu/lvcnr_format"

        if not os.path.isdir(save_wf_path):
            warnings.warn('Not a valid place to save the waveforms. Please provide a valid path.')
            return

        # make sure that waveform names in the list actually exist.
        actual_waveforms = np.intersect1d(waveforms, self.simulations)
        not_waveforms = np.setxor1d(actual_waveforms, waveforms)
        if len(actual_waveforms) != len(waveforms):
            print(
                "Some of the waveforms in the list don't exist in the metadata. Removing those names.\nNames removed:")
            print(*not_waveforms, sep="\n")
            waveforms = actual_waveforms

        # check the total size of waveforms and make sure the user wants to download them if the size exceed 10 GB
        memory_tot = 0
        if lvcnr_format:
            for i in range(len(waveforms)):
                memory_tot += self.get_parameters_for_simulation(waveforms[i])[Parameter.LVCNR_SIZE_GB]
            print(f"Total size of requested waveforms is {memory_tot:0.2f} GB.")
        else:
            for i in range(len(waveforms)):
                memory_tot += self.get_parameters_for_simulation(waveforms[i])[Parameter.MAYA_SIZE_GB]
            print(f"Total size of requested waveforms is {memory_tot:0.2f} GB.")

        if memory_tot >= 10 and safety:
            val = input("The total size of the waveforms exceeds 10 GB. Enter 'y' to proceed: ")
            if val != 'y':
                return

        # looping over waveform names to download them.
        print(f"Downloading {len(waveforms)} waveforms and saving in {save_wf_path}.")

        for i in range(len(waveforms)):
            print(f"----------------------------------------------------------")
            print(f"Downloading waveform {waveforms[i]}. This waveform has the following parameters")
            self._print_parameters(waveforms[i])
            url = base_website + f"/{waveforms[i]}.h5"
            try:
                wget.download(url, out=save_wf_path)
            except:
                print(f"Unable to download {waveforms[i]}.h5")
