import numpy as np
import os
import glob
import warnings
import h5py
import re
import pandas as pd
from abc import ABC, abstractmethod
from mayawaves.compactobject import CompactObject
from enum import Enum
from functools import total_ordering
from datetime import date
import romspline
from mayawaves.coalescence import Coalescence
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import shutil

_POSSIBLE_OUTPUT_SUFFIXES = [".out", ".err", "stdout", "stderr"]
_TIMESTEP = None


class _CompactObjectFileHandler(ABC):
    @staticmethod
    @abstractmethod
    def store_compact_object_data_from_filetype(filepaths, compact_object_dict, parameter_file_group):
        metadata_dict_info = {}
        return compact_object_dict, metadata_dict_info

    @staticmethod
    @abstractmethod
    def export_compact_object_data_to_ascii(compact_object_dict, coalescence_output_directory, metadata_dict,
                                            parfile_dict):
        pass

    @staticmethod
    @abstractmethod
    def objects_with_data_source(parfile_dict):
        pass

    @staticmethod
    def get_header(object_number, num_objects):
        pass


class _BH_diagnostics(_CompactObjectFileHandler):
    column_list = [
        CompactObject.Column.ITT,
        CompactObject.Column.TIME,
        CompactObject.Column.X,
        CompactObject.Column.Y,
        CompactObject.Column.Z,
        CompactObject.Column.MIN_RADIUS,
        CompactObject.Column.MAX_RADIUS,
        CompactObject.Column.MEAN_RADIUS,
        CompactObject.Column.QUADRUPOLE_XX,
        CompactObject.Column.QUADRUPOLE_XY,
        CompactObject.Column.QUADRUPOLE_XZ,
        CompactObject.Column.QUADRUPOLE_YY,
        CompactObject.Column.QUADRUPOLE_YZ,
        CompactObject.Column.QUADRUPOLE_ZZ,
        CompactObject.Column.MIN_X,
        CompactObject.Column.MAX_X,
        CompactObject.Column.MIN_Y,
        CompactObject.Column.MAX_Y,
        CompactObject.Column.MIN_Z,
        CompactObject.Column.MAX_Z,
        CompactObject.Column.XY_PLANE_CIRCUMFERENCE,
        CompactObject.Column.XZ_PLANE_CIRCUMFERENCE,
        CompactObject.Column.YZ_PLANE_CIRCUMFERENCE,
        CompactObject.Column.RATIO_OF_XZ_XY_PLANE_CIRCUMFERENCES,
        CompactObject.Column.RATIO_OF_YZ_XY_PLANE_CIRCUMFERENCES,
        CompactObject.Column.AREA,
        CompactObject.Column.M_IRREDUCIBLE,
        CompactObject.Column.AREAL_RADIUS,
        CompactObject.Column.EXPANSION_THETA_L,
        CompactObject.Column.INNER_EXPANSION_THETA_N,
        CompactObject.Column.PRODUCT_OF_THE_EXPANSIONS,
        CompactObject.Column.MEAN_CURVATURE,
        CompactObject.Column.GRADIENT_OF_THE_AREAL_RADIUS,
        CompactObject.Column.GRADIENT_OF_THE_EXPANSION_THETA_L,
        CompactObject.Column.GRADIENT_OF_THE_INNER_EXPANSION_THETA_N,
        CompactObject.Column.GRADIENT_OF_THE_PRODUCT_OF_THE_EXPANSIONS,
        CompactObject.Column.GRADIENT_OF_THE_MEAN_CURVATURE,
        CompactObject.Column.MINIMUM_OF_THE_MEAN_CURVATURE,
        CompactObject.Column.MAXIMUM_OF_THE_MEAN_CURVATURE,
        CompactObject.Column.INTEGRAL_OF_THE_MEAN_CURVATURE
    ]

    @staticmethod
    def store_compact_object_data_from_filetype(filepaths, compact_object_dict, parameter_file_group):
        global _TIMESTEP
        for filename, filepaths in filepaths.items():
            compact_object_num = int(filename[17:-3]) - 1

            if compact_object_num not in compact_object_dict:
                compact_object_dict[compact_object_num] = pd.DataFrame(np.ones((0, 0)), columns=[])
            raw_data = _stitch_timeseries_data(filepaths)

            if raw_data is None:
                continue

            if len(raw_data) == 0:
                continue

            # set the global timeset (how much time changes per iteration)
            _TIMESTEP = raw_data[-1][1] / raw_data[-1][0]

            indices = raw_data[:, 0].astype(int)

            raw_dataframe = pd.DataFrame(data=raw_data, index=indices, columns=list(_BH_diagnostics.column_list))
            raw_dataframe[CompactObject.Column.TIME] = raw_dataframe[CompactObject.Column.ITT] * _TIMESTEP

            compact_object_dict[compact_object_num] = raw_dataframe.combine_first(
                compact_object_dict[compact_object_num])

        return compact_object_dict, {}

    @staticmethod
    def objects_with_data_source(parfile_dict):
        if 'par' in parfile_dict:
            parfile_content = parfile_dict['par']
        elif 'rpar' in parfile_dict:
            parfile_content = parfile_dict['rpar']
        else:
            return []
        for line in parfile_content.splitlines():
            if 'AHFinderDirect::N_horizons' in line:
                compact_object_count = int(line.split('=')[-1].strip())
                return list(range(0, compact_object_count))
        else:
            return []

    @staticmethod
    def export_compact_object_data_to_ascii(compact_object_dict, coalescence_output_directory, metadata_dict,
                                            parfile_dict):
        objects_with_data = _BH_diagnostics.objects_with_data_source(parfile_dict)

        num_objects = len(compact_object_dict)
        for object_number in compact_object_dict:
            compact_object_data = compact_object_dict[object_number]

            if object_number not in objects_with_data:
                continue

            header_list = compact_object_data.attrs["header"].tolist()
            bh_diagnostics_column_idxs = [header_list.index(col.header_text) for col in _BH_diagnostics.column_list if
                                          col.header_text in header_list]
            if len(bh_diagnostics_column_idxs) < len(_BH_diagnostics.column_list):
                continue
            BH_diagnostics_data = compact_object_data[
                                      ~np.isnan(compact_object_data[:, bh_diagnostics_column_idxs[-1]])][:,
                                  bh_diagnostics_column_idxs]
            _export_ascii_file("BH_diagnostics.ah%d.gp" % (object_number + 1), BH_diagnostics_data,
                               coalescence_output_directory,
                               header=_BH_diagnostics.get_header(object_number, num_objects))

    @staticmethod
    def get_header(object_number, num_objects):
        return f"""apparent horizon {object_number + 1}/{num_objects}
column  1 = cctk_iteration
column  2 = cctk_time
column  3 = centroid_x
column  4 = centroid_y
column  5 = centroid_z
column  6 = min radius
column  7 = max radius
column  8 = mean radius
column  9 = quadrupole_xx
column 10 = quadrupole_xy
column 11 = quadrupole_xz
column 12 = quadrupole_yy
column 13 = quadrupole_yz
column 14 = quadrupole_zz
column 15 = min x
column 16 = max x
column 17 = min y
column 18 = max y
column 19 = min z
column 20 = max z
column 21 = xy-plane circumference
column 22 = xz-plane circumference
column 23 = yz-plane circumference
column 24 = ratio of xz/xy-plane circumferences
column 25 = ratio of yz/xy-plane circumferences
column 26 = area
column 27 = m_irreducible
column 28 = areal radius
column 29 = expansion Theta_(l)
column 30 = inner expansion Theta_(n)
column 31 = product of the expansions
column 32 = mean curvature
column 33 = gradient of the areal radius
column 34 = gradient of the expansion Theta_(l)
column 35 = gradient of the inner expansion Theta_(n)
column 36 = gradient of the product of the expansions
column 37 = gradient of the mean curvature
column 38 = minimum  of the mean curvature
column 39 = maximum  of the mean curvature
column 40 = integral of the mean curvature"""


class _Ihspin_hn(_CompactObjectFileHandler):
    column_list = [
        None,
        CompactObject.Column.SX,
        CompactObject.Column.SY,
        CompactObject.Column.SZ,
        CompactObject.Column.PX,
        CompactObject.Column.PY,
        CompactObject.Column.PZ
    ]

    @staticmethod
    def store_compact_object_data_from_filetype(filepaths, compact_object_dict, parameter_file_group):
        global _TIMESTEP
        if _TIMESTEP is None:
            return compact_object_dict, {}

        for filename, filepaths in filepaths.items():
            compact_object_num = int(filename[10:-4])
            if compact_object_num not in compact_object_dict:
                compact_object_dict[compact_object_num] = pd.DataFrame(np.ones((0, 0)), columns=[])
            raw_data = _stitch_timeseries_data(filepaths)

            if raw_data is None:
                continue

            if len(raw_data) == 0:
                continue

            not_none_indices = [i for i, val in enumerate(_Ihspin_hn.column_list) if val is not None]
            not_none_data = raw_data[:, not_none_indices]
            not_none_columns = list(np.array(_Ihspin_hn.column_list)[not_none_indices])

            indices = np.round(raw_data[:, 0] / _TIMESTEP).astype(int)

            raw_dataframe = pd.DataFrame(data=not_none_data, index=indices, columns=list(not_none_columns))
            compact_object_dict[compact_object_num] = raw_dataframe.combine_first(
                compact_object_dict[compact_object_num])

        return compact_object_dict, {}

    @staticmethod
    def objects_with_data_source(parfile_dict):
        if 'par' in parfile_dict:
            parfile_content = parfile_dict['par']
        elif 'rpar' in parfile_dict:
            parfile_content = parfile_dict['rpar']
        else:
            return []
        for line in parfile_content.splitlines():
            if 'IHSpin::num_horizons' in line:
                compact_object_count = int(line.split('=')[-1].strip())
                return list(range(0, compact_object_count))
        else:
            return []

    @staticmethod
    def export_compact_object_data_to_ascii(compact_object_dict, coalescence_output_directory, metadata_dict,
                                            parfile_dict):
        objects_with_data = _Ihspin_hn.objects_with_data_source(parfile_dict)

        num_objects = len(compact_object_dict)
        for object_number in compact_object_dict:
            compact_object_data = compact_object_dict[object_number]

            if object_number not in objects_with_data:
                continue

            header_list = compact_object_data.attrs["header"].tolist()
            ih_spin_column_idxs = [header_list.index(col.header_text) for col in _Ihspin_hn.column_list if
                                   col is not None and col.header_text in header_list]
            ih_spin_column_idxs.insert(0, header_list.index(CompactObject.Column.TIME.header_text))
            if len(ih_spin_column_idxs) < len(_Ihspin_hn.column_list):
                continue
            ihspin_hn_data = compact_object_data[~np.isnan(compact_object_data[:, ih_spin_column_idxs[-1]])][:,
                             ih_spin_column_idxs]
            _export_ascii_file("ihspin_hn_%d.asc" % object_number, ihspin_hn_data,
                               coalescence_output_directory, header=_Ihspin_hn.get_header(object_number, num_objects))

    @staticmethod
    def get_header(object_number, num_objects):
        return f"""IHSpin
horizon no.={object_number}
gnuplot column index:
1:t 2:Sx 3:Sy 4:Sz 5:Px 6:Py 7:Pz"""


class _Shifttracker(_CompactObjectFileHandler):
    column_list = [
        CompactObject.Column.ITT,
        CompactObject.Column.TIME,
        CompactObject.Column.X,
        CompactObject.Column.Y,
        CompactObject.Column.Z,
        CompactObject.Column.VX,
        CompactObject.Column.VY,
        CompactObject.Column.VZ,
        CompactObject.Column.AX,
        CompactObject.Column.AY,
        CompactObject.Column.AZ
    ]

    @staticmethod
    def store_compact_object_data_from_filetype(filepaths, compact_object_dict, parameter_file_group):
        global _TIMESTEP

        for filename, filepaths in filepaths.items():
            compact_object_num = int(filename[12:-4])
            if compact_object_num not in compact_object_dict:
                compact_object_dict[compact_object_num] = pd.DataFrame(np.ones((0, 0)), columns=[])
            raw_data = _stitch_timeseries_data(filepaths)

            if raw_data is None:
                continue

            if len(raw_data) == 0:
                continue

            _TIMESTEP = raw_data[-1][1] / raw_data[-1][0]

            indices = raw_data[:, 0].astype(int)

            raw_dataframe = pd.DataFrame(data=raw_data, index=indices, columns=list(_Shifttracker.column_list))

            combined = raw_dataframe.combine_first(compact_object_dict[compact_object_num].copy())

            compact_object_dict[compact_object_num] = combined

        return compact_object_dict, {}

    @staticmethod
    def objects_with_data_source(parfile_dict):
        if 'par' in parfile_dict:
            parfile_content = parfile_dict['par']
        elif 'rpar' in parfile_dict:
            parfile_content = parfile_dict['rpar']
        else:
            return []
        for line in parfile_content.splitlines():
            if 'ShiftTracker::num_trackers' in line:
                compact_object_count = int(line.split('=')[-1].strip())
                return list(range(0, compact_object_count))
        else:
            return []

    @staticmethod
    def export_compact_object_data_to_ascii(compact_object_dict, coalescence_output_directory, metadata_dict,
                                            parfile_dict):
        objects_with_data = _Shifttracker.objects_with_data_source(parfile_dict)
        num_objects = len(compact_object_dict)
        for object_number in compact_object_dict:
            compact_object_data = compact_object_dict[object_number]

            if object_number not in objects_with_data:
                continue

            header_list = compact_object_data.attrs["header"].tolist()
            shifttracker_column_idxs = [header_list.index(col.header_text) for col in _Shifttracker.column_list if
                                        col.header_text in header_list]
            if len(shifttracker_column_idxs) < len(_Shifttracker.column_list):
                continue
            shifttracker_data = compact_object_data[~np.isnan(compact_object_data[:, shifttracker_column_idxs[-1]])][:,
                                shifttracker_column_idxs]
            _export_ascii_file("ShiftTracker%d.asc" % object_number, shifttracker_data, coalescence_output_directory,
                               header=_Shifttracker.get_header(object_number, num_objects))

    @staticmethod
    def get_header(object_number, num_objects):
        return f"""ShiftTracker{object_number}.asc:
itt   time    x       y       z       vx      vy      vz      ax      ay      az
======================================================================="""


class _Puncturetracker(_CompactObjectFileHandler):
    column_list_primary_object = [
        CompactObject.Column.ITT,
        None, None, None, None, None, None, None, None, None, None, None,
        CompactObject.Column.TIME,
        None, None, None, None, None, None, None, None, None,
        CompactObject.Column.X,
        None, None, None, None, None, None, None, None, None,
        CompactObject.Column.Y,
        None, None, None, None, None, None, None, None, None,
        CompactObject.Column.Z,
        None, None, None, None, None, None, None, None, None
    ]

    @staticmethod
    def store_compact_object_data_from_filetype(filepaths, compact_object_dict, parameter_file_group):
        global _TIMESTEP
        header_info = ""
        with open(list(filepaths.values())[0][0]) as f:
            for line in f:
                if line.strip() == '#':
                    break
                header_info = header_info + line

        for filename, filepaths in filepaths.items():
            for compact_object_num in range(0, 10):
                raw_data = _stitch_timeseries_data(filepaths)

                if raw_data is None:
                    continue

                if len(raw_data) == 0:
                    continue

                # this is different for each compact object. Time, X, Y, Z are shifted by compact_object_num.
                # Only the first column stays the same.
                not_none_indices = [i for i, val in enumerate(_Puncturetracker.column_list_primary_object)
                                    if val is not None]
                column_indices_for_compact_object_num = [0] + [i + compact_object_num for i, val in
                                                               enumerate(_Puncturetracker.column_list_primary_object)
                                                               if val is not None and not i == 0]
                not_none_data = raw_data[:, column_indices_for_compact_object_num]
                not_none_columns = list(np.array(_Puncturetracker.column_list_primary_object)[not_none_indices])

                if np.all(not_none_data[:, 1:] == 0):
                    continue

                first_non_zero_index = np.argmax(not_none_data[:, 0] > 0)
                _TIMESTEP = not_none_data[first_non_zero_index][1] / not_none_data[first_non_zero_index][0]
                indices = not_none_data[:, 0].astype(int)

                raw_dataframe = pd.DataFrame(data=not_none_data, index=indices, columns=not_none_columns)

                if compact_object_num not in compact_object_dict:
                    compact_object_dict[compact_object_num] = pd.DataFrame(np.ones((0, 0)), columns=[])

                compact_object_dict[compact_object_num] = raw_dataframe.combine_first(
                    compact_object_dict[compact_object_num])

        return compact_object_dict, {'Puncturetracker_header_info': header_info, 'Puncturetracker_surface_count': 10}

    @staticmethod
    def objects_with_data_source(parfile_dict):
        if 'par' in parfile_dict:
            parfile_content = parfile_dict['par']
        elif 'rpar' in parfile_dict:
            parfile_content = parfile_dict['rpar']
        else:
            return []

        compact_object_nums = []
        for line in parfile_content.splitlines():
            if 'PunctureTracker::which_surface_to_store_info' in line:
                compact_object_num = int(line.split('=')[-1].strip())
                compact_object_nums.append(compact_object_num)

        return compact_object_nums

    @staticmethod
    def export_compact_object_data_to_ascii(compact_object_dict, coalescence_output_directory, metadata_dict,
                                            parfile_dict):
        if 'Puncturetracker_header_info' not in metadata_dict:
            return

        objects_with_data = _Puncturetracker.objects_with_data_source(parfile_dict)

        puncturetracker_dataframe = pd.DataFrame(np.zeros((0, 0)), columns=[])

        co_objects_with_data = []

        for object_number in compact_object_dict:
            compact_object_data = compact_object_dict[object_number]

            if object_number not in objects_with_data:
                continue

            header_list = compact_object_data.attrs["header"].tolist()
            puncturetracker_column_idxs = [header_list.index(col.header_text) for col in
                                           _Puncturetracker.column_list_primary_object if
                                           col is not None and col.header_text in header_list]

            puncturetracker_co_data = compact_object_data[
                                          ~np.isnan(compact_object_data[:, puncturetracker_column_idxs[-1]])][:,
                                      puncturetracker_column_idxs]

            indices = puncturetracker_co_data[:, 0]
            columns = [col.header_text + f'_{object_number}' if col is not CompactObject.Column.ITT else col.header_text
                       for col in _Puncturetracker.column_list_primary_object if col is not None]

            raw_dataframe = pd.DataFrame(data=puncturetracker_co_data, index=indices, columns=columns)

            puncturetracker_dataframe = raw_dataframe.combine_first(puncturetracker_dataframe)
            co_objects_with_data.append(object_number)

        time_columns = [f'{CompactObject.Column.TIME.header_text}_{object_number}' for object_number in
                        co_objects_with_data]
        puncturetracker_dataframe[CompactObject.Column.TIME.header_text] = \
        puncturetracker_dataframe[time_columns].fillna(method='ffill', axis=1)[time_columns[0]]
        puncturetracker_array = np.zeros(
            (len(puncturetracker_dataframe), 2 + len(_Puncturetracker.column_list_primary_object)))

        puncturetracker_array[:, 0] = puncturetracker_dataframe[CompactObject.Column.ITT.header_text]
        puncturetracker_array[:, 2] = puncturetracker_dataframe[CompactObject.Column.ITT.header_text]
        puncturetracker_array[:, 1] = puncturetracker_dataframe[CompactObject.Column.TIME.header_text]
        puncturetracker_array[:, 10] = puncturetracker_dataframe[CompactObject.Column.TIME.header_text]

        for co_number in co_objects_with_data:
            columns = [col.header_text + f'_{co_number}' for col in _Puncturetracker.column_list_primary_object if
                       col is not None and col is not CompactObject.Column.ITT]
            column_idxs = [2 + i + co_number for i, val in enumerate(_Puncturetracker.column_list_primary_object) if
                           val is not None and not i == 0]
            puncturetracker_array[:, column_idxs] = puncturetracker_dataframe[columns]

        header = metadata_dict['Puncturetracker_header_info'] + """#
# PUNCTURETRACKER::PT_LOC (puncturetracker-pt_loc)
#"""

        fmt = """# iteration %d   time %f
# time level 0
# refinement level 0   multigrid level 0   map 0   component 0
# column format: 1:it	2:tl	3:rl 4:c 5:ml	6:ix 7:iy 8:iz	9:time	10:x 11:y 12:z	13:data
# data columns: 13:pt_loc_t[0] 14:pt_loc_t[1] 15:pt_loc_t[2] 16:pt_loc_t[3] 17:pt_loc_t[4] 18:pt_loc_t[5] 19:pt_loc_t[6] 20:pt_loc_t[7] 21:pt_loc_t[8] 22:pt_loc_t[9] 23:pt_loc_x[0] 24:pt_loc_x[1] 25:pt_loc_x[2] 26:pt_loc_x[3] 27:pt_loc_x[4] 28:pt_loc_x[5] 29:pt_loc_x[6] 30:pt_loc_x[7] 31:pt_loc_x[8] 32:pt_loc_x[9] 33:pt_loc_y[0] 34:pt_loc_y[1] 35:pt_loc_y[2] 36:pt_loc_y[3] 37:pt_loc_y[4] 38:pt_loc_y[5] 39:pt_loc_y[6] 40:pt_loc_y[7] 41:pt_loc_y[8] 42:pt_loc_y[9] 43:pt_loc_z[0] 44:pt_loc_z[1] 45:pt_loc_z[2] 46:pt_loc_z[3] 47:pt_loc_z[4] 48:pt_loc_z[5] 49:pt_loc_z[6] 50:pt_loc_z[7] 51:pt_loc_z[8] 52:pt_loc_z[9]
%d\t""" + "\t".join(["%f"] * (len(_Puncturetracker.column_list_primary_object) - 1)) + "\n"

        filename = os.path.join(coalescence_output_directory, "puncturetracker-pt_loc..asc")
        np.savetxt(filename, puncturetracker_array, fmt=fmt, comments='', header=header)


class _QuasiLocalMeasures(_CompactObjectFileHandler):
    column_list = [CompactObject.Column.ITT,
                   CompactObject.Column.TIME,
                   CompactObject.Column.EQUATORIAL_CIRCUMFERENCE,
                   CompactObject.Column.POLAR_CIRCUMFERENCE_0,
                   CompactObject.Column.POLAR_CIRCUMFERENCE_PI_2,
                   CompactObject.Column.AREA,
                   CompactObject.Column.M_IRREDUCIBLE,
                   CompactObject.Column.AREAL_RADIUS,
                   CompactObject.Column.SPIN_GUESS,
                   CompactObject.Column.MASS_GUESS,
                   CompactObject.Column.KILLING_EIGENVALUE_REAL,
                   CompactObject.Column.KILLING_EIGENVALUE_IMAG,
                   CompactObject.Column.SPIN_MAGNITUDE,
                   CompactObject.Column.NPSPIN,
                   CompactObject.Column.WSSPIN,
                   CompactObject.Column.SPIN_FROM_PHI_COORDINATE_VECTOR,
                   CompactObject.Column.SX,
                   CompactObject.Column.SY,
                   CompactObject.Column.SZ,
                   CompactObject.Column.HORIZON_MASS,
                   CompactObject.Column.ADM_ENERGY,
                   CompactObject.Column.ADM_MOMENTUM_X,
                   CompactObject.Column.ADM_MOMENTUM_Y,
                   CompactObject.Column.ADM_MOMENTUM_Z,
                   CompactObject.Column.ADM_ANGULAR_MOMENTUM_X,
                   CompactObject.Column.ADM_ANGULAR_MOMENTUM_Y,
                   CompactObject.Column.ADM_ANGULAR_MOMENTUM_Z,
                   CompactObject.Column.WEINBERG_ENERGY,
                   CompactObject.Column.WEINBERG_MOMENTUM_X,
                   CompactObject.Column.WEINBERG_MOMENTUM_Y,
                   CompactObject.Column.WEINBERG_MOMENTUM_Z,
                   CompactObject.Column.WEINBERG_ANGULAR_MOMENTUM_X,
                   CompactObject.Column.WEINBERG_ANGULAR_MOMENTUM_Y,
                   CompactObject.Column.WEINBERG_ANGULAR_MOMENTUM_Z]

    @staticmethod
    def store_compact_object_data_from_filetype(filepaths, compact_object_dict, parameter_file_group):
        global _TIMESTEP
        header_info = ""
        with open(list(filepaths.values())[0][0]) as f:
            for line in f:
                if line.strip() == '#':
                    break
                header_info = header_info + line

        compact_object_count = None
        if 'par_content' in parameter_file_group.attrs:
            parfile_content = parameter_file_group.attrs['par_content']
        elif 'rpar_content' in parameter_file_group.attrs:
            parfile_content = parameter_file_group.attrs['rpar_content']
        else:
            warnings.warn('No parameter file information. Unable to read QuasiLocalMeasures data.')
            return
        for line in parfile_content.splitlines():
            if 'QuasiLocalMeasures::num_surfaces' in line:
                compact_object_count = int(line.split('=')[-1].strip())

        if compact_object_count == None:
            return compact_object_dict

        for filename, filepaths in filepaths.items():
            for compact_object_num in range(0, compact_object_count):
                raw_data = _stitch_timeseries_data(filepaths)

                if raw_data is None:
                    continue

                if len(raw_data) == 0:
                    continue

                data_columns = 12 + np.arange(0, 33) * compact_object_count + compact_object_num
                data_columns = np.insert(data_columns, 0, 0)

                qlm_data = raw_data[:, data_columns]

                if np.all(qlm_data[:, 1:] == 0):
                    continue

                indices = qlm_data[:, 0].astype(int)

                if _TIMESTEP is None:
                    _TIMESTEP = qlm_data[1, 1] / qlm_data[1, 0]

                raw_dataframe = pd.DataFrame(data=qlm_data, index=indices, columns=_QuasiLocalMeasures.column_list)
                raw_dataframe = raw_dataframe[(raw_dataframe[CompactObject.Column.M_IRREDUCIBLE] != 0)]

                raw_dataframe.drop_duplicates(subset=CompactObject.Column.TIME, inplace=True)

                if compact_object_num not in compact_object_dict:
                    compact_object_dict[compact_object_num] = pd.DataFrame(np.ones((0, 0)), columns=[])

                compact_object_dict[compact_object_num] = raw_dataframe.combine_first(
                    compact_object_dict[compact_object_num])

        return compact_object_dict, {'QuasiLocalMeasures_header_info': header_info,
                                     'QuasiLocalMeasures_surface_count': compact_object_count}

    @staticmethod
    def objects_with_data_source(parfile_dict):
        if 'par' in parfile_dict:
            parfile_content = parfile_dict['par']
        elif 'rpar' in parfile_dict:
            parfile_content = parfile_dict['rpar']
        else:
            return []
        for line in parfile_content.splitlines():
            if 'QuasiLocalMeasures::num_surfaces ' in line:
                compact_object_count = int(line.split('=')[-1].strip())
                return list(range(0, compact_object_count))
        else:
            return []

    @staticmethod
    def export_compact_object_data_to_ascii(compact_object_dict, coalescence_output_directory, metadata_dict,
                                            parfile_dict):
        if 'QuasiLocalMeasures_header_info' not in metadata_dict:
            return

        objects_with_data = _QuasiLocalMeasures.objects_with_data_source(parfile_dict)

        num_surfaces = metadata_dict['QuasiLocalMeasures_surface_count']

        quasilocalmeasures_dataframe = pd.DataFrame(np.zeros((0, 0)), columns=[])

        co_objects_with_data = []

        for object_number in compact_object_dict:
            compact_object_data = compact_object_dict[object_number]

            if object_number not in objects_with_data:
                continue

            header_list = compact_object_data.attrs["header"].tolist()
            qlm_column_idxs = [header_list.index(col.header_text) for col in
                               _QuasiLocalMeasures.column_list if
                               col is not None and col.header_text in header_list]

            qlm_co_data = compact_object_data[
                              ~np.isnan(compact_object_data[:, qlm_column_idxs[-1]])][:,
                          qlm_column_idxs]

            indices = qlm_co_data[:, 0]
            columns = [col.header_text + f'_{object_number}' if col is not CompactObject.Column.ITT else col.header_text
                       for col in _QuasiLocalMeasures.column_list if col is not None]

            raw_dataframe = pd.DataFrame(data=qlm_co_data, index=indices, columns=columns)

            quasilocalmeasures_dataframe = raw_dataframe.combine_first(quasilocalmeasures_dataframe)
            co_objects_with_data.append(object_number)

        time_columns = [f'{CompactObject.Column.TIME.header_text}_{object_number}' for object_number in
                        co_objects_with_data]
        quasilocalmeasures_dataframe[CompactObject.Column.TIME.header_text] = \
        quasilocalmeasures_dataframe[time_columns].fillna(method='ffill', axis=1)[time_columns[-1]]

        qlm_array = np.zeros(
            (len(quasilocalmeasures_dataframe), 14 + num_surfaces * (len(_QuasiLocalMeasures.column_list) - 1)))

        quasilocalmeasures_dataframe.fillna(method='ffill', inplace=True, axis=0)

        qlm_array[:, 0] = quasilocalmeasures_dataframe[CompactObject.Column.ITT.header_text]
        qlm_array[:, 2] = quasilocalmeasures_dataframe[CompactObject.Column.ITT.header_text]
        qlm_array[:, 1] = quasilocalmeasures_dataframe[CompactObject.Column.TIME.header_text]
        qlm_array[:, 10] = quasilocalmeasures_dataframe[CompactObject.Column.TIME.header_text]

        for co_number in co_objects_with_data:
            columns = [col.header_text + f'_{co_number}' for col in _QuasiLocalMeasures.column_list if
                       col is not None and col is not CompactObject.Column.ITT]
            column_idxs = [14 + (i - 1) * num_surfaces + co_number for i, val in
                           enumerate(_QuasiLocalMeasures.column_list) if
                           val is not None and not i == 0]
            qlm_array[:, column_idxs] = quasilocalmeasures_dataframe[columns]

        header = """# Export from Mayawaves. Some values (time, irreducible mass, area, radius) saved using BHDiagnostics data rather than the raw QLM data.
""" + metadata_dict['QuasiLocalMeasures_header_info'] + """#
# QUASILOCALMEASURES::QLM_SCALARS (quasilocalmeasures-qlm_scalars)
#"""

        data_columns = []
        column_counter = 13
        column_names = ['qlm_time', 'qlm_equatorial_circumference', 'qlm_polar_circumference_0',
                        'qlm_polar_circumference_pi_2', 'qlm_area', 'qlm_irreducible_mass', 'qlm_radius',
                        'qlm_spin_guess', 'qlm_mass_guess', 'qlm_killing_eigenvalue_re', 'qlm_killing_eigenvalue_im',
                        'qlm_spin', 'qlm_npspin', 'qlm_wsspin', 'qlm_cvspin', 'qlm_coordspinx', 'qlm_coordspiny',
                        'qlm_coordspinz', 'qlm_mass', 'qlm_adm_energy', 'qlm_adm_momentum_x', 'qlm_adm_momentum_y',
                        'qlm_adm_momentum_z', 'qlm_adm_angular_momentum_x', 'qlm_adm_angular_momentum_y',
                        'qlm_adm_angular_momentum_z', 'qlm_w_energy', 'qlm_w_momentum_x', 'qlm_w_momentum_y',
                        'qlm_w_momentum_z', 'qlm_w_angular_momentum_x', 'qlm_w_angular_momentum_y',
                        'qlm_w_angular_momentum_z', ]
        for col_name in column_names:
            for i in range(num_surfaces):
                data_columns.append(f'{column_counter}:{col_name}[{i}]')
                column_counter += 1

        fmt = """# iteration %d   time %f
# time level 0
# refinement level 0   multigrid level 0   map 0   component 0
# column format: 1:it	2:tl	3:rl 4:c 5:ml	6:ix 7:iy 8:iz	9:time	10:x 11:y 12:z	13:data
# data columns: """ + ' '.join(data_columns) + """
%d\t""" + "\t".join(["%f"] * (11 + num_surfaces * (len(_QuasiLocalMeasures.column_list) - 1))) + "\n"

        filename = os.path.join(coalescence_output_directory, "quasilocalmeasures-qlm_scalars..asc")
        np.savetxt(filename, qlm_array, fmt=fmt, comments='', header=header)


class _RadiativeFileHandler(ABC):
    @staticmethod
    @abstractmethod
    def store_radiative_data_from_filetype(filepaths, psi4_group):
        return psi4_group

    @staticmethod
    def get_header(radius, l_value=0, m_value=0, radius_index=0, com_correction: bool = False):
        pass

    @staticmethod
    @abstractmethod
    def export_radiative_data_to_ascii(psi4_data_dict, coalescence_output_directory, com_correction: bool = False):
        pass

    @staticmethod
    def store_stitched_psi4_data_from_dict(psi4_data, psi4_group):
        for r, psi4_data_at_radius in psi4_data.items():
            # find common times per radius
            common_times = sorted(list(set.intersection(*[set(data[:, 0]) for data in psi4_data_at_radius.values()])))

            extraction_radius_group_name = "radius=%s" % r

            if extraction_radius_group_name not in psi4_group.keys():
                extraction_radius_group = psi4_group.create_group(extraction_radius_group_name)
                extraction_radius_group.create_dataset("time", data=common_times)
            else:
                extraction_radius_group = psi4_group[extraction_radius_group_name]
                old_time = extraction_radius_group['time'][()]
                updated_common_times = sorted(list(set.intersection(*[set(common_times), set(old_time)])))
                if 'analysis' in extraction_radius_group:
                    analysis_data = extraction_radius_group['analysis']
                    updated_analysis_data = analysis_data[old_time.searchsorted(updated_common_times)]
                    del extraction_radius_group['analysis']
                    extraction_radius_group.create_dataset("analysis", data=updated_analysis_data)

                del extraction_radius_group['time']
                extraction_radius_group.create_dataset("time", data=common_times)

            if "modes" not in extraction_radius_group.keys():
                modes_group = extraction_radius_group.create_group("modes")
            else:
                modes_group = extraction_radius_group["modes"]

            for mode, data in psi4_data_at_radius.items():
                l, m = mode
                data = data[data[:, 0].searchsorted(common_times)]

                l_group_name = "l=%d" % l
                m_group_name = "m=%d" % m

                if l_group_name not in modes_group.keys():
                    l_group = modes_group.create_group(l_group_name)
                else:
                    l_group = modes_group[l_group_name]

                m_group = l_group.create_group(m_group_name)
                m_group.create_dataset("real", data=data[:, 1])
                m_group.create_dataset("imaginary", data=data[:, 2])


class _Ylm_WEYLSCAL4_ASC(_RadiativeFileHandler):
    @staticmethod
    def store_radiative_data_from_filetype(filepaths, psi4_group):
        psi4_group.attrs["source"] = "YLM_WEYLSCAL4_ASC"
        psi4_data = {}

        # stitch the data together first
        for filename, filepaths in filepaths["radiative"][_RadiativeFilenames.YLM_WEYLSCAL4_ASC].items():
            data = _stitch_timeseries_data(filepaths)
            if data is None:
                continue
            l, m, r = _l_m_radius_from_psi4_filename(filename)
            if r not in psi4_data:
                psi4_data[r] = {}
            psi4_data[r][(l, m)] = data

        _RadiativeFileHandler.store_stitched_psi4_data_from_dict(psi4_data, psi4_group)

    @staticmethod
    def get_header(radius, l_value=0, m_value=0, radius_index=0, com_correction: bool = False):
        com_drift_str = "\nCorrected for center of mass drift"
        return f"""Ylm_Decomp
WEYLSCAL4::Psi4r l={l_value} m={m_value} r={radius:.2f}
1:t 2:re(y) 3:im(y){com_drift_str if com_correction else ''}"""

    @staticmethod
    def export_radiative_data_to_ascii(psi4_data_dict, coalescence_output_directory, com_correction: bool = False):
        radii = list(psi4_data_dict.keys())
        for radius in radii:
            modes = list(psi4_data_dict[radius].keys())
            for mode in modes:
                l_val = mode[0]
                m_val = mode[1]
                _export_ascii_file("Ylm_WEYLSCAL4::Psi4r_l%d_m%d_r%.2f.asc" % (l_val, m_val, radius),
                                   psi4_data_dict[radius][mode], coalescence_output_directory,
                                   header=_Ylm_WEYLSCAL4_ASC.get_header(radius, l_value=l_val, m_value=m_val,
                                                                        com_correction=com_correction))


class _MP_Psi4_ASC(_RadiativeFileHandler):
    @staticmethod
    def store_radiative_data_from_filetype(filepaths, psi4_group):
        psi4_group.attrs["source"] = "MP_PSI4_ASC"
        psi4_data = {}

        # stitch the data together first
        for filename, filepaths in filepaths["radiative"][_RadiativeFilenames.MP_PSI4_ASC].items():
            data = _stitch_timeseries_data(filepaths)
            if data is None:
                continue
            l, m, r = _l_m_radius_from_psi4_filename(filename)
            if r not in psi4_data:
                psi4_data[r] = {}
            psi4_data[r][(l, m)] = data

        _RadiativeFileHandler.store_stitched_psi4_data_from_dict(psi4_data, psi4_group)

    @staticmethod
    def get_header(radius, l_value=0, m_value=0, radius_index=0, com_correction: bool = False):
        if com_correction:
            return "Corrected for center of mass drift"
        else:
            return None

    @staticmethod
    def export_radiative_data_to_ascii(psi4_data_dict, coalescence_output_directory, com_correction: bool = False):
        radii = list(psi4_data_dict.keys())
        for radius in radii:
            modes = list(psi4_data_dict[radius].keys())
            for mode in modes:
                l_val = mode[0]
                m_val = mode[1]
                _export_ascii_file("mp_Psi4_l%d_m%d_r%.2f.asc" % (l_val, m_val, radius),
                                   psi4_data_dict[radius][mode], coalescence_output_directory,
                                   header=_MP_Psi4_ASC.get_header(radius, l_value=l_val, m_value=m_val,
                                                                  com_correction=com_correction))


class _MP_Psi4_H5(_RadiativeFileHandler):
    @staticmethod
    def store_radiative_data_from_filetype(filepaths, psi4_group):
        psi4_group.attrs["source"] = _RadiativeFilenames.MP_PSI4_H5.string_name
        psi4_data = {}

        for filename, filepaths in filepaths["radiative"][_RadiativeFilenames.MP_PSI4_H5].items():
            # read the data from the first output
            if len(filepaths) == 0:
                continue

            psi4_h5_file = h5py.File(filepaths[0], 'r')
            for dataset_name in psi4_h5_file.keys():
                split_name = dataset_name.split('_')
                l = int(split_name[0][1:])
                m = int(split_name[1][1:])
                r = split_name[2][1:]

                data = psi4_h5_file[dataset_name][()]

                if r not in psi4_data:
                    psi4_data[r] = {}
                psi4_data[r][(l, m)] = data

            if len(filepaths) == 1:
                continue

            for filepath in filepaths[1:]:
                psi4_h5_file = h5py.File(filepath, 'r')
                for dataset_name in psi4_h5_file.keys():
                    split_name = dataset_name.split('_')
                    l = int(split_name[0][1:])
                    m = int(split_name[1][1:])
                    r = split_name[2][1:]

                    temp_data = psi4_h5_file[dataset_name][()]

                    # crop out duplicated times/iterations
                    prev_last_time = psi4_data[r][(l, m)][-1, 0]
                    index_to_start = np.argmax(temp_data[:, 0] > prev_last_time)
                    cropped_data = temp_data[index_to_start:]
                    stitched_data = np.append(psi4_data[r][(l, m)], cropped_data, axis=0)
                    psi4_data[r][(l, m)] = stitched_data

        _RadiativeFileHandler.store_stitched_psi4_data_from_dict(psi4_data, psi4_group)

    @staticmethod
    def export_radiative_data_to_ascii(psi4_data_dict, coalescence_output_directory, com_correction: bool = False):
        h5_filename = os.path.join(coalescence_output_directory, 'mp_psi4.h5')
        h5_file = h5py.File(h5_filename, 'w')
        radii = list(psi4_data_dict.keys())
        for radius in radii:
            modes = list(psi4_data_dict[radius].keys())
            for mode in modes:
                l_val = mode[0]
                m_val = mode[1]
                psi4_data = psi4_data_dict[radius][mode]
                dataset_name = f'l{l_val:d}_m{m_val:d}_r{radius:.2f}'
                h5_file.create_dataset(dataset_name, data=psi4_data)
        if com_correction:
            h5_file.attrs['frame'] = 'Corrected for center of mass drift'
        h5_file.close()


class _MP_WeylScal4_ASC(_RadiativeFileHandler):
    @staticmethod
    def store_radiative_data_from_filetype(filepaths, psi4_group):
        psi4_group.attrs["source"] = "MP_WEYLSCAL4_ASC"
        psi4_data = {}

        # stitch the data together first
        for filename, filepaths in filepaths["radiative"][_RadiativeFilenames.MP_WEYLSCAL4_ASC].items():
            data = _stitch_timeseries_data(filepaths)
            if data is None:
                continue
            l, m, r = _l_m_radius_from_psi4_filename(filename)
            if r not in psi4_data:
                psi4_data[r] = {}
            psi4_data[r][(l, m)] = data

        _RadiativeFileHandler.store_stitched_psi4_data_from_dict(psi4_data, psi4_group)

    @staticmethod
    def get_header(radius, l_value=0, m_value=0, radius_index=0, com_correction: bool = False):
        if com_correction:
            return "Corrected for center of mass drift"
        else:
            return None

    @staticmethod
    def export_radiative_data_to_ascii(psi4_data_dict, coalescence_output_directory, com_correction: bool = False):
        radii = list(psi4_data_dict.keys())
        for radius in radii:
            modes = list(psi4_data_dict[radius].keys())
            for mode in modes:
                l_val = mode[0]
                m_val = mode[1]
                _export_ascii_file("mp_WeylScal4::Psi4i_l%d_m%d_r%.2f.asc" % (l_val, m_val, radius),
                                   psi4_data_dict[radius][mode], coalescence_output_directory,
                                   header=_MP_WeylScal4_ASC.get_header(radius, l_value=l_val, m_value=m_val,
                                                                       com_correction=com_correction))


@total_ordering
class _CompactObjectFilenames(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, processing_order, regex, class_reference, string_name):
        self.processing_order = processing_order
        self.regex = regex
        self.class_reference = class_reference
        self.string_name = string_name

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.processing_order < other.processing_order
        return NotImplemented

    BH_DIAGNOSTICS = 1, r'BH_diagnostics.ah\d+.gp', _BH_diagnostics, 'BH_DIAGNOSTICS'
    IHSPIN_HN = 4, r'ihspin_hn_\d+.asc', _Ihspin_hn, 'IHSPIN_HN'
    SHIFTTRACKER = 2, r'ShiftTracker\d+\.asc', _Shifttracker, 'SHIFTTRACKER'
    PUNCTURETRACKER = 3, r'puncturetracker-pt_loc\.\.asc', _Puncturetracker, 'PUNCTURETRACKER'
    QUASILOCALMEASURES = 0, r'quasilocalmeasures-qlm_scalars\.\.asc', _QuasiLocalMeasures, 'QUASILOCALMEASURES'


class _RadiativeFilenames(Enum):
    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, processing_order, regex, class_reference, string_name):
        self.processing_order = processing_order
        self.regex = regex
        self.class_reference = class_reference
        self.string_name = string_name

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.processing_order < other.processing_order
        return NotImplemented

    YLM_WEYLSCAL4_ASC = 1, r'Ylm_WEYLSCAL4::Psi4[i,r]_l-?\d_m-?\d_r\d+(\.\d+)?.asc', _Ylm_WEYLSCAL4_ASC, 'YLM_WEYLSCAL4_ASC'
    MP_PSI4_ASC = 2, r'mp_Psi4[i,r]?_l-?\d_m-?\d_r\d+(\.\d+)?.asc', _MP_Psi4_ASC, 'MP_PSI4_ASC'
    MP_PSI4_H5 = 3, r'mp_psi4.h5', _MP_Psi4_H5, 'MP_PSI4_H5'
    MP_WEYLSCAL4_ASC = 4, r'mp_WeylScal4::Psi4[i,r]?_l-?\d_m-?\d_r\d+(\.\d+)?.asc', _MP_WeylScal4_ASC, 'MP_WEYLSCAL4_ASC'


class _MiscDataFilenames(Enum):
    RUNSTATS = r'runstats\.asc'


class _OutputFilenames(Enum):
    OUT = r".*\.out"
    ERR = r".*\.err"
    STDOUT = r"stdout"
    STDERR = r"stderr"


def _simulation_name(raw_directory: str) -> str:
    """Name of the simulation.

    The name of the simulation from the name of the directory containing the simulation data.

    Args:
        raw_directory (str): the directory that contains all simulation data

    Returns:
        str: simulation name constructed from the directory name

    """
    simulation_name = raw_directory.split('/')[-1]
    return simulation_name

def _get_parameter_file_name_and_content(raw_directory: str) -> tuple:
    """Store the parameter file in the h5 file

    Search for .rpar and .par files and store in a dictionary

    Args:
        raw_directory (str): the directory that contains all simulation data

    Returns:
        tuple: base name of the parameter file and a dictionary containing the rpar and par content

    """
    parfile_name = None
    parfile_dict = {}

    # check if SIMFACTORY directory exists
    rpar_file = None
    par_file = None
    if os.path.isdir(os.path.join(raw_directory, "SIMFACTORY/par")):
        files_with_rpar = glob.glob(os.path.join(raw_directory, "SIMFACTORY/par/*.rpar"))
        if len(files_with_rpar) > 0:  # if there is an rpar file
            rpar_file = files_with_rpar[0]
        files_with_par = glob.glob(os.path.join(raw_directory, "SIMFACTORY/par/*.par"))
        if len(files_with_par) > 0:  # if there is a par file
            par_file = files_with_par[0]

    # if we haven't found a par file or a rpar file
    if rpar_file is None or par_file is None:
        output_directories, _ = _ordered_output_directories(raw_directory)
        output_directories.reverse()  # most recent output directory is first now

        for output_dir in output_directories:
            if rpar_file is None:
                files_with_rpar = glob.glob(os.path.join(output_dir, "*.rpar"))
                if len(files_with_rpar) > 0:  # if there is a rpar file
                    rpar_file = files_with_rpar[0]
            if par_file is None:
                files_with_par = glob.glob(os.path.join(output_dir, "*.par"))
                if len(files_with_par) > 0:  # if there is a par file
                    par_file = files_with_par[0]

            if rpar_file is not None and par_file is not None:
                break

    # if we never found a parameter file
    if rpar_file is None and par_file is None:
        warnings.warn("Unable to locate a .rpar or .par file in this simulation directory.")
        return None, None

    created_par = False

    # store the rpar file if it exists and create parfile
    if rpar_file is not None:
        rpar_name = rpar_file.split('/')[-1]
        rpar_name_base = rpar_name[:rpar_name.rfind('.')]
        parfile_name = rpar_name_base
        with open(rpar_file, 'r') as f:
            content = f.read()
        parfile_dict['rpar_content'] = content
        if par_file is None:
            temporary_rpar = rpar_file[:rpar_file.rfind('/') + 1] + "temp.rpar"
            par_file = rpar_file[:rpar_file.rfind('/') + 1] + "temp.par"
            with open(temporary_rpar, 'w') as f:
                f.write(parfile_dict['rpar_content'])
            os.system(f'chmod +x {temporary_rpar}')
            os.system('%s' % temporary_rpar)
            created_par = True
            os.remove(temporary_rpar)

    # store the par file
    if parfile_name is None:
        par_name = par_file.split('/')[-1]
        parfile_name = par_name[:par_name.rfind('.')]
    if os.path.exists(par_file):
        with open(par_file, 'r') as f:
            content = f.read()
        parfile_dict['par_content'] = content

    if created_par and os.path.exists(par_file):
        os.remove(par_file)

    return parfile_name, parfile_dict


def _ordered_output_directories(raw_directory: str) -> tuple:
    """The output directories within the raw directory, ordered.

    Find all the output directories within the raw directory and sort in ascending order. If the data has been
    prestitched, there aren't individual output directories.

    Args:
        raw_directory (str): the directory that contains all simulation data

    Returns:
        tuple: the output directories sorted in ascending order, boolean specifying whether the data has been
            prestitched

    """
    prestitched = False
    output_directories = glob.glob(
        os.path.join(raw_directory, "output-[0-9][0-9][0-9][0-9]"))
    output_directories.sort()
    if len(output_directories) == 0:
        # pre-stitched data
        output_directories = [raw_directory]
        prestitched = True
    return output_directories, prestitched


def _ordered_data_directories(raw_directory: str, parameter_file: str, parameter_file_name_base: str) -> list:
    """The directories containing the simulation data files, ordered.

    Within each output directory is a data directory. Returns an ordered list of all the data directories. If the data
    is prestitched, this is the same as the output directory and raw directory.

    Args:
        raw_directory (str): the directory that contains all simulation data
        parameter_file_name_base (str): the base name of the parameter file

    Returns:
        list: ordered list containing the ordered data directories

    """
    output_directories, prestitched = _ordered_output_directories(raw_directory)

    simulation_name = raw_directory.split('/')[-1]

    if prestitched:
        # pre-stitched data
        data_directories = output_directories
    else:
        if parameter_file is None:
            return None
        data_dir_name = ""

        result = re.search('IO::out_dir\s*=\s*(\S*)\s*\n', parameter_file)
        try:
            data_dir_name = result.group(1)
            data_dir_name = data_dir_name.strip('"\'')
        except:
            warnings.warn("Can't find name of the data directory, assuming it is an empty string")
        if data_dir_name == '\$parfile' or data_dir_name == '$parfile':
            data_dir_name = parameter_file_name_base
        if data_dir_name == '@SIMULATION_NAME@':
            data_dir_name = simulation_name

        data_directories = [os.path.join(output_directory, data_dir_name) for output_directory in
                            output_directories]
    return data_directories

def _store_parameter_file(parfile_dict: dict, h5_file: h5py.File):
    """Store the parameter file in the h5 file

    Store .rpar and .par file information

    Args:
        parfile_dict (dict): dictionary containing the rpar and par content
        h5_file (h5py.file): the h5 file to store the parameter file in

    """
    if parfile_dict is None or ("par_content" not in parfile_dict and "rpar_content" not in parfile_dict):
        return

    parfile_group = h5_file.create_group('parfile')

    if "par_content" in parfile_dict:
        parfile_group.attrs['par_content'] = parfile_dict['par_content']

    if "rpar_content" in parfile_dict:
        parfile_group.attrs['rpar_content'] = parfile_dict['rpar_content']

def _all_relevant_data_filepaths(raw_directory: str, parameter_file: str, parameter_file_name_base: str) -> dict:
    """Dictionary of all relevant data files.

    The dictionary points from a data type to a filename which in turn points to a list of filepaths with that filename.

    Args:
        raw_directory (str): the directory that contains all simulation data
        parameter_file_name_base (str): the base name of the parameter file

    Returns:
        dict: dictionary containing prefix -> filename -> list of filepaths to relevant files

    """
    data_directories = _ordered_data_directories(raw_directory, parameter_file=parameter_file, parameter_file_name_base=parameter_file_name_base)
    relevant_data_filepaths = {"compact_object": {}, "radiative": {}, "misc": {}}

    # go through all output directories
    for output_directory in data_directories:
        data_path = os.path.join(output_directory, "*")

        # go through all datafiles in the output directory
        for filepath in glob.glob(data_path):
            # grab only the filename, not the full path
            filename = filepath.split('/')[-1]
            # for all data prefixes we may want

            for filetype in _RadiativeFilenames:
                re_match = re.fullmatch(filetype.regex, filename)
                if re_match:
                    if filetype not in relevant_data_filepaths["radiative"]:
                        relevant_data_filepaths["radiative"][filetype] = {}
                    if filename not in relevant_data_filepaths["radiative"][filetype]:
                        relevant_data_filepaths["radiative"][filetype][filename] = []
                    relevant_data_filepaths["radiative"][filetype][filename].append(filepath)

            for filetype in _CompactObjectFilenames:
                re_match = re.fullmatch(filetype.regex, filename)
                if re_match:
                    if filetype not in relevant_data_filepaths["compact_object"]:
                        relevant_data_filepaths["compact_object"][filetype] = {}
                    if filename not in relevant_data_filepaths["compact_object"][filetype]:
                        relevant_data_filepaths["compact_object"][filetype][filename] = []
                    relevant_data_filepaths["compact_object"][filetype][filename].append(filepath)

            for filetype in _MiscDataFilenames:
                re_match = re.fullmatch(filetype.value, filename)
                if re_match:
                    if filetype not in relevant_data_filepaths["misc"]:
                        relevant_data_filepaths["misc"][filetype] = {}
                    if filename not in relevant_data_filepaths["misc"][filetype]:
                        relevant_data_filepaths["misc"][filetype][filename] = []
                    relevant_data_filepaths["misc"][filetype][filename].append(filepath)

    return relevant_data_filepaths


def _all_relevant_output_filepaths(raw_directory: str) -> dict:
    """Dictionary of all relevant output files.

    The dictionary points from the suffixes included in _POSSIBLE_OUTPUT_SUFFIXES to a filename which in turn points to
    a list of filepaths with that filename.

    Args:
        raw_directory (str): the directory that contains all simulation data

    Returns:
        dict: dictionary containing suffix -> filename -> list of filepaths to relevant files

    """
    output_directories, _ = _ordered_output_directories(raw_directory)

    relevant_output_filepaths = {}
    # go through all output directories
    for output_directory in output_directories:
        data_path = os.path.join(output_directory, "*")

        # go through all datafiles in the output directory
        for filepath in glob.glob(data_path):
            # grab only the filename, not the full path
            filename = filepath.split('/')[-1]
            # for all data suffixes we may want

            for suffix in _POSSIBLE_OUTPUT_SUFFIXES:
                # if this file has that suffix
                if suffix in filename:
                    # if the key for the suffix doesn't already exist
                    if filename not in relevant_output_filepaths:
                        # create a dictionary with that suffix
                        relevant_output_filepaths[filename] = []
                    # add the filepath to the dictionary
                    relevant_output_filepaths[filename].append(filepath)

    return relevant_output_filepaths


def _stitch_timeseries_data(filepaths: list) -> np.ndarray:
    """Stitch together the data from a list of filepaths.

    Take the data from a list of filenames and stitch them together such that no times are repeated.

    Args:
        filepaths (list): all filenames to stitch together, ordered

    Returns:
        numpy.ndarray: array containing the stitched data

    """
    if len(filepaths) == 0:
        warnings.warn("There is no data to stitch.")
        return None

    # read the data from the first output
    stitched_timeseries = np.loadtxt(filepaths[0])
    if stitched_timeseries.ndim == 1:
        stitched_timeseries = stitched_timeseries.reshape(1, len(stitched_timeseries))

    # read and append data from remaining outputs
    if len(filepaths) > 1:
        for i in range(1, len(filepaths)):
            temp_timeseries = np.loadtxt(filepaths[i])
            if temp_timeseries.ndim == 1:
                temp_timeseries = temp_timeseries.reshape(1, len(temp_timeseries))

            # crop out duplicated times/iterations
            prev_last_iteration = stitched_timeseries[-1][0]
            if prev_last_iteration >= temp_timeseries[-1][0]:
                continue
            index_to_start = np.argmax(temp_timeseries[:, 0] > prev_last_iteration)
            cropped_timeseries = temp_timeseries[index_to_start:]
            stitched_timeseries = np.append(stitched_timeseries, cropped_timeseries, axis=0)

    if stitched_timeseries.shape[0] < 2:
        warnings.warn("This data file is empty")
        return None

    if len(stitched_timeseries[:, 0]) != len(set(stitched_timeseries[:, 0])):
        unique_values, unique_indices = np.unique(stitched_timeseries[:, 0], return_index=True)
        stitched_timeseries = stitched_timeseries[np.sort(unique_indices)]

    return stitched_timeseries


def _l_m_radius_from_psi4_filename(psi4_filename: str) -> tuple:
    """The values of l, m, and radius obtained from a :math:`\Psi_4` filename.

    Extracts and returns the l and m values of the mode and the value of the extraction radius from the filename.

    Args:
        psi4_filename (str): the filename containing :math:`\Psi_4` data

    Returns:
        tuple: l value of mode, m value of mode, extraction radius

    """
    result = re.search(r'_l(?P<l>-?\d)_m(?P<m>-?\d)_r(?P<r>\d+.\d+).', psi4_filename)
    l = int(result.group('l'))
    m = int(result.group('m'))
    r = result.group('r')
    return l, m, r


def _store_radiative_data(h5_file: h5py.File, relevant_filepaths: dict):
    """Add all radiative data to the radiative group within the h5 file.

    Stitch together all radiative data for each mode and extraction radii and store within the h5 file.

    Args:
        h5_file (h5py.File): the h5 file to store the data
        relevant_filepaths (dict): dictionary containing all relevant filepaths

    """

    radiative_group = h5_file.create_group("radiative")

    psi4_group = radiative_group.create_group("psi4")

    psi4_stored = False
    for filetype in sorted(relevant_filepaths["radiative"].keys()):
        if not psi4_stored:
            filetype.class_reference.store_radiative_data_from_filetype(relevant_filepaths, psi4_group)
            psi4_stored = True


def _store_compact_object_data(h5_file: h5py.File, relevant_filepaths: dict):
    """Store the compact object data in the h5 file.

    Stitches and combines all the compact object data and stores the data for each compact object in the provided
    h5 file.

    Args:
        h5_file (h5py.File): the h5 file to store the compact object data
        relevant_filepaths (dict): a dictionary containing all relevant filepaths

    """

    compact_object_group = h5_file.create_group("compact_object")

    parameter_file_group = h5_file['parfile']

    compact_object_dict = {}

    metadata_dict = {}

    for filetype in sorted(relevant_filepaths["compact_object"].keys()):

        compact_object_dict, metadata_dict_temp = filetype.class_reference.store_compact_object_data_from_filetype(
            relevant_filepaths["compact_object"][filetype], compact_object_dict, parameter_file_group)
        for key, value in metadata_dict_temp.items():
            metadata_dict[key] = value

    for compact_object_num in compact_object_dict.keys():
        compact_object_name = f"object={compact_object_num}"
        header = [col.header_text for col in compact_object_dict[compact_object_num].columns.values.tolist()]
        individual_compact_object_dataset = compact_object_group.create_dataset(compact_object_name,
                                                                                data=compact_object_dict[
                                                                                    compact_object_num].to_numpy(),
                                                                                fillvalue=np.nan)
        individual_compact_object_dataset.attrs["header"] = header

    for key, val in metadata_dict.items():
        compact_object_group.attrs[key] = val


def _get_data_from_columns(full_dataset: np.ndarray, header: list, column_names: list) -> np.ndarray:
    """Get the data in the requested columns (properties).

    Given a list of columns, returns the data at iterations where all columns have values.

    Args:
        column_names (list): list of desired columns, in the form of enums

    Returns:
        np.ndarray: the data at the requested columns

    """

    try:
        columns = [header.index(column.header_text) for column in column_names]
    except ValueError:
        return None

    nan_columns = np.isnan(full_dataset[:, columns]).any(axis=1)
    if len(columns) == 1:
        data = full_dataset[~nan_columns][:, columns[0]]
    else:
        data = full_dataset[~nan_columns][:, columns]
    return data


def _get_dimensional_spin_from_parfile(parfile_group: h5py.Group) -> tuple:
    """Provides the dimensional spin vectors (:math:`\pmb{S}` or :math:`\pmb{J}`) from the parameter file.

    Parses through the parameter file and returns the dimensional spin of both the primary and secondary compact
    objects.

    Args:
        parfile_group (h5py.Group): parameter file group in h5 file

    Returns:
        tuple: a tuple of the dimensional spin for the primary object and the dimensional spin for the secondary object

    """
    if 'par_content' not in parfile_group.attrs:
        warnings.warn('There is no parfile information in this simulation')
        return None

    spin0 = [np.nan, np.nan, np.nan]
    spin1 = [np.nan, np.nan, np.nan]
    parfile_content = parfile_group.attrs['par_content']
    for line in parfile_content.splitlines():
        if line.startswith("$spx"):
            spin0[0] = float(line.strip().split()[-1][:-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.startswith("$spy"):
            spin0[1] = float(line.strip().split()[-1][:-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.startswith("$spz"):
            spin0[2] = float(line.strip().split()[-1][:-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.startswith("$smx"):
            spin1[0] = float(line.strip().split()[-1][:-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.startswith("$smy"):
            spin1[1] = float(line.strip().split()[-1][:-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.startswith("$smz"):
            spin1[2] = float(line.strip().split()[-1][:-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.lower().startswith("twopunctures::par_s_plus[0]") and "$" not in line:
            spin0[0] = float(line.strip().split()[-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.lower().startswith("twopunctures::par_s_plus[1]") and "$" not in line:
            spin0[1] = float(line.strip().split()[-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.lower().startswith("twopunctures::par_s_plus[2]") and "$" not in line:
            spin0[2] = float(line.strip().split()[-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.lower().startswith("twopunctures::par_s_minus[0]") and "$" not in line:
            spin1[0] = float(line.strip().split()[-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.lower().startswith("twopunctures::par_s_minus[1]") and "$" not in line:
            spin1[1] = float(line.strip().split()[-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break
        elif line.lower().startswith("twopunctures::par_s_minus[2]") and "$" not in line:
            spin1[2] = float(line.strip().split()[-1])
            if not np.any(np.isnan(spin0)) and not np.any(np.isnan(spin1)):
                break

    for i, spin_component in enumerate(spin0):
        if np.isnan(spin_component):
            spin0[i] = 0

    for i, spin_component in enumerate(spin1):
        if np.isnan(spin_component):
            spin1[i] = 0

    return spin0, spin1


def _get_initial_dimensional_spins(h5_file: h5py.File) -> tuple:
    """Initial dimensional spins (:math:`\pmb{S}` or :math:`\pmb{J}`).

    Tries to get the initial dimensional spins from the compact object data. Otherwise, returns the spins set in the
    parameter file.

    Args:
        h5_file (h5py.File): h5 file containing all the data about the simulation

    Returns:
        tuple: a tuple of the dimensional spin for the primary object and the dimensional spin for the secondary object

    """
    compact_object_data_0 = h5_file["compact_object"]["object=0"]
    compact_object_data_1 = h5_file["compact_object"]["object=1"]

    try:
        dimensional_spin_data_0 = _get_data_from_columns(compact_object_data_0[()],
                                                         list(compact_object_data_0.attrs["header"]),
                                                         [CompactObject.Column.SX, CompactObject.Column.SY,
                                                          CompactObject.Column.SZ])
        dimensional_spin_data_1 = _get_data_from_columns(compact_object_data_1[()],
                                                         list(compact_object_data_1.attrs["header"]),
                                                         [CompactObject.Column.SX, CompactObject.Column.SY,
                                                          CompactObject.Column.SZ])

        if len(dimensional_spin_data_0) == 0 or len(dimensional_spin_data_1) == 0:
            if 'parfile' not in h5_file:
                warnings.warn('There is no parameter file data for this simulation.')
                return
            else:
                return _get_dimensional_spin_from_parfile(h5_file['parfile'])

    except ValueError as e:
        if 'parfile' not in h5_file:
            warnings.warn('There is no parameter file data for this simulation.')
            return
        else:
            return _get_dimensional_spin_from_parfile(h5_file['parfile'])

    spin0 = dimensional_spin_data_0[0].tolist()
    spin1 = dimensional_spin_data_1[0].tolist()

    return spin0, spin1


def _get_initial_dimensionless_spins(dimensional_spin_0: list, dimensional_spin_1: list,
                                     horizon_mass_0: float, horizon_mass_1: float):
    """Initial dimensionless spins (:math:`\pmb{a}` or :math:`\pmb{\chi} = \pmb{J}/M^2`).

    Computes and returns the dimensionless spins given the dimensional spins and the horizon masses.

    Args:
        dimensional_spin_0 (list): the dimensional spin of the primary compact object
        dimensional_spin_1 (list): the dimensional spin of the secondary compact object
        horizon_mass_0 (float): the horizon mass of the primary compact object
        horizon_mass_1 (float): the horizon mass of the secondary compact object

    Returns:
        tuple: a tuple of the dimensionless spin of the primary object and the dimensionless spin of the secondary
        object

    """
    dimensionless_spin_0 = [spin_component / (horizon_mass_0 * horizon_mass_0) for spin_component in
                            dimensional_spin_0]
    dimensionless_spin_1 = [spin_component / (horizon_mass_1 * horizon_mass_1) for spin_component in
                            dimensional_spin_1]
    return dimensionless_spin_0, dimensionless_spin_1


def _get_spin_configuration(dimensionless_spin_0: list, dimensionless_spin_1: list) -> str:
    """The spin configuration.

    Determines the spin configuration provided the dimensionless spins. Can be non-spinning, aligned-spins, precessing,
    or Unknown

    Args:
        dimensionless_spin_0 (list): the dimensionless spin of the primary compact object
        dimensionless_spin_1 (list): the dimensionless spin of the secondary compact object

    Returns:
        str: spin configuration

    """
    if np.any(np.isnan(dimensionless_spin_0)) or np.any(np.isnan(dimensionless_spin_1)):
        return "Unknown"

    if np.all(np.isclose(dimensionless_spin_0, 0, atol=1e-3)) and np.all(
            np.isclose(dimensionless_spin_1, 0, atol=5e-3)):
        return "non-spinning"
    elif not np.isclose(dimensionless_spin_0[0], 0, atol=1e-3) or not np.isclose(dimensionless_spin_0[1], 0,
                                                                                 atol=1e-3) \
            or not np.isclose(dimensionless_spin_1[0], 0, atol=1e-3) or not np.isclose(dimensionless_spin_1[1], 0,
                                                                                       atol=1e-3):
        return "precessing"
    else:
        return "aligned-spins"


def _irreducible_mass_to_horizon_mass(irreducible_mass: float, dimensional_spin: list) -> float:
    """Convert the irreducible mass to the horizon mass.

    Provided the irreducible mass and the dimensional spin, compute and return the horizon mass as
    :math:`M_{horizon} = sqrt(M_{irr}^2 + \frac{S^2}{4M_{irr}^2})`

    Args:
        irreducible_mass (float): the irreducible mass of the object
        dimensional_spin (list): the dimensional spin of the object

    Returns:
        float: the horizon mass of the object

    """
    spin_magnitude = np.linalg.norm(dimensional_spin)

    horizon_mass = np.sqrt(irreducible_mass ** 2 + spin_magnitude ** 2 / (4 * irreducible_mass ** 2))

    return horizon_mass


def _horizon_mass_to_irreducible_mass(horizon_mass: float, dimensional_spin: list):
    """Convert the horizon mass to the irreducible mass.

    Provided the horizon mass and the dimensional spin, compute and return the irreducible mass as
    :math:`M_{irr} = sqrt((M_{horizon}^2 + sqrt(M_{horizon}^4 - S^2))/2)`

    Args:
        horizon_mass (float): the horizon mass of the object
        dimensional_spin (list): the dimensional spin of the object

    Returns:
        float: the irreducible mass of the object

    """
    spin_magnitude = np.linalg.norm(dimensional_spin)

    irreducible_mass = np.sqrt((horizon_mass ** 2 + np.sqrt(horizon_mass ** 4 - spin_magnitude ** 2)) / 2)

    return irreducible_mass


def _get_masses_from_out_file(relevant_output_filepaths: dict, dimensional_spin_0: list, dimensional_spin_1: list):
    """Values of the irreducible and horizon masses from the .out files.

    Parses through the .out file to locate the horizon masses and then computes the irreducible masses.

    Args:
        relevant_output_filepaths (dict): dictionary containing all the relevant .out files
        dimensional_spin_0 (list): the dimensional spin of the primary compact object
        dimensional_spin_1 (list): the dimensional spin of the secondary compact object

    Returns:
        tuple: tuple containing the irreducible mass of the primary object, the irreducible mass of the secondary
            object, the horizon mass of the primary object, and the horizon mass of the secondary object

    """
    horizon_mass0 = np.nan
    horizon_mass1 = np.nan

    for output_filename in relevant_output_filepaths:
        if "out" in output_filename:
            for filename in relevant_output_filepaths[output_filename]:
                with open(filename) as f:
                    for line in f:
                        if line.startswith("INFO (TwoPunctures): ADM mass for puncture: m+"):
                            horizon_mass0 = float(line.strip()[line.find('=') + 1:])
                            if not np.isnan(horizon_mass0) and not np.isnan(horizon_mass1):
                                break
                        elif line.startswith("INFO (TwoPunctures): ADM mass for puncture: m-"):
                            horizon_mass1 = float(line.strip()[line.find('=') + 1:])
                            if not np.isnan(horizon_mass0) and not np.isnan(horizon_mass1):
                                break

    irreducible_mass0 = _horizon_mass_to_irreducible_mass(horizon_mass0, dimensional_spin_0)
    irreducible_mass1 = _horizon_mass_to_irreducible_mass(horizon_mass1, dimensional_spin_1)

    return irreducible_mass0, irreducible_mass1, horizon_mass0, horizon_mass1


def _get_initial_masses(h5_file: h5py.File, dimensional_spin0: list, dimensional_spin1: list,
                        relevant_output_filepaths: dict) -> tuple:
    """Values of the irreducible masses and horizon masses.

    Obtain the initial masses from either the compact object data or the .out files.

    Args:
        h5_file (h5py.File): h5 file containing all the data about the simulation
        dimensional_spin0 (list): the dimensional spin of the primary compact object
        dimensional_spin1 (list): the dimensional spin of the secondary compact object
        relevant_output_filepaths (dict): dictionary containing all the relevant .out files

    Returns:
        tuple: irreducible mass of primary object, irreducible mass of secondary object, horizon mass of primary object,
            horizon mass of secondary object

    """
    compact_object_data_0 = h5_file["compact_object"]["object=0"]
    compact_object_data_1 = h5_file["compact_object"]["object=1"]

    try:
        irreducible_mass_data_0 = _get_data_from_columns(compact_object_data_0[()],
                                                         list(compact_object_data_0.attrs["header"]),
                                                         [CompactObject.Column.M_IRREDUCIBLE])
        irreducible_mass_data_1 = _get_data_from_columns(compact_object_data_1[()],
                                                         list(compact_object_data_1.attrs["header"]),
                                                         [CompactObject.Column.M_IRREDUCIBLE])

        if irreducible_mass_data_0 is None or irreducible_mass_data_1 is None or len(
                irreducible_mass_data_0) == 0 or len(irreducible_mass_data_1) == 0:
            print('using out file due to length')
            return _get_masses_from_out_file(relevant_output_filepaths, dimensional_spin0, dimensional_spin1)

        irreducible_mass0 = irreducible_mass_data_0[0]
        irreducible_mass1 = irreducible_mass_data_1[0]

        horizon_mass0 = _irreducible_mass_to_horizon_mass(irreducible_mass0, dimensional_spin0)
        horizon_mass1 = _irreducible_mass_to_horizon_mass(irreducible_mass1, dimensional_spin1)

        return float(irreducible_mass0), float(irreducible_mass1), float(horizon_mass0), float(horizon_mass1)

    except ValueError as e:
        print('using out file due to value error')
        return _get_masses_from_out_file(relevant_output_filepaths, dimensional_spin0, dimensional_spin1)


def _get_initial_separation_from_parfile(parfile_group: h5py.Group) -> float:
    """Obtain the initial separation from the parameter file

    Find the initial coordinate separation from the .rpar or .par file

    Args:
        parfile_group (h5py.Group): parameter file for the simulation

    Returns:
        float: initial separation

    """
    initial_separation = np.nan

    if 'par_content' not in parfile_group.attrs:
        warnings.warn('There is no parameter file information for this simulation.')
        return

    parfile_content = parfile_group.attrs['par_content']
    for line in parfile_content.splitlines():
        if line.startswith("TwoPunctures::par_b"):
            separation_data = line.split()[-1]
            initial_separation = 2 * float(separation_data)
            break

    return initial_separation


def _get_initial_separation(h5_file: h5py.File) -> float:
    """Initial separation for the simulation.

    Obtain the initial coordinate separation of the simulation either from the coordinate trajectories of the black
    holes or from the parameter file.

    Args:
        h5_file (h5py.File): h5 file containing all the data about the simulation

    Returns:
        float: initial separation

    """
    compact_object_data_0 = h5_file["compact_object"]["object=0"]
    compact_object_data_1 = h5_file["compact_object"]["object=1"]

    try:
        position_0_data = _get_data_from_columns(compact_object_data_0[()],
                                                 list(compact_object_data_0.attrs["header"]),
                                                 [CompactObject.Column.TIME, CompactObject.Column.X,
                                                  CompactObject.Column.Y, CompactObject.Column.Z])

        position_1_data = _get_data_from_columns(compact_object_data_1[()],
                                                 list(compact_object_data_0.attrs["header"]),
                                                 [CompactObject.Column.TIME, CompactObject.Column.X,
                                                  CompactObject.Column.Y, CompactObject.Column.Z])

        if position_0_data is None or position_1_data is None or len(position_0_data) == 0 or len(position_1_data) == 0:
            if 'parfile' not in h5_file:
                warnings.warn(
                    'There is no parameter file data for this simulation. Cannot compute the initial separation.')
                return
            else:
                return _get_initial_separation_from_parfile(h5_file['parfile'])

    except ValueError as e:
        if 'parfile' not in h5_file:
            warnings.warn('There is no parameter file data for this simulation. Cannot compute the initial separation.')
            return
        else:
            return _get_initial_separation_from_parfile(h5_file['parfile'])

    time_0 = position_0_data[:, 0]
    time_1 = position_1_data[:, 0]
    position_0 = position_0_data[:, 1:]
    position_1 = position_1_data[:, 1:]

    time, time_indices_0, time_indices_1 = np.intersect1d(time_0, time_1, assume_unique=True, return_indices=True)

    position_0 = position_0[time_indices_0]
    position_1 = position_1[time_indices_1]

    initial_position_0 = position_0[0]
    initial_position_1 = position_1[0]

    initial_separation_vector = initial_position_1 - initial_position_0

    initial_separation_mag = np.linalg.norm(initial_separation_vector)

    return initial_separation_mag


def _get_initial_orbital_frequency(h5_file: h5py.File) -> float:
    """Initial orbital frequency for the simulation.

    Obtain the initial orbital frequency of the simulation from the coordinate separation vector of the black holes.
    Waits 75M for junk radiation to settle.

    Args:
        h5_file (h5py.File): h5 file containing all the data about the simulation

    Returns:
        float: orbital frequency after junk

    """
    compact_object_data_0 = h5_file["compact_object"]["object=0"]
    compact_object_data_1 = h5_file["compact_object"]["object=1"]

    try:
        position_0_data = _get_data_from_columns(compact_object_data_0[()],
                                                 list(compact_object_data_0.attrs["header"]),
                                                 [CompactObject.Column.TIME, CompactObject.Column.X,
                                                  CompactObject.Column.Y, CompactObject.Column.Z])

        position_1_data = _get_data_from_columns(compact_object_data_1[()],
                                                 list(compact_object_data_1.attrs["header"]),
                                                 [CompactObject.Column.TIME, CompactObject.Column.X,
                                                  CompactObject.Column.Y, CompactObject.Column.Z])

        if position_0_data is None or position_1_data is None or len(position_0_data) == 0 or len(position_1_data) == 0:
            return np.nan

    except ValueError as e:
        return np.nan

    time_0 = position_0_data[:, 0]
    time_1 = position_1_data[:, 0]
    position_0 = position_0_data[:, 1:]
    position_1 = position_1_data[:, 1:]

    time, time_indices_0, time_indices_1 = np.intersect1d(time_0, time_1, assume_unique=True, return_indices=True)
    position_0 = position_0[time_indices_0]
    position_1 = position_1[time_indices_1]

    separation_vector = position_1 - position_0

    # flip nhat to it points from secondary to primary
    nhat = -1 * separation_vector / np.linalg.norm(separation_vector, axis=1).reshape(separation_vector.shape[0], 1)
    dnhat_dt = np.gradient(nhat, axis=0) / (np.gradient(time).reshape(nhat.shape[0], 1))
    orbital_frequency = np.cross(nhat, dnhat_dt, axis=1)
    mag_orbital_frequency = np.linalg.norm(orbital_frequency, axis=1)

    # grab the frequency after junk so that it has settled
    if time[-1] <= 75:
        return None
    index_75M = np.argmax(time > 75)
    return mag_orbital_frequency[index_75M]


def _store_meta_data(h5_file: h5py.File, relevant_data_filepaths: dict, relevant_output_filepaths: dict,
                     simulation_name: str, catalog_id: str = None):
    """Store the metadata in the h5 file

    Compute and store the metadata (name, initial spins, initial masses, etc) in the h5 file

    Args:
        h5_file (h5py.File): h5 file containing all the data about the simulation
        relevant_data_filepaths (dict): dictionary containing all the relevant data files
        relevant_output_filepaths (dict): dictionary containing all the relevant .out files
        simulation_name (str): the name of the simulation
        catalog_id (:obj:`str`, optional): id of the simulation within a catalog

    """
    # store runstats
    if 'misc' in relevant_data_filepaths.keys():
        if _MiscDataFilenames.RUNSTATS in relevant_data_filepaths["misc"].keys():
            for filename, filepaths in relevant_data_filepaths["misc"][_MiscDataFilenames.RUNSTATS].items():
                data = _stitch_timeseries_data(filepaths)
                runstats_dataset = h5_file.create_dataset("runstats", data=data)
                header_array = ["iteration", "coord_time", "wall_time", "speed (hours^-1)", "period (minutes)",
                                "cputime (cpu hours)"]
                runstats_dataset.attrs["header"] = header_array

    # store attributes
    # name
    h5_file.attrs["name"] = simulation_name
    if catalog_id is not None:
        h5_file.attrs["catalog id"] = catalog_id
    # initial dimensional spins
    dimensional_spin_0, dimensional_spin_1 = _get_initial_dimensional_spins(h5_file)
    h5_file.attrs["dimensional spin 0"] = dimensional_spin_0
    h5_file.attrs["dimensional spin 1"] = dimensional_spin_1
    # initial masses
    irreducible_mass0, irreducible_mass1, horizon_mass_0, horizon_mass_1 = _get_initial_masses(
        h5_file, dimensional_spin_0, dimensional_spin_1, relevant_output_filepaths)
    h5_file.attrs["irreducible mass 0"] = irreducible_mass0
    h5_file.attrs["irreducible mass 1"] = irreducible_mass1
    h5_file.attrs["horizon mass 0"] = horizon_mass_0
    h5_file.attrs["horizon mass 1"] = horizon_mass_1
    # mass ratio
    h5_file.attrs["mass ratio"] = horizon_mass_0 / horizon_mass_1
    # initial dimensionless spins
    dimensionless_spin_0, dimensionless_spin_1 = _get_initial_dimensionless_spins(
        dimensional_spin_0, dimensional_spin_1, horizon_mass_0, horizon_mass_1)
    h5_file.attrs["dimensionless spin 0"] = dimensionless_spin_0
    h5_file.attrs["dimensionless spin 1"] = dimensionless_spin_1
    # spin configuration
    h5_file.attrs["spin configuration"] = _get_spin_configuration(dimensionless_spin_0,
                                                                  dimensionless_spin_1)
    # initial separation
    h5_file.attrs["initial separation"] = _get_initial_separation(h5_file)
    # initial frequency
    h5_file.attrs["initial orbital frequency"] = _get_initial_orbital_frequency(h5_file)


def create_h5_from_simulation(raw_directory: str, output_directory: str, catalog_id: str = None) -> str:
    """Create a Mayawaves compatible h5 file storing all important information from the raw simulation.

    Stitch and store the simulation data from the raw simulation into a h5 file which can then be read by the
    Coalescence class.

    Args:
        raw_directory (str): directory of the raw simulation
        output_directory (str): directory to store the created h5 file
        catalog_id (:obj:`str`, optional): id for the simulation within a catalog

    Returns:
        str: path to the generated h5 file

    """
    raw_directory = raw_directory.rstrip('/')

    # determine the name of the simulation
    if not os.path.isdir(raw_directory):
        warnings.warn('That is not a simulation directory. Aborting.')
        return

    simulation_name = _simulation_name(raw_directory)
    parameter_file_name_base, parameter_file_dict = _get_parameter_file_name_and_content(raw_directory)

    if catalog_id is not None:
        h5_filename = os.path.join(output_directory, catalog_id + ".h5")
    else:
        h5_filename = os.path.join(output_directory, simulation_name + ".h5")

    # open h5file
    h5_file = h5py.File(h5_filename, 'w')

    # store parameter file
    print("storing parameter file")
    _store_parameter_file(parameter_file_dict, h5_file)

    # get all relevant filepaths
    if "parfile" in h5_file.keys():
        if 'par_content'in h5_file["parfile"].attrs:
            parameter_file = h5_file["parfile"].attrs["par_content"]
        elif 'rpar_content'in h5_file["parfile"].attrs:
            parameter_file = h5_file["parfile"].attrs["rpar_content"]
        else:
            parameter_file = None
    relevant_data_filepaths = _all_relevant_data_filepaths(raw_directory, parameter_file, parameter_file_name_base)
    relevant_output_filepaths = _all_relevant_output_filepaths(raw_directory)

    # process radiative data
    print("storing radiative information")
    _store_radiative_data(h5_file, relevant_data_filepaths)

    # process compact object data
    print("storing compact object information")
    _store_compact_object_data(h5_file, relevant_data_filepaths)

    # process metadata
    print("storing metadata")
    _store_meta_data(h5_file, relevant_data_filepaths, relevant_output_filepaths,
                     simulation_name, catalog_id)

    # close h5file
    h5_file.close()

    return h5_filename


def get_stitched_data(raw_directory: str, filename: str) -> np.ndarray:
    """Stitch together the data for a given file.

    Stitch the data from all outputs for a given file and return the full data as a numpy array with the same number of
    columns as the original files.

    Args:
        raw_directory (str): directory of the raw simulation
        filename (str): filename to be stitched

    Returns:
        numpy.ndarray: stitched data for the given filename

    """

    if not os.path.isdir(raw_directory):
        warnings.warn("That directory does not exist")
        return None

    parameter_file_name_base, parameter_file_dict = _get_parameter_file_name_and_content(raw_directory)

    if parameter_file_dict is None or 'par_content' not in parameter_file_dict:
        warnings.warn('Unable to determine file structure due to lack of parameter file')
        return None

    data_directories = _ordered_data_directories(raw_directory, parameter_file=parameter_file_dict['par_content'], parameter_file_name_base=parameter_file_name_base)
    filepaths = []

    # go through all output directories
    for output_directory in data_directories:
        matching_filepaths = glob.glob(os.path.join(output_directory, filename))
        if len(matching_filepaths) > 0:
            filepath = glob.glob(os.path.join(output_directory, filename))[0]
            filepaths.append(filepath)

    if len(filepaths) == 0:
        warnings.warn('There are no files with that name in the output directory')
        return None

    print(f'Stitching {len(filepaths)} files')

    return _stitch_timeseries_data(filepaths)


def _nr_frequency_to_physical(nr_frequency: float, mass: float) -> float:
    """Convert NR frequency to physical units (Hz).

    Takes the frequency in geometric units and uses the total mass to compute the physical frequency in Hz.

    Args:
        nr_frequency (float): the frequency in geometric units
        mass (float): the total mass

    Returns:
        float: the frequency in Hz

    """
    G = 6.67428e-11  # m^3/(kg s^2)
    mass_sun = 1.98892e30  # kg
    c = 2.99792458e8  # m/s
    mass_sec = G * mass_sun / (c ** 3)
    simulation_mass_sec = mass * mass_sec
    physical_freq = nr_frequency / simulation_mass_sec
    return physical_freq


def _crop_between_times(time: np.ndarray, data: np.ndarray, start_time: float, end_time: float) -> tuple:
    """Crop the time and data between the given times (inclusive)

    Args:
        time (numpy.ndarray): time stamps associated with data
        data (numpy.ndarray): timeseries to crop
        start_time (float): time at which to begin cropping data
        end_time (float): time at which to end cropping data

    Returns:
        tuple: cropped time, cropped data

    """
    if start_time > time[-1]:
        return time[:0], data[:0]

    if end_time >= time[-1]:
        start_index = np.argmax(time >= start_time)
        cropped_time = time[start_index:]
        cropped_data = data[start_index:]
        return cropped_time, cropped_data

    start_index = np.argmax(time >= start_time)
    end_index = np.argmax(time > end_time)

    cropped_time = time[start_index: end_index]
    cropped_data = data[start_index: end_index]
    return cropped_time, cropped_data


def _find_first_significant_gap_time(time: np.ndarray) -> float:
    """Find the first gap that occurs in the time data

    Args:
        time (numpy.ndarray): time data

    Returns:
        float: last time before the gap

    """
    step_sizes = time[1:] - time[:-1]
    first_step_size = step_sizes[0]
    first_gap_index = np.argmax(np.isclose(step_sizes, first_step_size, rtol=5) == False)
    if first_gap_index == 0:
        first_gap_index = -1
    first_gap_time = time[first_gap_index]
    return first_gap_time


def _find_last_significant_gap(time: np.ndarray) -> float:
    """Find the last gap that occurs in the time data

    Args:
        time (numpy.ndarray): time data

    Returns:
        float: first time after the last significant gap in time data

    """
    time = np.flip(time)
    step_sizes = time[1:] - time[:-1]
    first_step_size = step_sizes[0]
    first_gap_index = np.argmax(np.isclose(step_sizes, first_step_size, rtol=5) == False)
    if first_gap_index == 0:
        first_gap_index = -1
    first_gap_time = time[first_gap_index]
    return first_gap_time


def _store_compact_object_timeseries_data(coalescence: Coalescence, lal_h5_file: h5py.File, lvc_format: int,
                                          time_shift: float, initial_horizon_time: float):
    """Store all timeseries data relating to the compact objects

    Store the timeseries data for masses, spins, and trajectories of the compact objects as well as the orbital
    frequency and angular momentum vector

    Args:
        coalescence (Coalescence): the coalescence object for the simulation
        lal_h5_file (h5py.File): the h5 file to store the data in
        lvc_format (int): the format (1, 2, 3) of the data. This informs what data to store.
        time_shift (float): time shift to apply to the data
        initial_horizon_time (float): initial time to crop the data at

    """
    merge_time = coalescence.merge_time
    low_pass_freq_cutoff = coalescence.orbital_frequency_at_time(merge_time) * 5

    if lvc_format == 2 or lvc_format == 3:
        primary_object = coalescence.primary_compact_object

        print("Storing mass1-vs-time")
        time_mass, primary_mass = primary_object.horizon_mass
        first_gap_time = _find_first_significant_gap_time(time_mass)
        time_mass, primary_mass = _crop_between_times(time_mass, primary_mass, initial_horizon_time,
                                                      min(first_gap_time, merge_time))
        primary_mass = low_pass_filter(time_mass, primary_mass, low_pass_freq_cutoff)
        time_mass = time_mass - time_shift
        spline = romspline.ReducedOrderSpline(time_mass, primary_mass, verbose=False)
        primary_mass_group = lal_h5_file.create_group("mass1-vs-time")
        spline.write(primary_mass_group)

        time_spin, primary_spin = primary_object.dimensionless_spin_vector
        first_gap_time = _find_first_significant_gap_time(time_spin)
        time_spin, primary_spin = _crop_between_times(time_spin, primary_spin, initial_horizon_time,
                                                      min(first_gap_time, merge_time))
        primary_spin = low_pass_filter(time_spin, primary_spin, low_pass_freq_cutoff)
        time_spin = time_spin - time_shift

        print("Storing spin1x-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, primary_spin[:, 0], verbose=False)
        primary_spinx_group = lal_h5_file.create_group("spin1x-vs-time")
        spline.write(primary_spinx_group)

        print("Storing spin1y-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, primary_spin[:, 1], verbose=False)
        primary_spiny_group = lal_h5_file.create_group("spin1y-vs-time")
        spline.write(primary_spiny_group)

        print("Storing spin1z-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, primary_spin[:, 2], verbose=False)
        primary_spinz_group = lal_h5_file.create_group("spin1z-vs-time")
        spline.write(primary_spinz_group)

        time_position, primary_position = primary_object.position_vector
        first_gap_time = _find_first_significant_gap_time(time_position)
        time_position, primary_position = _crop_between_times(time_position, primary_position, initial_horizon_time,
                                                              min(first_gap_time, merge_time))
        primary_position = low_pass_filter(time_position, primary_position, low_pass_freq_cutoff)
        time_position = time_position - time_shift

        print("Storing position1x-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, primary_position[:, 0], verbose=False)
        primary_positionx_group = lal_h5_file.create_group("position1x-vs-time")
        spline.write(primary_positionx_group)

        print("Storing position1y-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, primary_position[:, 1], verbose=False)
        primary_positiony_group = lal_h5_file.create_group("position1y-vs-time")
        spline.write(primary_positiony_group)

        print("Storing position1z-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, primary_position[:, 2], verbose=False)
        primary_positionz_group = lal_h5_file.create_group("position1z-vs-time")
        spline.write(primary_positionz_group)

        secondary_object = coalescence.secondary_compact_object

        print("Storing mass2-vs-time")
        time_mass, secondary_mass = secondary_object.horizon_mass
        first_gap_time = _find_first_significant_gap_time(time_mass)
        time_mass, secondary_mass = _crop_between_times(time_mass, secondary_mass, initial_horizon_time,
                                                        min(first_gap_time, merge_time))
        secondary_mass = low_pass_filter(time_mass, secondary_mass, low_pass_freq_cutoff)
        time_mass = time_mass - time_shift
        spline = romspline.ReducedOrderSpline(time_mass, secondary_mass, verbose=False)
        secondary_mass_group = lal_h5_file.create_group("mass2-vs-time")
        spline.write(secondary_mass_group)

        time_spin, secondary_spin = secondary_object.dimensionless_spin_vector
        first_gap_time = _find_first_significant_gap_time(time_spin)
        time_spin, secondary_spin = _crop_between_times(time_spin, secondary_spin, initial_horizon_time,
                                                        min(first_gap_time, merge_time))
        secondary_spin = low_pass_filter(time_spin, secondary_spin, low_pass_freq_cutoff)
        time_spin = time_spin - time_shift

        print("Storing spin2x-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, secondary_spin[:, 0], verbose=False)
        secondary_spinx_group = lal_h5_file.create_group("spin2x-vs-time")
        spline.write(secondary_spinx_group)

        print("Storing spin2y-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, secondary_spin[:, 1], verbose=False)
        secondary_spiny_group = lal_h5_file.create_group("spin2y-vs-time")
        spline.write(secondary_spiny_group)

        print("Storing spin2z-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, secondary_spin[:, 2], verbose=False)
        secondary_spinz_group = lal_h5_file.create_group("spin2z-vs-time")
        spline.write(secondary_spinz_group)

        time_position, secondary_position = secondary_object.position_vector
        first_gap_time = _find_first_significant_gap_time(time_position)
        time_position, secondary_position = _crop_between_times(time_position, secondary_position, initial_horizon_time,
                                                                min(first_gap_time, merge_time))
        secondary_position = low_pass_filter(time_position, secondary_position, low_pass_freq_cutoff)
        time_position = time_position - time_shift

        print("Storing position2x-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, secondary_position[:, 0], verbose=False)
        secondary_positionx_group = lal_h5_file.create_group("position2x-vs-time")
        spline.write(secondary_positionx_group)

        print("Storing position2y-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, secondary_position[:, 1], verbose=False)
        secondary_positiony_group = lal_h5_file.create_group("position2y-vs-time")
        spline.write(secondary_positiony_group)

        print("Storing position2z-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, secondary_position[:, 2], verbose=False)
        secondary_positionz_group = lal_h5_file.create_group("position2z-vs-time")
        spline.write(secondary_positionz_group)

        time_lhat, lhat = coalescence.orbital_angular_momentum_unit_vector
        first_gap_time = _find_first_significant_gap_time(time_lhat)
        time_lhat, lhat = _crop_between_times(time_lhat, lhat, initial_horizon_time, min(first_gap_time, merge_time))
        lhat = low_pass_filter(time_lhat, lhat, low_pass_freq_cutoff)
        time_lhat = time_lhat - time_shift

        print("Storing LNhatx-vs-time")
        spline = romspline.ReducedOrderSpline(time_lhat, lhat[:, 0], verbose=False)
        lhatx_group = lal_h5_file.create_group("LNhatx-vs-time")
        spline.write(lhatx_group)

        print("Storing LNhaty-vs-time")
        spline = romspline.ReducedOrderSpline(time_lhat, lhat[:, 1], verbose=False)
        lhaty_group = lal_h5_file.create_group("LNhaty-vs-time")
        spline.write(lhaty_group)

        print("Storing LNhatz-vs-time")
        spline = romspline.ReducedOrderSpline(time_lhat, lhat[:, 2], verbose=False)
        lhatz_group = lal_h5_file.create_group("LNhatz-vs-time")
        spline.write(lhatz_group)

        print("Storing Omega-vs-time")
        time_orb_freq, orb_freq = coalescence.orbital_frequency
        first_gap_time = _find_first_significant_gap_time(time_orb_freq)
        time_orb_freq, orb_freq = _crop_between_times(time_orb_freq, orb_freq, initial_horizon_time, min(first_gap_time,
                                                                                                         merge_time))
        orb_freq = low_pass_filter(time_orb_freq, orb_freq, low_pass_freq_cutoff)
        time_orb_freq = time_orb_freq - time_shift
        spline = romspline.ReducedOrderSpline(time_orb_freq, orb_freq, verbose=False)
        omega_group = lal_h5_file.create_group("Omega-vs-time")
        spline.write(omega_group)

    if lvc_format == 3:
        final_object = coalescence.final_compact_object

        print("Storing remnant-mass-vs-time")
        time_mass, remnant_mass = final_object.horizon_mass
        gap_time = _find_last_significant_gap(time_mass)
        time_mass, remnant_mass = _crop_between_times(time_mass, remnant_mass, max(gap_time, time_shift + 20),
                                                      time_mass[-1])
        remnant_mass = low_pass_filter(time_mass, remnant_mass, low_pass_freq_cutoff)
        time_mass = time_mass - time_shift
        spline = romspline.ReducedOrderSpline(time_mass, remnant_mass, verbose=False)
        remnant_mass_group = lal_h5_file.create_group("remnant-mass-vs-time")
        spline.write(remnant_mass_group)

        time_spin, remnant_spin = final_object.dimensionless_spin_vector
        gap_time = _find_last_significant_gap(time_spin)
        time_spin, remnant_spin = _crop_between_times(time_spin, remnant_spin, max(gap_time, time_shift + 20),
                                                      time_spin[-1])
        remnant_spin = low_pass_filter(time_spin, remnant_spin, low_pass_freq_cutoff)
        time_spin = time_spin - time_shift

        print("Storing remnant-spinx-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, remnant_spin[:, 0], verbose=False)
        remnant_spinx_group = lal_h5_file.create_group("remnant-spinx-vs-time")
        spline.write(remnant_spinx_group)

        print("Storing remnant-spiny-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, remnant_spin[:, 1], verbose=False)
        remnant_spiny_group = lal_h5_file.create_group("remnant-spiny-vs-time")
        spline.write(remnant_spiny_group)

        print("Storing remnant-spinz-vs-time")
        spline = romspline.ReducedOrderSpline(time_spin, remnant_spin[:, 2], verbose=False)
        remnant_spinz_group = lal_h5_file.create_group("remnant-spinz-vs-time")
        spline.write(remnant_spinz_group)

        time_position, remnant_position = final_object.position_vector
        gap_time = _find_last_significant_gap(time_position)
        time_position, remnant_position = _crop_between_times(time_position, remnant_position,
                                                              max(gap_time, time_shift + 20), time_position[-1])
        remnant_position = low_pass_filter(time_position, remnant_position, low_pass_freq_cutoff)
        time_position = time_position - time_shift

        print("Storing remnant-positionx-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, remnant_position[:, 0], verbose=False)
        remnant_positionx_group = lal_h5_file.create_group("remnant-positionx-vs-time")
        spline.write(remnant_positionx_group)

        print("Storing remnant-positiony-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, remnant_position[:, 1], verbose=False)
        remnant_positiony_group = lal_h5_file.create_group("remnant-positiony-vs-time")
        spline.write(remnant_positiony_group)

        print("Storing remnant-positionz-vs-time")
        spline = romspline.ReducedOrderSpline(time_position, remnant_position[:, 2], verbose=False)
        remnant_positionz_group = lal_h5_file.create_group("remnant-positionz-vs-time")
        spline.write(remnant_positionz_group)


def _store_lal_metadata(coalescence: Coalescence, lal_h5_file, name: str, alternative_names: list,
                        initial_time_horizon: float, omega_22_nr: float, lvc_format: int, NR_group: str, NR_code: str,
                        bibtex_keys: str, contact_email: str, license_type: str = 'LVC-internal', nr_techniques: str = None,
                        comparable_simulation: str = None, files_in_error_series: str = '', production_run: bool = True):
    """Store the metadata in the provided h5 file.

    Args:
        coalescence (Coalescence): the Coalescence object being exported
        lal_h5_file (h5py.File): the h5 file being exported to
        name (str): the name of the simulation
        alternative_names (list): alternative names for the simulation
        initial_time_horizon (float): the initial horizon time to report the metadata
        omega_22_nr (float): the initial frequency of the 22 mode
        lvc_format (int): 1, 2, or 3, based upon the metadata and timeseries provided
        NR_group (str): NR group that performed this simulation
        NR_code (str): NR code that performed this simulation
        bibtex_keys (str): bibtex keys to use when citing this simulation
        contact_email (str): email to use if questions arise regarding this simulation
        license_type (:obj:`str`, optional): whether it is public or LVC-internal
        nr_techniques (:obj:`str`, optional): what techniques were used in this simulation
        comparable_simulation (:obj:`str`, optional): other similar simulations
        files_in_error_series (:obj:`str`, optional): other simulations in the same error series
        production_run (:obj:`bool`, optional): whether this is a production run. Default True.

    """
    lal_h5_file.attrs["Format"] = lvc_format
    lal_h5_file.attrs["type"] = "NRinjection"
    lal_h5_file.attrs["name"] = name
    lal_h5_file.attrs["alternative-names"] = ",".join(alternative_names)
    lal_h5_file.attrs["NR-code"] = NR_code
    lal_h5_file.attrs["NR-group"] = NR_group
    lal_h5_file.attrs["PN_approximant"] = "None"
    lal_h5_file.attrs["Warning"] = "Please do not use m=0 modes. For caution, they are set to zero!"

    today = date.today()
    lal_h5_file.attrs["modification-date"] = today.strftime("%Y-%m-%d")

    lal_h5_file.attrs["point-of-contact-email"] = contact_email
    lal_h5_file.attrs["simulation-type"] = coalescence.spin_configuration
    lal_h5_file.attrs["INSPIRE-bibtex-keys"] = bibtex_keys
    lal_h5_file.attrs["license"] = license_type
    lal_h5_file.attrs["Lmax"] = coalescence.l_max
    if nr_techniques is None:
        nr_techniques = 'Puncture-ID, BSSN, Psi4-integrated, Extrapolated-Waveform, ApproxKillingVector-Spin, ' \
                        'Christodoulou-Mass'
    lal_h5_file.attrs["NR-techniques"] = nr_techniques
    lal_h5_file.attrs["files-in-error-series"] = files_in_error_series

    if comparable_simulation is None:
        if coalescence.spin_configuration == "precessing":
            comparable_simulation = 'GT0560'
        else:
            comparable_simulation = 'GT0582'
    lal_h5_file.attrs["comparable-simulation"] = comparable_simulation

    lal_h5_file.attrs["production-run"] = int(production_run)
    lal_h5_file.attrs["object1"] = "BH"
    lal_h5_file.attrs["object2"] = "BH"

    raise_io_error = False
    io_error_cause = ""

    last_available_spin_time = min(coalescence.primary_compact_object.last_available_spin_data_time,
                                   coalescence.secondary_compact_object.last_available_spin_data_time)

    if last_available_spin_time == 0:
        mass1 = coalescence.primary_compact_object.initial_horizon_mass
        mass2 = coalescence.secondary_compact_object.initial_horizon_mass
        spin1 = coalescence.primary_compact_object.initial_dimensionless_spin
        spin2 = coalescence.secondary_compact_object.initial_dimensionless_spin
        if mass1 is None or mass2 is None:
            raise_io_error = True
            io_error_cause = "mass"
        elif spin1 is None or spin2 is None:
            raise_io_error = True
            io_error_cause = "spin"
        elif coalescence.spin_configuration == "precessing":
            raise_io_error = True
            io_error_cause = "spin_precession"
    elif last_available_spin_time < initial_time_horizon:
        mass1 = coalescence.primary_compact_object.horizon_mass_at_time(last_available_spin_time)
        mass2 = coalescence.secondary_compact_object.horizon_mass_at_time(last_available_spin_time)
        spin1 = coalescence.primary_compact_object.dimensionless_spin_at_time(last_available_spin_time)
        spin2 = coalescence.secondary_compact_object.dimensionless_spin_at_time(last_available_spin_time)
        if mass1 is None or mass2 is None:
            raise_io_error = True
            io_error_cause = "mass"
        elif spin1 is None or spin2 is None:
            raise_io_error = True
            io_error_cause = "spin"
        if coalescence.spin_configuration == "precessing":
            raise_io_error = True
            io_error_cause = "spin_precession"

    else:
        # computed metadata
        mass1 = coalescence.primary_compact_object.horizon_mass_at_time(initial_time_horizon)
        spin1 = coalescence.primary_compact_object.dimensionless_spin_at_time(initial_time_horizon)
        mass2 = coalescence.secondary_compact_object.horizon_mass_at_time(initial_time_horizon)
        spin2 = coalescence.secondary_compact_object.dimensionless_spin_at_time(initial_time_horizon)
        if mass1 is None or mass2 is None:
            raise_io_error = True
            io_error_cause = "mass"
        elif spin1 is None or spin2 is None:
            raise_io_error = True
            io_error_cause = "spin"

    lal_h5_file.attrs["mass1"] = round(mass1, 3) if mass1 is not None else np.nan
    lal_h5_file.attrs["mass2"] = round(mass2, 3) if mass2 is not None else np.nan
    eta = coalescence.symmetric_mass_ratio
    lal_h5_file.attrs["eta"] = eta if eta is not None else np.nan
    if spin1 is None:
        lal_h5_file.attrs["spin1x"] = np.nan
        lal_h5_file.attrs["spin1y"] = np.nan
        lal_h5_file.attrs["spin1z"] = np.nan
    else:
        lal_h5_file.attrs["spin1x"] = spin1[0] if abs(spin1[0]) > 1e-3 else 0.0
        lal_h5_file.attrs["spin1y"] = spin1[1] if abs(spin1[1]) > 1e-3 else 0.0
        lal_h5_file.attrs["spin1z"] = spin1[2] if abs(spin1[2]) > 1e-3 else 0.0
    if spin2 is None:
        lal_h5_file.attrs["spin2x"] = np.nan
        lal_h5_file.attrs["spin2y"] = np.nan
        lal_h5_file.attrs["spin2z"] = np.nan
    else:
        lal_h5_file.attrs["spin2x"] = spin2[0] if abs(spin2[0]) > 1e-3 else 0.0
        lal_h5_file.attrs["spin2y"] = spin2[1] if abs(spin2[1]) > 1e-3 else 0.0
        lal_h5_file.attrs["spin2z"] = spin2[2] if abs(spin2[2]) > 1e-3 else 0.0

    # omega_strain_22 = coalescence.compute_psi4_omega_22_at_time(desired_time=initial_time_strain)
    f_22 = omega_22_nr / (2 * np.pi)
    lal_h5_file.attrs["f_lower_at_1MSUN"] = _nr_frequency_to_physical(f_22, 1)
    lal_h5_file.attrs["Omega"] = coalescence.orbital_frequency_at_time(desired_time=initial_time_horizon)
    LNhat = coalescence.orbital_angular_momentum_unit_vector_at_time(desired_time=initial_time_horizon)
    # flip nhat so it points from the secondary to the primary
    nhat = -1 * coalescence.separation_unit_vector_at_time(desired_time=initial_time_horizon)
    lal_h5_file.attrs["LNhatx"] = LNhat[0] if abs(LNhat[0]) > 1e-5 else 0.0
    lal_h5_file.attrs["LNhaty"] = LNhat[1] if abs(LNhat[1]) > 1e-5 else 0.0
    lal_h5_file.attrs["LNhatz"] = LNhat[2] if abs(LNhat[2]) > 1e-5 else 0.0
    lal_h5_file.attrs["nhatx"] = nhat[0] if abs(nhat[0]) > 1e-5 else 0.0
    lal_h5_file.attrs["nhaty"] = nhat[1] if abs(nhat[1]) > 1e-5 else 0.0
    lal_h5_file.attrs["nhatz"] = nhat[2] if abs(nhat[2]) > 1e-5 else 0.0

    aux_group = lal_h5_file.create_group("auxiliary-info")
    separation_after_junk = coalescence.separation_at_time(desired_time=initial_time_horizon)
    if separation_after_junk is not None:
        aux_group.attrs["separation"] = separation_after_junk

    from mayawaves.radiation import Frame
    if coalescence.radiation_frame == Frame.COM_CORRECTED:
        aux_group.attrs["frame"] = "Center of mass drift corrected"

    eccentricity, mean_anomaly = coalescence.eccentricity_and_mean_anomaly_at_time(initial_time_horizon,
                                                                                   desired_time=initial_time_horizon)
    if eccentricity is None or np.isnan(eccentricity):
        raise_io_error = True
        io_error_cause = "eccentricity"
        lal_h5_file.attrs["eccentricity"] = np.nan
        lal_h5_file.attrs["mean_anomaly"] = np.nan
    else:
        lal_h5_file.attrs["eccentricity"] = eccentricity
        lal_h5_file.attrs["mean_anomaly"] = mean_anomaly

    if raise_io_error:
        if io_error_cause == "mass":
            raise IOError("Mass wasn't defined")
        elif io_error_cause == "spin":
            raise IOError("Spin wasn't defined")
        elif io_error_cause == "spin_precession":
            raise IOError(
                "Spin data was not available at the desired time. It has been provided at the last available time.")
        elif io_error_cause == "eccentricity":
            raise IOError(
                "Unable to compute the eccentricity.")
        else:
            raise IOError(
                "Unknown IOError.")


def _get_max_time_all_strain_modes(strain_modes: dict) -> float:
    """The maximum amplitude time when all modes are added together using the squares of their amplitudes.

    Args:
        strain_modes (dict): dictionary containing the data for all the modes

    Returns:
        float: the time at which the combination of the modes peaks

    """
    squared_sum = 0
    time = None
    for mode in strain_modes.keys():
        time = strain_modes[mode][0]
        amp = strain_modes[mode][1]
        squared_sum += amp * amp

    max_iter = np.argmax(squared_sum)
    max_time = time[max_iter]
    return max_time


def _get_omega_at_time(time: np.ndarray, phase: np.ndarray, initial_time: float) -> float:
    """Frequency at a given time.

    Returns the frequency, the derivative of the provided phase with respect to time.

    Args:
        time (numpy.ndarray): time array
        phase (numpy.ndarray): phase array
        initial_time (float): desired time

    Returns:
        float: frequency at the initial time

    """
    omega = -1 * np.gradient(phase) / np.gradient(time)
    time_index = np.argmax(time > initial_time)
    omega_at_time = omega[time_index]
    return omega_at_time


def _export_ascii_file(filename: str, data: np.ndarray, coalescence_directory: str, header: str = None):
    """Export specified data array to ascii.

    Export specified data array to an ascii file of the given name with the provided header.

    Args:
        filename (str): the filename to be exported to
        data (np.ndarray): array containing the data to be exported
        coalescence_directory (str): the path to store the exported data
        header (str): the header to put at the top of the exported ascii file

    """
    if header is not None:
        np.savetxt(os.path.join(coalescence_directory, filename), data, header=header)
    else:
        np.savetxt(os.path.join(coalescence_directory, filename), data)


def _put_data_in_lal_compatible_format(coalescence: Coalescence, lal_h5_file_name: str, name: str,
                                       alternative_names: list, extraction_radius: float,
                                       NR_group: str, NR_code: str, bibtex_keys: str, contact_email: str,
                                       license_type = 'LVC-internal', nr_techniques: str = None,
                                       comparable_simulation: str = None, files_in_error_series: str = '',
                                       production_run: bool = True, center_of_mass_correction: bool = False):
    """Exports the Coalescence object to a format compatible with LALSuite.

    Exports the Coalescence object into the format required by https://arxiv.org/abs/1703.01076 in order to be
    readable by LALSuite and PyCBC.

    Args:
        coalescence (Coalescence): the Coalescence object being exported
        lal_h5_file_name (str): the path to the h5 file being exported to
        name (str): the name of the simulation
        alternative_names (list): alternative names for the simulation
        extraction_radius (float): the extraction radius being exported. If 0, will be extrapolated to infinite radius.
        NR_group (str): NR group that performed this simulation
        NR_code (str): NR code that performed this simulation
        bibtex_keys (str): bibtex keys to use when citing this simulation
        contact_email (str): email to use if questions arise regarding this simulation
        license_type (:obj:`str`, optional): whether it is public or LVC-internal
        nr_techniques (:obj:`str`, optional): what techniques were used in this simulation
        comparable_simulation (:obj:`str`, optional): other similar simulations
        files_in_error_series (:obj:`str`, optional): other simulations in the same error series
        production_run (:obj:`bool`, optional): whether this is a production run. Default True.
        center_of_mass_correction (:obj:'bool', optional): whether to correct for center of mass drift. Default False.

    """
    from mayawaves.radiation import Frame
    if center_of_mass_correction:
        coalescence.set_radiation_frame(center_of_mass_corrected=True)
    else:
        if not coalescence.radiation_frame == Frame.RAW:
            coalescence.set_radiation_frame()

    initial_time_horizon = 75
    if extraction_radius != 0:
        initial_time_strain = initial_time_horizon + extraction_radius
    else:
        initial_time_strain = initial_time_horizon + coalescence.radiationbundle.radius_for_extrapolation

    lal_h5_file = h5py.File(lal_h5_file_name, 'w')
    # strain
    included_modes = coalescence.included_modes
    time = None
    strain_modes = {}
    omega_22_nr = None
    raise_mode_error = False
    for mode in included_modes:
        l = int(mode[0])
        m = int(mode[1])
        if l < 2:
            continue
        time_raw, amp_raw, phase_raw = coalescence.strain_amp_phase_for_mode(l, m, extraction_radius)
        if time_raw is None or amp_raw is None or phase_raw is None:
            raise_mode_error = True
            continue
        if l == 2 and m == 2:
            omega_22_nr = _get_omega_at_time(time_raw, phase_raw, initial_time_strain)
        initial_index = np.argmax(time_raw > initial_time_strain)
        time = time_raw[initial_index:]
        amp = amp_raw[initial_index:]
        phase = phase_raw[initial_index:]
        strain_modes[mode] = (time, amp, phase)

    max_time = _get_max_time_all_strain_modes(strain_modes)

    for mode in strain_modes.keys():
        l = int(mode[0])
        m = int(mode[1])
        print("Storing (l, m)=(%d, %d)" % (l, m))
        time = strain_modes[mode][0] - max_time
        amp = strain_modes[mode][1]
        phase = strain_modes[mode][2]
        # Temporarily set m=0 modes to 0
        if m == 0:
            amp = np.zeros(amp.shape)
            phase = np.zeros(phase.shape)
        spline = romspline.ReducedOrderSpline(time, amp, verbose=False)
        mode_amp_group = lal_h5_file.create_group("amp_l%d_m%d" % (l, m))
        spline.write(mode_amp_group)
        spline = romspline.ReducedOrderSpline(time, phase, verbose=False)
        mode_phase_group = lal_h5_file.create_group("phase_l%d_m%d" % (l, m))
        spline.write(mode_phase_group)

    lal_h5_file.create_dataset("NRtimes", data=time)

    # metadata
    try:
        lvc_format = determine_lvc_format(coalescence, initial_horizon_time=initial_time_horizon)

        _store_lal_metadata(coalescence, lal_h5_file, name, alternative_names, initial_time_horizon, omega_22_nr,
                            lvc_format, NR_group, NR_code, bibtex_keys, contact_email, license_type, nr_techniques,
                            comparable_simulation, files_in_error_series, production_run)

        time_shift = max_time - (
            coalescence.radiationbundle.radius_for_extrapolation if extraction_radius == 0 else extraction_radius)
        _store_compact_object_timeseries_data(coalescence, lal_h5_file, lvc_format, time_shift, initial_time_horizon)

        lal_h5_file.close()
        if coalescence.radiation_frame != Frame.RAW:
            coalescence.set_radiation_frame()

    except Exception as e:
        lal_h5_file.close()
        if coalescence.radiation_frame != Frame.RAW:
            coalescence.set_radiation_frame()
        raise e

    if raise_mode_error:
        if coalescence.radiation_frame != Frame.RAW:
            coalescence.set_radiation_frame()
        raise IOError(f"Data is missing for one of the included modes")


def low_pass_filter(time: np.ndarray, data: np.ndarray, low_pass_freq_cutoff: float) -> np.ndarray:
    """Filter out high frequency noise using a butter filter.

    Args:
        time (numpy.ndarray): time stamps associated with data
        data (numpy.ndarray): data to filter
        low_pass_freq_cutoff (float): frequency cutoff to use with butter filter

    Returns:
        numpy.ndarray: data with high frequencies removed

    """
    step_size = time[1] - time[0]
    fs = 1 / step_size

    Wn = 2 * (low_pass_freq_cutoff / (2 * np.pi)) / fs

    b, a = butter(4, Wn, analog=False)
    filtered = filtfilt(b, a, data, axis=0)

    return filtered


def determine_lvc_format(coalescence: Coalescence, initial_horizon_time: float) -> int:
    """Determine the LVC format number based upon the data available from the simulation and the definitions in
    https://arxiv.org/abs/1703.01076

    Args:
        coalescence (Coalescence): the Coalescence object being exported
        initial_horizon_time (float): the time for initial data at the horizon

    Returns:
        int: the LVC format (1, 2, 3)

    """
    merge_time = coalescence.merge_time
    if initial_horizon_time > merge_time:
        return 1

    has_primary_object_evolution_data = False
    has_secondary_object_evolution_data = False
    has_remnant_evolution_data = False

    remnant_object = coalescence.final_compact_object
    if remnant_object is not None:
        position_time, position = remnant_object.position_vector
        spin_time, spin = remnant_object.dimensionless_spin_vector
        mass_time, mass = remnant_object.horizon_mass
        if position is not None and spin is not None and mass is not None and len(position_time) > 0 and len(
                spin_time) > 0 and len(mass_time) > 0:
            if position_time[-1] - position_time[0] > 50 and spin_time[-1] - spin_time[0] > 50 \
                    and mass_time[-1] - mass_time[0] > 50:
                has_remnant_evolution_data = True

    primary_object = coalescence.primary_compact_object
    if primary_object is not None:
        position_time, position = primary_object.position_vector
        spin_time, spin = primary_object.dimensionless_spin_vector
        mass_time, mass = primary_object.horizon_mass

        if position is not None and spin is not None and mass is not None and len(position_time) > 0 and len(
                spin_time) > 0 and len(mass_time) > 0:
            if position_time[-1] > merge_time - 10 and spin_time[-1] > merge_time - 10 \
                    and mass_time[-1] > merge_time - 10:
                has_primary_object_evolution_data = True

    secondary_object = coalescence.secondary_compact_object
    if secondary_object is not None:
        position_time, position = secondary_object.position_vector
        spin_time, spin = secondary_object.dimensionless_spin_vector
        mass_time, mass = secondary_object.horizon_mass

        merge_time = coalescence.merge_time
        if position is not None and spin is not None and mass is not None and len(position_time) > 0 and len(
                spin_time) > 0 and len(mass_time) > 0:
            if position_time[-1] > merge_time - 10 and spin_time[-1] > merge_time - 10 \
                    and mass_time[-1] > merge_time - 10:
                has_secondary_object_evolution_data = True

    # determine format
    if has_primary_object_evolution_data and has_secondary_object_evolution_data:
        if has_remnant_evolution_data:
            lvc_format = 3
        else:
            lvc_format = 2
    else:
        lvc_format = 1

    return lvc_format


def export_to_lvcnr_catalog(coalescence: Coalescence, output_directory: str,
                            NR_group: str, NR_code: str, bibtex_keys: str, contact_email: str,
                            name: str = None, license_type='LVC-internal', nr_techniques: str = None,
                            comparable_simulation: str = None, files_in_error_series: str = '',
                            production_run: bool = True, center_of_mass_correction: bool = False):
    """Exports the Coalescence object to the format required by LIGO to be included in the LVC-NR catalog.

    Exports the Coalescence object into the format required by https://arxiv.org/abs/1703.01076 in order to be
    included in the LVC-NR catalog. Must have sufficient spin data and be able to compute eeccentricity. Will be
    extrapolated to infinite radius.

    Args:
        coalescence (Coalescence): the Coalescence object being exported
        output_directory (str): location to store the exported catalog file
        NR_group (str): NR group that performed this simulation
        NR_code (str): NR code that performed this simulation
        bibtex_keys (str): bibtex keys to use when citing this simulation
        contact_email (str): email to use if questions arise regarding this simulation
        name (:obj:`str`, optional): the tag to save the simulation as (e.g. MAYA0908)
        license_type (:obj:`str`, optional): whether it is public or LVC-internal
        nr_techniques (:obj:`str`, optional): what techniques were used in this simulation
        comparable_simulation (:obj:`str`, optional): other similar simulations
        files_in_error_series (:obj:`str`, optional): other simulations in the same error series
        production_run (:obj:`bool`, optional): whether this is a production run. Default True.
        center_of_mass_correction (:obj:`bool`, optional): whether to correct for center of mass drift. Default False.

    """
    catalog_id = coalescence.catalog_id
    simulation_name = coalescence.name
    if name is None:
        if catalog_id is not None:
            name = catalog_id
            alternative_names = [simulation_name]
        else:
            name = simulation_name
            alternative_names = [""]
    else:
        alternative_names = [simulation_name]
        if catalog_id is not None:
            alternative_names.append(catalog_id)
    h5_file_path = os.path.join(output_directory, name + ".h5")
    try:
        _put_data_in_lal_compatible_format(coalescence=coalescence, lal_h5_file_name=h5_file_path, name=name,
                                           alternative_names=alternative_names, extraction_radius=0,
                                           NR_group=NR_group, NR_code=NR_code, bibtex_keys=bibtex_keys,
                                           contact_email=contact_email, license_type=license_type,
                                           nr_techniques=nr_techniques, comparable_simulation=comparable_simulation,
                                           files_in_error_series=files_in_error_series, production_run=production_run,
                                           center_of_mass_correction=center_of_mass_correction)

    except IOError as e:
        print(e)
        warnings.warn("Unable to output to lvc catalog format due to missing spin data, non computed eccentricity, "
                      "or missing mode data.")
        os.remove(h5_file_path)


def export_to_lal_compatible_format(coalescence: Coalescence, output_directory,
                                    NR_group: str, NR_code: str, bibtex_keys: str, contact_email: str,
                                    extraction_radius: float = 0, name=None, license_type='LVC-internal',
                                    nr_techniques: str = None,
                                    comparable_simulation: str = None, files_in_error_series: str = '',
                                    production_run: bool = True, center_of_mass_correction: bool = False):
    """Exports the Coalescence object to a format compatible with LALSuite.

    Exports the Coalescence object into the format required by https://arxiv.org/abs/1703.01076 in order to be
    readable by LALSuite and PyCBC. Less stringent requirements than the LVC-NR catalog.

    Args:
        coalescence (Coalescence): the Coalescence object being exported
        output_directory (str): location to store the exported catalog file
        NR_group (str): NR group that performed this simulation
        NR_code (str): NR code that performed this simulation
        bibtex_keys (str): bibtex keys to use when citing this simulation
        contact_email (str): email to use if questions arise regarding this simulation
        extraction_radius (:obj:`float`, optional): radius at which to extract gravitational wave data
        name (:obj:`str`, optional): the tag to save the simulation as (e.g. MAYA0908)
        license_type (:obj:`str`, optional): whether it is public or LVC-internal
        nr_techniques (:obj:`str`, optional): what techniques were used in this simulation
        comparable_simulation (:obj:`str`, optional): other similar simulations
        files_in_error_series (:obj:`str`, optional): other simulations in the same error series
        production_run (:obj:`bool`, optional): whether this is a production run. Default True.
        center_of_mass_correction (:obj:`bool`, optional): whether to correct for center of mass drift. Default False.

    """
    simulation_name = coalescence.name
    if name is None:
        name = simulation_name
        alternative_names = [""]
    else:
        alternative_names = [simulation_name]
    h5_file_path = os.path.join(output_directory, name + ".h5")
    try:
        _put_data_in_lal_compatible_format(coalescence=coalescence, lal_h5_file_name=h5_file_path, name=name,
                                           alternative_names=alternative_names, extraction_radius=extraction_radius,
                                           NR_group=NR_group, NR_code=NR_code, bibtex_keys=bibtex_keys, contact_email=contact_email,
                                           license_type=license_type, nr_techniques=nr_techniques,
                                           comparable_simulation=comparable_simulation,
                                           files_in_error_series=files_in_error_series, production_run=production_run,
                                           center_of_mass_correction=center_of_mass_correction)
    except IOError as e:
        print(e)
        warnings.warn(
            "Spin data was not available at the cutoff time. It is provided at an earlier time. Or eccentricity "
            "couldn't be computed.")


def export_to_ascii(coalescence: Coalescence, output_directory: str, center_of_mass_correction: bool = False):
    """Export a Coalescence object to ascii format in the specified output directory.

    Export all information contained in the Coalescence object to individual ascii files in the specified output
    directory.

    Args:
        coalescence (Coalescence): the Coalescence object to be exported
        output_directory (str): the location to store the exported ascii data
        center_of_mass_correction (:obj:`bool`, optional): whether to correct for center of mass drift. Default False.

    """
    coalescence_name = coalescence.name
    coalescence_output_directory = os.path.join(output_directory, coalescence_name)
    os.mkdir(coalescence_output_directory)

    # export parameter file
    parfile_dict = coalescence.parameter_files
    for parfile_extension, content in parfile_dict.items():
        with open(os.path.join(coalescence_output_directory, f"{coalescence_name}.{parfile_extension}"), 'w') as f:
            f.write(content)

    # export compact object data
    object_numbers = coalescence.object_numbers

    # metadata
    metadata_dict = coalescence.compact_object_metadata_dict()

    compact_object_dict = {}
    for object_number in object_numbers:
        compact_object_dict[object_number] = coalescence.compact_object_data_for_object(object_number)

    for co_filetype in _CompactObjectFilenames:
        co_filetype.class_reference.export_compact_object_data_to_ascii(compact_object_dict,
                                                                        coalescence_output_directory, metadata_dict,
                                                                        parfile_dict)

    # export runstats
    runstats_data = coalescence.runstats_data
    runstats_header = """iteration, coord_time, wall_time, speed (hours^-1), period (minutes), cputime (cpu hours)"""
    if runstats_data is not None:
        _export_ascii_file("runstats.asc", runstats_data, coalescence_output_directory, header=runstats_header)

    # export psi4
    included_radii = coalescence.included_extraction_radii
    included_modes = coalescence.included_modes
    psi4_data_dict = {}
    for radius in included_radii:
        psi4_data_dict[radius] = {}
        for mode in included_modes:
            l_val = mode[0]
            m_val = mode[1]

            time, psi4_real, psi4_imag = coalescence.psi4_real_imag_for_mode(l_val, m_val, radius)
            psi4_data = np.column_stack([time, psi4_real, psi4_imag])
            psi4_data_dict[radius][(l_val, m_val)] = psi4_data

    filename_prefix = coalescence.psi4_source
    for filetype in _RadiativeFilenames:
        if filetype.string_name == filename_prefix:
            filetype.class_reference.export_radiative_data_to_ascii(psi4_data_dict, coalescence_output_directory,
                                                                    com_correction=center_of_mass_correction)


def summarize_coalescence(coalescence: Coalescence, output_directory: str = None):
    """Summarize data collected from a Coalescence simulation.

    Create a summary.txt document and important plots in the specified output directory.

    Args:
        coalescence (Coalescence): the Coalescence object to be summarized
        output_directory (:obj:`str`, optional): the location to store the summary file and plots. If a path is not
            provided, the plots are displayed rather than saved.

    """

    # Check that output_directory exists and create summary folder.
    coalescence_name = coalescence.name
    if output_directory is None:
        pass
    elif os.path.exists(output_directory):
        simulation_output_directory = os.path.join(output_directory, "summary_" + coalescence_name)
        # Delete previous summary for coalescence, if any.
        if os.path.exists(simulation_output_directory):
            shutil.rmtree(simulation_output_directory)
        os.makedirs(simulation_output_directory)
    else:
        warnings.warn("Output directory provided does not exist.")
        output_directory = None

    # Define primary and secondary objects, their separation data, and extraction radius.
    primary_compact_object = coalescence.primary_compact_object
    secondary_compact_object = coalescence.secondary_compact_object
    time, separation_vector = coalescence.separation_vector
    separations = np.linalg.norm(separation_vector, axis=1)
    current_time = time[-1]
    extraction_radii = coalescence.included_extraction_radii

    if 75 in extraction_radii:
        extraction_radius = 75
    else:
        extraction_radius = max([r for r in extraction_radii if r < 75])

    # Initial Values:
    initial_primary_horizon_mass = primary_compact_object.initial_horizon_mass
    initial_secondary_horizon_mass = secondary_compact_object.initial_horizon_mass
    initial_mass_ratio = coalescence.mass_ratio
    initial_primary_dimensionless_spin = primary_compact_object.initial_dimensionless_spin
    initial_secondary_dimensionless_spin = secondary_compact_object.initial_dimensionless_spin
    initial_separation = coalescence.initial_separation

    # After Junk Radiation:
    if current_time > 75:
        primary_horizon_mass = primary_compact_object.horizon_mass_at_time(75)
        secondary_horizon_mass = secondary_compact_object.horizon_mass_at_time(75)
        primary_dimensionless_spin = primary_compact_object.dimensionless_spin_at_time(75)
        secondary_dimensionless_spin = secondary_compact_object.dimensionless_spin_at_time(75)
        separation = coalescence.separation_at_time(75)
        separation_unit_vector = coalescence.separation_unit_vector_at_time(75)
        orbital_frequency = coalescence.orbital_frequency_at_time(75)
        orbital_angular_momentum_unit_vector = coalescence.orbital_angular_momentum_unit_vector_at_time(75)
        eccentricity = coalescence.eccentricity_and_mean_anomaly_at_time(75, 75)[0]

    # Remnant/Current Values:
    if coalescence.final_compact_object is None:
        current_separation = separations[-1]
        current_separation_unit_vector = separation_vector[-1]
        current_orbital_frequency = coalescence.orbital_frequency_at_time(current_time - 0.01)
        current_orbital_angular_momentum_unit_vector = coalescence.orbital_angular_momentum_unit_vector_at_time(
            current_time - 0.01)
    else:
        final_horizon_mass = coalescence.final_compact_object.final_horizon_mass
        final_dimensionless_spin = coalescence.final_compact_object.final_dimensionless_spin

    # Format information to be printed to summary.txt:
    # beginning includes information regarding initial data.
    mass_ratio_string = 'NaN' if initial_mass_ratio is None else f'{initial_mass_ratio:.4f}'
    primary_horizon_mass_string = 'NaN' if initial_primary_horizon_mass is None else f'{initial_primary_horizon_mass:.4f}'
    secondary_horizon_mass_string = 'NaN' if initial_secondary_horizon_mass is None else f'{initial_secondary_horizon_mass:.4f}'
    primary_spin_string = 'NaN' if initial_primary_dimensionless_spin is None else f'[{initial_primary_dimensionless_spin[0]:.4f}, {initial_primary_dimensionless_spin[1]:.4f}, {initial_primary_dimensionless_spin[2]:.4f}]'
    secondary_spin_string = 'NaN' if initial_secondary_dimensionless_spin is None else f'[{initial_secondary_dimensionless_spin[0]:.4f}, {initial_secondary_dimensionless_spin[1]:.4f}, {initial_secondary_dimensionless_spin[2]:.4f}]'
    separation_string = 'NaN' if initial_separation is None else f'{initial_separation:.4f}'
    beginning = f"Summary for {coalescence_name}\n" \
                f"\nInitial Values (t=0 M):\n" \
                f"{'-' * 70}\n" \
                f"mass ratio:\t\t\t\t{mass_ratio_string}\n" \
                f"primary horizon mass:\t\t\t{primary_horizon_mass_string}\n" \
                f"secondary horizon mass:\t\t\t{secondary_horizon_mass_string}\n" \
                f"primary dimensionless spin:\t\t{primary_spin_string}\n" \
                f"secondary dimensionless spin:\t\t{secondary_spin_string}\n" \
                f"separation:\t\t\t\t{separation_string} M\n"

    # middle includes information regarding data at t=75 M.
    if current_time < 75:
        if coalescence.final_compact_object is None:
            middle = f"\nAfter Junk Radiation (t=75 M):\n" \
                     f"{'-' * 70}\n" \
                     f"Simulation has not yet reached t=75 M.\n"
        else:
            middle = f"\nAfter Junk Radtation (t=75 M):\n" \
                     f"{'-' * 70}\n" \
                     f"Simulation completed before reaching t=75 M.\n"
    else:
        primary_horizon_mass_string = 'NaN' if primary_horizon_mass is None else f'{primary_horizon_mass:.4f}'
        secondary_horizon_mass_string = 'NaN' if secondary_horizon_mass is None else f'{secondary_horizon_mass:.4f}'
        primary_spin_string = 'NaN' if primary_dimensionless_spin is None else f'[{primary_dimensionless_spin[0]:.4f}, {primary_dimensionless_spin[1]:.4f}, {primary_dimensionless_spin[2]:.4f}]'
        secondary_spin_string = 'NaN' if secondary_dimensionless_spin is None else f'[{secondary_dimensionless_spin[0]:.4f}, {secondary_dimensionless_spin[1]:.4f}, {secondary_dimensionless_spin[2]:.4f}]'
        separation_string = 'NaN' if separation is None else f'{separation:.4f}'
        separation_unit_vector_string = 'NaN' if separation_unit_vector is None else f'[{separation_unit_vector[0]:.4f}, {separation_unit_vector[1]:.4f}, {separation_unit_vector[2]:.4f}]'
        orbital_frequency_string = 'NaN' if orbital_frequency is None else f'{orbital_frequency:.4f}'
        orbital_angular_momentum_unit_vector_string = 'NaN' if orbital_angular_momentum_unit_vector is None else f'[{orbital_angular_momentum_unit_vector[0]:.4f}, {orbital_angular_momentum_unit_vector[1]:.4f}, {orbital_angular_momentum_unit_vector[2]:.4f}]'
        eccentricity_string = 'NaN' if eccentricity is None else f'{eccentricity:.4f}'
        middle = f"\nAfter Junk Radiation (t=75 M):\n" \
                 f"{'-' * 70}\n" \
                 f"primary horizon mass:\t\t\t{primary_horizon_mass_string}\n" \
                 f"secondary horizon mass:\t\t\t{secondary_horizon_mass_string}\n" \
                 f"primary dimensionless spin:\t\t{primary_spin_string}\n" \
                 f"secondary dimensionless spin:\t\t{secondary_spin_string}\n" \
                 f"separation:\t\t\t\t{separation_string} M\n" \
                 f"separation unit vector:\t\t\t{separation_unit_vector_string}\n" \
                 f"orbital frequency:\t\t\t{orbital_frequency_string}\n" \
                 f"orbital angular momentum unit vector:\t{orbital_angular_momentum_unit_vector_string}\n" \
                 f"eccentricity:\t\t\t\t{eccentricity_string}\n"
        if primary_horizon_mass is None or secondary_horizon_mass is None or \
                primary_dimensionless_spin is None or secondary_dimensionless_spin is None:
            middle += r"**A 'NaN' value represents data that was not being tracked at t=75 M.**" + "\n"

    # ending includes information regarding most recent data.
    if coalescence.final_compact_object is None:
        time_string = 'NaN' if current_time is None else f'{current_time:.4f}'
        separation_string = 'NaN' if current_separation is None else f'{current_separation:.4f}'
        separation_unit_vector_string = 'NaN' if current_separation_unit_vector is None else f'[{current_separation_unit_vector[0]:.4f}, {current_separation_unit_vector[1]:.4f}, {current_separation_unit_vector[2]:.4f}]'
        orbital_frequency_string = 'NaN' if current_orbital_frequency is None else f'{current_orbital_frequency:.4f}'
        orbital_angular_momentum_unit_vector_string = 'NaN' if current_orbital_angular_momentum_unit_vector is None else f'[{current_orbital_angular_momentum_unit_vector[0]:.4f}, {current_orbital_angular_momentum_unit_vector[1]:.4f}, {current_orbital_angular_momentum_unit_vector[2]:.4f}]'
        ending = f"\nCurrent Values (Last available data):\n" \
                 f"{'-' * 70}\n" \
                 f"current time:\t\t\t\t{time_string}\n" \
                 f"separation:\t\t\t\t{separation_string}\n" \
                 f"separation unit vector:\t\t\t{separation_unit_vector_string}\n" \
                 f"orbital frequency:\t\t\t{orbital_frequency_string}\n" \
                 f"orbital angular momentum unit vector:\t{orbital_angular_momentum_unit_vector_string}"
    else:
        final_horizon_mass_string = 'NaN' if final_horizon_mass is None else f'{final_horizon_mass:.4f}'
        final_spin_string = 'NaN' if final_dimensionless_spin is None else f'[{final_dimensionless_spin[0]:.4f}, {final_dimensionless_spin[1]:.4f}, {final_dimensionless_spin[2]:.4f}]'
        ending = f"\nRemnant Values (Last available data):\n" \
                 f"{'-' * 70}\n" \
                 f"horizon mass:\t\t\t\t{final_horizon_mass_string}\n" \
                 f"dimensionless spin:\t\t\t{final_spin_string}"
        if final_horizon_mass is None or final_dimensionless_spin is None:
            ending += "\n" + r"** A 'NaN' value represents data that has not yet been calculated.**"

    summary = f"{beginning}{middle}{ending}"

    # Either print data or write to summary.txt.
    if output_directory is None:
        print(summary)
    else:
        path_to_summary_file = os.path.join(simulation_output_directory, 'summary.txt')
        summary_file = open(path_to_summary_file, 'w')
        summary_file.write(summary)
        summary_file.close()

    # Plot Separation
    plt.plot(time, separations)
    plt.xlabel('t (M)')
    plt.ylabel('Separation')
    plt.tight_layout()
    if output_directory is None:
        plt.show()
    else:
        path_to_separation_plot = os.path.join(simulation_output_directory, 'separation.png')
        plt.savefig(path_to_separation_plot)
    plt.close()

    # Plot Position
    time, primary_position = primary_compact_object.position_vector
    time, secondary_position = secondary_compact_object.position_vector
    plt.plot(primary_position[:, 0], primary_position[:, 1], label='Primary')
    plt.plot(secondary_position[:, 0], secondary_position[:, 1], label='Secondary')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.legend()
    if output_directory is None:
        plt.show()
    else:
        path_to_position_plot = os.path.join(simulation_output_directory, 'position.png')
        plt.savefig(path_to_position_plot)
    plt.close()

    # Plot psi4_22
    time, real, imag = coalescence.psi4_real_imag_for_mode(l=2, m=2, extraction_radius=extraction_radius)
    plt.plot(time, real, label='Real')
    plt.plot(time, imag, label='Imaginary')
    plt.xlabel('t (M)')
    plt.ylabel(r'$\Psi_{4, 22}$')
    plt.tight_layout()
    plt.legend()
    if output_directory is None:
        plt.show()
    else:
        path_to_psi4_plot = os.path.join(simulation_output_directory, 'psi4_22.png')
        plt.savefig(path_to_psi4_plot)
    plt.close()

    # Plot strain_22
    time, h_plus, h_cross = coalescence.strain_for_mode(l=2, m=2, extraction_radius=extraction_radius)
    plt.plot(time, h_plus, label=r'$h_+$')
    plt.plot(time, h_cross, label=r'$h_{\times}$')
    plt.xlabel('t (M)')
    plt.ylabel(r'$rh_{22}$')
    plt.tight_layout()
    plt.legend()
    if output_directory is None:
        plt.show()
    else:
        path_to_strain_plot = os.path.join(simulation_output_directory, 'strain_22.png')
        plt.savefig(path_to_strain_plot)
    plt.close()

    if output_directory is not None:
        exit_message = f"Summary successfully saved!\n{simulation_output_directory}\n"
        print(exit_message)
