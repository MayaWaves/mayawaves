from unittest import TestCase
import mayawaves.utils.postprocessingutils as pputils
import os
import io
import h5py
import numpy as np
import filecmp
import shutil
import re
from unittest.mock import patch, PropertyMock
from mayawaves.coalescence import Coalescence
from datetime import date
import romspline
import mock


class TestPostprocessingUtils(TestCase):
    simulation_name = None
    parameter_file_name = None
    raw_directory = None
    output_directory = None
    stitched_data_directory = None

    CURR_DIR = os.path.dirname(__file__)

    @classmethod
    def setUpClass(cls) -> None:
        TestPostprocessingUtils.output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                                "resources/test_output")
        if os.path.exists(TestPostprocessingUtils.output_directory):
            shutil.rmtree(TestPostprocessingUtils.output_directory)
        os.mkdir(TestPostprocessingUtils.output_directory)

        TestPostprocessingUtils.simulation_name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        TestPostprocessingUtils.parameter_file_name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        TestPostprocessingUtils.raw_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                             "resources/main_test_simulation/%s") % TestPostprocessingUtils.simulation_name
        TestPostprocessingUtils.stitched_data_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                                       "resources/main_test_simulation/stitched/%s") \
                                                          % TestPostprocessingUtils.simulation_name

        h5_file_original = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                        "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        os.mkdir(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp"))
        shutil.copy2(h5_file_original, os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp"))

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(
                "%s/%s.h5" % (TestPostprocessingUtils.output_directory, TestPostprocessingUtils.simulation_name)):
            os.remove("%s/%s.h5" % (TestPostprocessingUtils.output_directory, TestPostprocessingUtils.simulation_name))
        if os.path.exists(
                "%s/%s.rpar" % (TestPostprocessingUtils.output_directory, TestPostprocessingUtils.simulation_name)):
            os.remove(
                "%s/%s.rpar" % (TestPostprocessingUtils.output_directory, TestPostprocessingUtils.simulation_name))
        shutil.rmtree(TestPostprocessingUtils.output_directory)

        if os.path.exists(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")):
            os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        if os.path.exists(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_raw")):
            shutil.rmtree(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_raw"))
        if os.path.exists(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_output")):
            shutil.rmtree(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_output"))

        if os.path.exists(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp")):
            shutil.rmtree(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp"))

        if os.path.exists(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_stitched")):
            shutil.rmtree(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_stitched"))

        if os.path.exists(os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par")):
            os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par"))

    def test__simulation_name(self):
        from mayawaves.utils.postprocessingutils import _simulation_name
        expected = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        actual = _simulation_name(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulations/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"))
        self.assertEqual(expected, actual)

        expected = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        actual = _simulation_name("a/b/c/d/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")
        self.assertEqual(expected, actual)

        expected = "test_name"
        actual = _simulation_name("a/b/c/test_name")
        self.assertEqual(expected, actual)

    def test__get_parameter_file_name_and_content(self):
        from mayawaves.utils.postprocessingutils import _get_parameter_file_name_and_content
        temp_raw_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_raw")
        os.mkdir(temp_raw_directory)
        temp_output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_output")
        os.mkdir(temp_output_directory)
        simfactory_directory = "%s/SIMFACTORY" % temp_raw_directory
        os.mkdir(simfactory_directory)
        par_directory = "%s/par" % simfactory_directory
        os.mkdir(par_directory)
        output_0 = "%s/output-0000" % temp_raw_directory
        os.mkdir(output_0)
        output_1 = "%s/output-0001" % temp_raw_directory
        os.mkdir(output_1)
        output_2 = "%s/output-0002" % temp_raw_directory
        os.mkdir(output_2)

        # no parfile exists
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        self.assertIsNone(parameter_file_name_base)
        self.assertIsNone(parfile_dict)

        # rpar file in output-0002
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_2)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        expected_rpar_file = os.path.join(output_2, f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_rpar_file) as f:
            expected_rpar_content = f.read()
        expected_par_file = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/%s/output-0000/%s.par") % (
                                TestPostprocessingUtils.simulation_name,
                                TestPostprocessingUtils.simulation_name)
        with open(expected_par_file) as f:
            expected_par_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertTrue('rpar_content' in parfile_dict)
        self.assertEqual(expected_par_content, parfile_dict['par_content'])
        self.assertEqual(expected_rpar_content, parfile_dict['rpar_content'])
        os.remove(expected_rpar_file)

        # par file in output-0001
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_1)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        expected_parameter_file = os.path.join(output_1, f'{TestPostprocessingUtils.simulation_name}.par')
        with open(expected_parameter_file) as f:
            expected_par_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertFalse('rpar_content' in parfile_dict)
        self.assertEqual(expected_par_content, parfile_dict['par_content'])

        # rpar file in output-0001
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_1)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        expected_rpar_file = os.path.join(output_1, f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_rpar_file) as f:
            expected_rpar_content = f.read()
        expected_par_file = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/%s/output-0000/%s.par") % (
                                TestPostprocessingUtils.simulation_name,
                                TestPostprocessingUtils.simulation_name)
        with open(expected_par_file) as f:
            expected_par_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertTrue('rpar_content' in parfile_dict)
        self.assertEqual(expected_par_content, parfile_dict['par_content'])
        self.assertEqual(expected_rpar_content, parfile_dict['rpar_content'])
        os.remove(expected_rpar_file)
        os.remove(os.path.join(output_1, f'{TestPostprocessingUtils.simulation_name}.par'))

        # par file in output-0000
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_0)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        expected_parameter_file = os.path.join(output_0, f'{TestPostprocessingUtils.simulation_name}.par')
        with open(expected_parameter_file) as f:
            expected_par_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertFalse('rpar_content' in parfile_dict)
        self.assertEqual(expected_par_content, parfile_dict['par_content'])

        # rpar file in output-0000
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_0)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        expected_rpar_file = os.path.join(output_0, f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_rpar_file) as f:
            expected_rpar_content = f.read()
        expected_par_file = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name)
        with open(expected_par_file) as f:
            expected_par_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertTrue('rpar_content' in parfile_dict)
        self.assertEqual(expected_par_content, parfile_dict['par_content'])
        self.assertEqual(expected_rpar_content, parfile_dict['rpar_content'])

        # par file in simfactory
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), par_directory)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        expected_parameter_file = os.path.join(par_directory, f'{TestPostprocessingUtils.simulation_name}.par')
        with open(expected_parameter_file) as f:
            expected_par_content = f.read()
        expected_rpar_file = os.path.join(output_0, f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_rpar_file) as f:
            expected_rpar_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertTrue('rpar_content' in parfile_dict)
        self.assertEqual(expected_par_content, parfile_dict['par_content'])
        self.assertEqual(expected_rpar_content, parfile_dict['rpar_content'])

        # rpar file in simfactory
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), par_directory)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        expected_rpar_file = os.path.join(par_directory, f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_rpar_file) as f:
            expected_rpar_content = f.read()
        expected_par_file = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/%s/output-0000/%s.par") % (
                                TestPostprocessingUtils.simulation_name,
                                TestPostprocessingUtils.simulation_name)
        with open(expected_par_file) as f:
            expected_par_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertTrue('rpar_content' in parfile_dict)
        self.assertEqual(expected_par_content, parfile_dict['par_content'])
        self.assertEqual(expected_rpar_content, parfile_dict['rpar_content'])

        # prestitched data
        temp_stitched_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_stitched")
        os.mkdir(temp_stitched_directory)

        # if rpar in prestitched directory
        shutil.copy2(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                  "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.rpar"),
                     temp_stitched_directory)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_stitched_directory)
        expected_rpar_file = os.path.join(temp_stitched_directory,
                                               f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_rpar_file) as f:
            expected_rpar_content = f.read()
        os.system(os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.rpar"))
        expected_par_file = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                  "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par")
        with open(expected_par_file) as f:
            expected_par_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertTrue('rpar_content' in parfile_dict)
        print(len(expected_par_content))
        print(len(parfile_dict['par_content']))
        print(expected_par_content[:10])
        print(parfile_dict['par_content'][:10])
        for i in range(len(expected_par_content)):
            if expected_par_content[i] != parfile_dict['par_content'][i]:
                print(f'{i}: {expected_par_content[i]}, {parfile_dict["par_content"][i]}')
        self.assertEqual(expected_par_content, parfile_dict['par_content'])
        self.assertEqual(expected_rpar_content, parfile_dict['rpar_content'])
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par"))

        # if par in prestitched directory
        os.system(os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.rpar"))
        shutil.copy2(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                  "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par"),
                     temp_stitched_directory)
        parameter_file_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_stitched_directory)
        expected_rpar_file = os.path.join(temp_stitched_directory,
                                          f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_rpar_file) as f:
            expected_rpar_content = f.read()
        expected_par_file = os.path.join(temp_stitched_directory,
                                         f'{TestPostprocessingUtils.simulation_name}.par')
        with open(expected_par_file) as f:
            expected_par_content = f.read()
        self.assertIsNotNone(parfile_dict)
        self.assertEqual(TestPostprocessingUtils.simulation_name, parameter_file_name_base)
        self.assertTrue('par_content' in parfile_dict)
        self.assertTrue('rpar_content' in parfile_dict)
        self.assertEqual(expected_par_content, parfile_dict['par_content'])
        self.assertEqual(expected_rpar_content, parfile_dict['rpar_content'])
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par"))

        shutil.rmtree(temp_stitched_directory)
        shutil.rmtree(temp_raw_directory)
        shutil.rmtree(temp_output_directory)

    def test__ordered_output_directories(self):
        from mayawaves.utils.postprocessingutils import _ordered_output_directories
        expected_ordered_output_directories = [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000"),
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001")]
        actual_ordered_output_directories, _ = _ordered_output_directories(
            TestPostprocessingUtils.raw_directory)
        self.assertTrue(np.all(expected_ordered_output_directories == actual_ordered_output_directories))

        # Test if pre-stitched
        expected_ordered_output_directories = [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_output_directories, _ = _ordered_output_directories(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")
        )
        self.assertTrue(np.all(expected_ordered_output_directories == actual_ordered_output_directories))

    def test__ordered_data_directories(self):
        from mayawaves.utils.postprocessingutils import _ordered_data_directories

        parameter_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/SIMFACTORY/par/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.rpar")
        with open(parameter_filepath, 'r') as f:
            parfile_content = f.read()
        # parfile and simulation have same name
        expected_ordered_data_directories = [os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"),
                                             os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_data_directories = _ordered_data_directories(TestPostprocessingUtils.raw_directory, parfile_content,
                                                                    TestPostprocessingUtils.parameter_file_name)
        self.assertTrue(np.all(expected_ordered_data_directories == actual_ordered_data_directories))

        # parfile and simulation have different names
        expected_ordered_data_directories = [os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_different_name/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_data_directories = _ordered_data_directories(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                                                 "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_different_name"), parfile_content,
                                                                    "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")
        self.assertTrue(np.all(expected_ordered_data_directories == actual_ordered_data_directories))

        # Test if pre-stitched
        expected_ordered_output_directories = [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_output_directories = _ordered_data_directories(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"), parfile_content,
            "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        )
        self.assertTrue(np.all(expected_ordered_output_directories == actual_ordered_output_directories))

        # test when outdir isn't $parfile
        parfile_content = """#############################################################
# Output
#############################################################

IO::out_dir                          = D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67
IO::out_fileinfo                     = "all"

"""
        expected_ordered_data_directories = [os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"),
                                             os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_data_directories = _ordered_data_directories(TestPostprocessingUtils.raw_directory,
                                                                    parfile_content,
                                                                    TestPostprocessingUtils.parameter_file_name)
        self.assertTrue(np.all(expected_ordered_data_directories == actual_ordered_data_directories))

        # test when outdir isn't $parfile and has quotes
        parfile_content = """#############################################################
        # Output
        #############################################################

        IO::out_dir                          = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        IO::out_fileinfo                     = "all"

        """
        expected_ordered_data_directories = [os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"),
                                             os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_data_directories = _ordered_data_directories(TestPostprocessingUtils.raw_directory,
                                                                    parfile_content,
                                                                    TestPostprocessingUtils.parameter_file_name)
        self.assertTrue(np.all(expected_ordered_data_directories == actual_ordered_data_directories))

        # test when outdir isn't $parfile and has quotes
        parfile_content = """#############################################################
        # Output
        #############################################################

        IO::out_dir                          = 'D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67'
        IO::out_fileinfo                     = "all"

        """
        expected_ordered_data_directories = [os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"),
                                             os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_data_directories = _ordered_data_directories(TestPostprocessingUtils.raw_directory,
                                                                    parfile_content,
                                                                    TestPostprocessingUtils.parameter_file_name)
        self.assertTrue(np.all(expected_ordered_data_directories == actual_ordered_data_directories))

        # test when outdir is @SIMULATION_NAME@
        parfile_content = """#############################################################
        # Output
        #############################################################

        IO::out_dir                          = "@SIMULATION_NAME@"
        IO::out_fileinfo                     = "all"

        """
        expected_ordered_data_directories = [os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"),
                                             os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_data_directories = _ordered_data_directories(TestPostprocessingUtils.raw_directory,
                                                                    parfile_content,
                                                                    'par_name')
        self.assertTrue(np.all(expected_ordered_data_directories == actual_ordered_data_directories))

        # test when outdir is @SIMULATION_NAME@
        parfile_content = """#############################################################
                # Output
                #############################################################

                IO::out_dir                          = '@SIMULATION_NAME@'
                IO::out_fileinfo                     = "all"

                """
        expected_ordered_data_directories = [os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"),
                                             os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_data_directories = _ordered_data_directories(TestPostprocessingUtils.raw_directory,
                                                                    parfile_content,
                                                                    'par_name')
        self.assertTrue(np.all(expected_ordered_data_directories == actual_ordered_data_directories))

        # test when outdir is @SIMULATION_NAME@
        parfile_content = """#############################################################
                # Output
                #############################################################

                IO::out_dir                          = @SIMULATION_NAME@
                IO::out_fileinfo                     = "all"

                """
        expected_ordered_data_directories = [os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"),
                                             os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")]
        actual_ordered_data_directories = _ordered_data_directories(TestPostprocessingUtils.raw_directory,
                                                                    parfile_content,
                                                                    'par_name')
        self.assertTrue(np.all(expected_ordered_data_directories == actual_ordered_data_directories))

    def test__store_parameter_file(self):
        from mayawaves.utils.postprocessingutils import _store_parameter_file
        from mayawaves.utils.postprocessingutils import _get_parameter_file_name_and_content
        temp_raw_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_raw")
        os.mkdir(temp_raw_directory)
        temp_output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_output")
        os.mkdir(temp_output_directory)
        simfactory_directory = "%s/SIMFACTORY" % temp_raw_directory
        os.mkdir(simfactory_directory)
        par_directory = "%s/par" % simfactory_directory
        os.mkdir(par_directory)
        output_0 = "%s/output-0000" % temp_raw_directory
        os.mkdir(output_0)
        output_1 = "%s/output-0001" % temp_raw_directory
        os.mkdir(output_1)

        # no parfile exists
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        self.assertFalse('parfile' in temp_h5_file.keys())

        # clean up
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # par file in output-0000
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_0)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        expected_parameter_file = os.path.join(output_0, f'{TestPostprocessingUtils.simulation_name}.par')
        with open(expected_parameter_file) as f:
            expected_content = f.read()
        self.assertTrue('parfile' in temp_h5_file.keys())
        parfile_group = temp_h5_file['parfile']
        self.assertTrue('par_content' in parfile_group.attrs)
        self.assertFalse('rpar_content' in parfile_group.attrs)
        self.assertEqual(expected_content, parfile_group.attrs['par_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.par')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['par_content'])
        self.assertTrue(filecmp.cmp(expected_parameter_file, temp_parameter_file))

        # clean up
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_parfile.par"))

        # rpar file in output-0000
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_0)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        expected_parameter_file = os.path.join(output_0, f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_parameter_file) as f:
            expected_content = f.read()
        self.assertTrue('parfile' in temp_h5_file.keys())
        parfile_group = temp_h5_file['parfile']
        self.assertTrue('rpar_content' in parfile_group.attrs)
        self.assertEqual(expected_content, parfile_group.attrs['rpar_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.rpar')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['rpar_content'])
        self.assertTrue(filecmp.cmp(expected_parameter_file, temp_parameter_file))
        with open(expected_parameter_file.replace('.rpar', '.par')) as f:
            expected_par_contents = f.read()
        self.assertTrue('par_content' in parfile_group.attrs)
        self.assertEqual(expected_par_contents, parfile_group.attrs['par_content'])

        # clean up
        temp_h5_file.close()
        os.remove(expected_parameter_file)
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_parfile.rpar"))

        # par file in output-0001
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_1)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        expected_parameter_file = os.path.join(output_1, f'{TestPostprocessingUtils.simulation_name}.par')
        with open(expected_parameter_file) as f:
            expected_content = f.read()
        self.assertTrue('parfile' in temp_h5_file.keys())
        parfile_group = temp_h5_file['parfile']
        self.assertTrue('par_content' in parfile_group.attrs)
        self.assertFalse('rpar_content' in parfile_group.attrs)
        self.assertEqual(expected_content, parfile_group.attrs['par_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.par')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['par_content'])
        self.assertTrue(filecmp.cmp(expected_parameter_file, temp_parameter_file))

        # clean up
        temp_h5_file.close()
        os.remove(expected_parameter_file)
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_parfile.par"))

        # rpar file in output-0001
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), output_1)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        expected_parameter_file = os.path.join(output_1, f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_parameter_file) as f:
            expected_content = f.read()
        self.assertTrue('parfile' in temp_h5_file.keys())
        parfile_group = temp_h5_file['parfile']
        self.assertTrue('rpar_content' in parfile_group.attrs)
        self.assertEqual(expected_content, parfile_group.attrs['rpar_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.rpar')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['rpar_content'])
        self.assertTrue(filecmp.cmp(expected_parameter_file, temp_parameter_file))
        expected_par_file = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name)
        with open(expected_par_file) as f:
            expected_par_contents = f.read()
        self.assertTrue('par_content' in parfile_group.attrs)
        self.assertEqual(expected_par_contents, parfile_group.attrs['par_content'])

        # clean up
        temp_h5_file.close()
        os.remove(expected_parameter_file)
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_parfile.rpar"))

        # par file in simfactory
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), par_directory)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        expected_parameter_file = os.path.join(par_directory, f'{TestPostprocessingUtils.simulation_name}.par')
        with open(expected_parameter_file) as f:
            expected_content = f.read()
        self.assertTrue('parfile' in temp_h5_file.keys())
        parfile_group = temp_h5_file['parfile']
        self.assertTrue('par_content' in parfile_group.attrs)
        self.assertFalse('rpar_content' in parfile_group.attrs)
        self.assertEqual(expected_content, parfile_group.attrs['par_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.par')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['par_content'])
        self.assertTrue(filecmp.cmp(expected_parameter_file, temp_parameter_file))

        # clean up
        temp_h5_file.close()
        os.remove(expected_parameter_file)
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_parfile.par"))

        # rpar file in simfactory
        shutil.copy2(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name), par_directory)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_raw_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        expected_parameter_file = os.path.join(par_directory, f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_parameter_file) as f:
            expected_content = f.read()
        self.assertTrue('parfile' in temp_h5_file.keys())
        parfile_group = temp_h5_file['parfile']
        self.assertTrue('rpar_content' in parfile_group.attrs)
        self.assertEqual(expected_content, parfile_group.attrs['rpar_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.rpar')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['rpar_content'])
        self.assertTrue(filecmp.cmp(expected_parameter_file, temp_parameter_file))
        expected_par_file = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/main_test_simulation/%s/output-0000/%s.par") % (
                TestPostprocessingUtils.simulation_name,
                TestPostprocessingUtils.simulation_name)
        with open(expected_par_file) as f:
            expected_par_contents = f.read()
        self.assertTrue('par_content' in parfile_group.attrs)
        self.assertEqual(expected_par_contents, parfile_group.attrs['par_content'])

        # clean up
        temp_h5_file.close()
        os.remove(expected_parameter_file)
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_parfile.rpar"))

        # prestitched data
        temp_stitched_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_stitched")
        os.mkdir(temp_stitched_directory)

        # if par in prestitched directory
        os.system(os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.rpar"))
        shutil.copy2(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                  "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par"),
                     temp_stitched_directory)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_stitched_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        expected_parameter_file = os.path.join(temp_stitched_directory,
                                               f'{TestPostprocessingUtils.simulation_name}.par')
        with open(expected_parameter_file) as f:
            expected_content = f.read()
        self.assertTrue('parfile' in temp_h5_file.keys())
        parfile_group = temp_h5_file['parfile']
        self.assertTrue('par_content' in parfile_group.attrs)
        self.assertFalse('rpar_content' in parfile_group.attrs)
        self.assertEqual(expected_content, parfile_group.attrs['par_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.par')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['par_content'])
        self.assertTrue(filecmp.cmp(expected_parameter_file, temp_parameter_file))

        # clean up
        temp_h5_file.close()
        os.remove(expected_parameter_file)
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_parfile.par"))

        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par"))

        # if rpar in prestitched directory
        shutil.copy2(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                  "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.rpar"),
                     temp_stitched_directory)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_name_base, parfile_dict = _get_parameter_file_name_and_content(temp_stitched_directory)
        _store_parameter_file(parfile_dict, temp_h5_file)
        expected_parameter_file = os.path.join(temp_stitched_directory,
                                               f'{TestPostprocessingUtils.simulation_name}.rpar')
        with open(expected_parameter_file) as f:
            expected_content = f.read()
        self.assertTrue('parfile' in temp_h5_file.keys())
        parfile_group = temp_h5_file['parfile']
        self.assertTrue('rpar_content' in parfile_group.attrs)
        self.assertEqual(expected_content, parfile_group.attrs['rpar_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.rpar')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['rpar_content'])
        self.assertTrue(filecmp.cmp(expected_parameter_file, temp_parameter_file))
        self.assertTrue('par_content' in parfile_group.attrs)

        # clean up
        temp_h5_file.close()
        os.remove(expected_parameter_file)
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp_parfile.rpar"))

        shutil.rmtree(temp_stitched_directory)
        shutil.rmtree(temp_raw_directory)
        shutil.rmtree(temp_output_directory)

        # parfile dict is None
        parfile_dict = None
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        _store_parameter_file(parfile_dict, temp_h5_file)
        self.assertFalse('parfile' in temp_h5_file.keys())
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # neither par_content nor rpar_content in parfile_dict
        parfile_dict = {}
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        _store_parameter_file(parfile_dict, temp_h5_file)
        self.assertFalse('parfile' in temp_h5_file.keys())
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # both par_content and rpar_content in parfile_dict
        parfile_dict = {
            'par_content': 'ABCDE',
            'rpar_content': 'FGHIJ'
        }
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        _store_parameter_file(parfile_dict, temp_h5_file)
        self.assertTrue('parfile' in temp_h5_file.keys())
        self.assertTrue('par_content' in temp_h5_file['parfile'].attrs)
        self.assertTrue('rpar_content' in temp_h5_file['parfile'].attrs)
        self.assertEqual('ABCDE', temp_h5_file['parfile'].attrs['par_content'])
        self.assertEqual('FGHIJ', temp_h5_file['parfile'].attrs['rpar_content'])
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # par_content in parfile_dict
        parfile_dict = {
            'par_content': '12345'
        }
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        _store_parameter_file(parfile_dict, temp_h5_file)
        self.assertTrue('parfile' in temp_h5_file.keys())
        self.assertTrue('par_content' in temp_h5_file['parfile'].attrs)
        self.assertFalse('rpar_content' in temp_h5_file['parfile'].attrs)
        self.assertEqual('12345', temp_h5_file['parfile'].attrs['par_content'])
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # rpar_content in parfile_dict
        parfile_dict = {
            'rpar_content': '67890'
        }
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        _store_parameter_file(parfile_dict, temp_h5_file)
        self.assertTrue('parfile' in temp_h5_file.keys())
        self.assertFalse('par_content' in temp_h5_file['parfile'].attrs)
        self.assertTrue('rpar_content' in temp_h5_file['parfile'].attrs)
        self.assertEqual('67890', temp_h5_file['parfile'].attrs['rpar_content'])
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

    def test__all_relevant_data_filepaths(self):
        from mayawaves.utils.postprocessingutils import _all_relevant_data_filepaths
        from mayawaves.utils.postprocessingutils import _RadiativeFilenames
        from mayawaves.utils.postprocessingutils import _CompactObjectFilenames
        from mayawaves.utils.postprocessingutils import _MiscDataFilenames

        parameter_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                          "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/SIMFACTORY/par/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.rpar")
        with open(parameter_filepath, 'r') as f:
            parfile_content = f.read()

        # simulation has same name as parameter file
        actual_relevant_data_filepaths = _all_relevant_data_filepaths(
            TestPostprocessingUtils.raw_directory, parfile_content, TestPostprocessingUtils.parameter_file_name)

        actual_psi4_filetypes = set(list(actual_relevant_data_filepaths["radiative"].keys()))
        expected_psi4_filetypes = {_RadiativeFilenames.YLM_WEYLSCAL4_ASC}
        self.assertEqual(expected_psi4_filetypes, actual_psi4_filetypes)

        expected_filename_counts = {_RadiativeFilenames.YLM_WEYLSCAL4_ASC: 273}
        for filetype in actual_psi4_filetypes:
            filenames = list(actual_relevant_data_filepaths["radiative"][filetype].keys())
            actual_filename_count = len(filenames)
            self.assertEqual(expected_filename_counts[filetype], actual_filename_count)

            for filename in filenames:
                self.assertTrue(re.match(filetype.regex, filename))
                filepaths = actual_relevant_data_filepaths["radiative"][filetype][filename]
                self.assertEqual(len(filepaths), len(set(filepaths)))
                for filepath in filepaths:
                    self.assertTrue(filepath.endswith(filename))

        actual_compact_object_filetypes = set(list(actual_relevant_data_filepaths["compact_object"].keys()))
        expected_compact_object_filetypes = {_CompactObjectFilenames.BH_DIAGNOSTICS,
                                             _CompactObjectFilenames.SHIFTTRACKER,
                                             _CompactObjectFilenames.IHSPIN_HN}
        self.assertEqual(expected_compact_object_filetypes, actual_compact_object_filetypes)

        expected_filename_counts = {_CompactObjectFilenames.BH_DIAGNOSTICS: 3, _CompactObjectFilenames.SHIFTTRACKER: 2,
                                    _CompactObjectFilenames.IHSPIN_HN: 3}
        for filetype in actual_compact_object_filetypes:
            filenames = list(actual_relevant_data_filepaths["compact_object"][filetype].keys())
            actual_filename_count = len(filenames)
            self.assertEqual(expected_filename_counts[filetype], actual_filename_count)

            for filename in filenames:
                self.assertTrue(re.match(filetype.regex, filename))
                filepaths = actual_relevant_data_filepaths["compact_object"][filetype][filename]
                self.assertEqual(len(filepaths), len(set(filepaths)))
                for filepath in filepaths:
                    self.assertTrue(filepath.endswith(filename))

        actual_misc_filetypes = set(list(actual_relevant_data_filepaths["misc"].keys()))
        expected_misc_filetypes = {_MiscDataFilenames.RUNSTATS}
        self.assertEqual(expected_misc_filetypes, actual_misc_filetypes)

        expected_filename_counts = {_MiscDataFilenames.RUNSTATS: 1}
        for filetype in actual_misc_filetypes:
            filenames = list(actual_relevant_data_filepaths["misc"][filetype].keys())
            actual_filename_count = len(filenames)
            self.assertEqual(expected_filename_counts[filetype], actual_filename_count)

            for filename in filenames:
                self.assertTrue(re.match(filetype.value, filename))
                filepaths = actual_relevant_data_filepaths["misc"][filetype][filename]
                self.assertEqual(len(filepaths), len(set(filepaths)))
                for filepath in filepaths:
                    self.assertTrue(filepath.endswith(filename))

        # expect BH_diagnostics.ah1.gp and BH_diagnostics.ah2.gp to have 1 filepath and BH_diagnostics.ah3.gp to have 2 filepaths
        expected_BH_diagnostics_filepath_counts = [1, 1, 2]

        self.assertEqual(expected_BH_diagnostics_filepath_counts[0],
                         len(actual_relevant_data_filepaths["compact_object"][_CompactObjectFilenames.BH_DIAGNOSTICS][
                                 "BH_diagnostics.ah1.gp"]))
        self.assertEqual(expected_BH_diagnostics_filepath_counts[1],
                         len(actual_relevant_data_filepaths["compact_object"][_CompactObjectFilenames.BH_DIAGNOSTICS][
                                 "BH_diagnostics.ah2.gp"]))
        self.assertEqual(expected_BH_diagnostics_filepath_counts[2],
                         len(actual_relevant_data_filepaths["compact_object"][_CompactObjectFilenames.BH_DIAGNOSTICS][
                                 "BH_diagnostics.ah3.gp"]))

        # simulation has different name than parameter file
        actual_relevant_data_filepaths = _all_relevant_data_filepaths(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                                                   "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_different_name"),
                                                                      parfile_content,
                                                                      "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")
        actual_included_prefixes = set(list(actual_relevant_data_filepaths["radiative"].keys())
                                       + list(actual_relevant_data_filepaths["compact_object"].keys())
                                       + list(actual_relevant_data_filepaths["misc"].keys()))
        expected_included_prefixes = {_CompactObjectFilenames.BH_DIAGNOSTICS, _CompactObjectFilenames.IHSPIN_HN,
                                      _MiscDataFilenames.RUNSTATS, _CompactObjectFilenames.SHIFTTRACKER,
                                      _RadiativeFilenames.YLM_WEYLSCAL4_ASC}
        self.assertEqual(expected_included_prefixes, actual_included_prefixes)

        # test prestitched data
        actual_relevant_data_filepaths = _all_relevant_data_filepaths(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"), parfile_content,
            "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")

        actual_psi4_filetypes = set(list(actual_relevant_data_filepaths["radiative"].keys()))
        expected_psi4_filetypes = {_RadiativeFilenames.YLM_WEYLSCAL4_ASC}
        self.assertEqual(expected_psi4_filetypes, actual_psi4_filetypes)

        expected_filename_counts = {_RadiativeFilenames.YLM_WEYLSCAL4_ASC: 273}
        for filetype in actual_psi4_filetypes:
            filenames = list(actual_relevant_data_filepaths["radiative"][filetype].keys())
            actual_filename_count = len(filenames)
            self.assertEqual(expected_filename_counts[filetype], actual_filename_count)

            for filename in filenames:
                self.assertTrue(re.match(filetype.regex, filename))
                filepaths = actual_relevant_data_filepaths["radiative"][filetype][filename]
                self.assertEqual(len(filepaths), len(set(filepaths)))
                self.assertTrue(len(filepaths) == 1)
                for filepath in filepaths:
                    self.assertTrue(filepath.endswith(filename))

        actual_horizon_filetypes = set(list(actual_relevant_data_filepaths["compact_object"].keys()))
        expected_horizon_filetypes = {_CompactObjectFilenames.BH_DIAGNOSTICS, _CompactObjectFilenames.SHIFTTRACKER,
                                      _CompactObjectFilenames.IHSPIN_HN}
        self.assertEqual(expected_horizon_filetypes, actual_horizon_filetypes)

        expected_filename_counts = {_CompactObjectFilenames.BH_DIAGNOSTICS: 3, _CompactObjectFilenames.SHIFTTRACKER: 2,
                                    _CompactObjectFilenames.IHSPIN_HN: 3}
        for filetype in actual_horizon_filetypes:
            filenames = list(actual_relevant_data_filepaths["compact_object"][filetype].keys())
            actual_filename_count = len(filenames)
            self.assertEqual(expected_filename_counts[filetype], actual_filename_count)

            for filename in filenames:
                self.assertTrue(re.match(filetype.regex, filename))
                filepaths = actual_relevant_data_filepaths["compact_object"][filetype][filename]
                self.assertEqual(len(filepaths), len(set(filepaths)))
                self.assertTrue(len(filepaths) == 1)
                for filepath in filepaths:
                    self.assertTrue(filepath.endswith(filename))

        actual_misc_filetypes = set(list(actual_relevant_data_filepaths["misc"].keys()))
        expected_misc_filetypes = {_MiscDataFilenames.RUNSTATS}
        self.assertEqual(expected_misc_filetypes, actual_misc_filetypes)

        expected_filename_counts = {_MiscDataFilenames.RUNSTATS: 1}
        for filetype in actual_misc_filetypes:
            filenames = list(actual_relevant_data_filepaths["misc"][filetype].keys())
            actual_filename_count = len(filenames)
            self.assertEqual(expected_filename_counts[filetype], actual_filename_count)

            for filename in filenames:
                self.assertTrue(re.match(filetype.value, filename))
                filepaths = actual_relevant_data_filepaths["misc"][filetype][filename]
                self.assertEqual(len(filepaths), len(set(filepaths)))
                self.assertTrue(len(filepaths) == 1)
                for filepath in filepaths:
                    self.assertTrue(filepath.endswith(filename))

    def test__all_relevant_output_filepaths(self):
        from mayawaves.utils.postprocessingutils import _all_relevant_output_filepaths
        actual_relevant_output_filepaths = _all_relevant_output_filepaths(TestPostprocessingUtils.raw_directory)
        actual_included_filenames = list(actual_relevant_output_filepaths.keys())
        actual_included_filenames.sort()
        expected_included_filenames = ["D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out", "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.err"]
        expected_included_filenames.sort()
        self.assertEqual(expected_included_filenames, actual_included_filenames)

        expected_out_filecount = 2
        expected_err_filecount = 2
        self.assertEqual(expected_out_filecount,
                         len(actual_relevant_output_filepaths["D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out"]))
        self.assertEqual(expected_err_filecount,
                         len(actual_relevant_output_filepaths["D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.err"]))

        for filename in actual_included_filenames:
            filepaths = actual_relevant_output_filepaths[filename]
            self.assertEqual(len(filepaths), len(set(filepaths)))
            for filepath in filepaths:
                self.assertTrue(filepath.endswith(filename))

        # test prestitched data
        actual_relevant_output_filepaths = _all_relevant_output_filepaths(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/out_files"))
        actual_included_filenames = list(actual_relevant_output_filepaths.keys())
        actual_included_filenames.sort()
        expected_included_filenames = ["D9_q1.277_a0.644_0.676_theta_1.33_1.60_m240_modified.out", "stdout", "stdout-1",
                                       "stdout-2"]
        expected_included_filenames.sort()
        self.assertEqual(expected_included_filenames, actual_included_filenames)

        self.assertEqual(1,
                         len(actual_relevant_output_filepaths[
                                 "D9_q1.277_a0.644_0.676_theta_1.33_1.60_m240_modified.out"]))
        self.assertEqual(1,
                         len(actual_relevant_output_filepaths["stdout"]))
        self.assertEqual(1,
                         len(actual_relevant_output_filepaths["stdout-1"]))
        self.assertEqual(1,
                         len(actual_relevant_output_filepaths["stdout-2"]))

        for filename in actual_included_filenames:
            filepaths = actual_relevant_output_filepaths[filename]
            self.assertEqual(len(filepaths), len(set(filepaths)))
            for filepath in filepaths:
                self.assertTrue(filepath.endswith(filename))

    def test__stitch_timeseries_data(self):
        from mayawaves.utils.postprocessingutils import _stitch_timeseries_data
        filepaths = [
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/outputs/output-0000/psi4analysis_r75.00.asc"),
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/outputs/output-0001/psi4analysis_r75.00.asc"),
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/outputs/output-0002/psi4analysis_r75.00.asc")]
        stitched_data = _stitch_timeseries_data(filepaths)

        expected_stitched_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                  "resources/outputs/stitched/psi4analysis_r75.00.asc")
        expected_stitched_data = np.loadtxt(expected_stitched_filepath)

        self.assertTrue(np.all(np.isclose(stitched_data, expected_stitched_data, atol=1e-4)))

        # test stitching one file
        filepaths = [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc")]
        stitched_data = _stitch_timeseries_data(filepaths)

        expected_stitched_data = np.loadtxt(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                         "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc"))

        self.assertTrue(np.all(np.isclose(stitched_data, expected_stitched_data, atol=1e-4)))

        # test stitching data with only one row of data
        filepaths = [
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/outputs/output-0000/test_short_data.asc"),
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/outputs/output-0001/test_short_data.asc")]
        stitched_data = _stitch_timeseries_data(filepaths)

        expected_stitched_data = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])

        self.assertTrue(expected_stitched_data.shape == (2, 5))
        self.assertTrue(np.all(np.isclose(stitched_data, expected_stitched_data, atol=1e-4)))

    def test__l_m_radius_from_psi4_filename(self):
        from mayawaves.utils.postprocessingutils import _l_m_radius_from_psi4_filename

        expected = (3, 2, "40.00")
        actual = _l_m_radius_from_psi4_filename("Ylm_WEYLSCAL4::Psi4r_l3_m2_r40.00.asc")
        self.assertTrue(np.all(expected == actual))

        expected = (4, -4, "140.00")
        actual = _l_m_radius_from_psi4_filename("Ylm_WEYLSCAL4::Psi4r_l4_m-4_r140.00.asc")
        self.assertTrue(np.all(expected == actual))

        expected = (7, 5, "50.00")
        actual = _l_m_radius_from_psi4_filename("mp_Psi4r_l7_m5_r50.00.asc")
        self.assertTrue(np.all(expected == actual))

        expected = (7, -5, "100.00")
        actual = _l_m_radius_from_psi4_filename("mp_Psi4r_l7_m-5_r100.00.asc")
        self.assertTrue(np.all(expected == actual))

        expected = (6, 5, "60.00")
        actual = _l_m_radius_from_psi4_filename("mp_WeylScal4::Psi4i_l6_m5_r60.00.asc")
        self.assertTrue(np.all(expected == actual))

    def test__store_radiative_data(self):
        from mayawaves.utils.postprocessingutils import _store_radiative_data
        from mayawaves.utils.postprocessingutils import _RadiativeFilenames
        from mayawaves.utils.postprocessingutils import _stitch_timeseries_data

        # if only mp_Psi4

        # create a temporary h5 file
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        relevant_filepaths = {"radiative":
            {
                _RadiativeFilenames.MP_PSI4_ASC:
                    {
                        "mp_Psi4r_l2_m2_r75.00.asc": [
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0000/mp_Psi4r_l2_m2_r75.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0001/mp_Psi4r_l2_m2_r75.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0002/mp_Psi4r_l2_m2_r75.00.asc")
                        ]
                    }
            }
        }
        _store_radiative_data(temp_h5_file, relevant_filepaths)

        # check that it created the right groups
        self.assertEqual(["psi4"], list(temp_h5_file["radiative"].keys()))
        self.assertEqual("MP_PSI4_ASC", temp_h5_file["radiative"]["psi4"].attrs["source"])
        radius_groups = list(temp_h5_file["radiative"]["psi4"].keys())
        radius_groups.sort()
        self.assertEqual(["radius=75.00"], radius_groups)
        psi4_keys = list(temp_h5_file["radiative"]["psi4"]["radius=75.00"])
        psi4_keys.sort()
        self.assertEqual(["modes", "time"], psi4_keys)

        # clean up
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # If Ylm_WEYLSCAL4 and mp_Psi4, default to Ylm_WEYLSCAL4
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        relevant_filepaths = {"radiative":
            {
                _RadiativeFilenames.MP_PSI4_ASC:
                    {
                        "mp_Psi4r_l2_m2_r75.00.asc": [
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0000/mp_Psi4r_l2_m2_r75.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0001/mp_Psi4r_l2_m2_r75.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0002/mp_Psi4r_l2_m2_r75.00.asc")
                        ]
                    },
                _RadiativeFilenames.YLM_WEYLSCAL4_ASC: {
                    "Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc": [
                        os.path.join(TestPostprocessingUtils.CURR_DIR,
                                     "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc"),
                        os.path.join(TestPostprocessingUtils.CURR_DIR,
                                     "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc")
                    ]
                }
            }
        }
        _store_radiative_data(temp_h5_file, relevant_filepaths)

        # check that it created the right groups
        self.assertEqual(["psi4"], list(temp_h5_file["radiative"].keys()))
        self.assertEqual("YLM_WEYLSCAL4_ASC", temp_h5_file["radiative"]["psi4"].attrs["source"])
        radius_groups = list(temp_h5_file["radiative"]["psi4"].keys())
        radius_groups.sort()
        self.assertEqual(["radius=70.00"], radius_groups)
        psi4_keys = list(temp_h5_file["radiative"]["psi4"]["radius=70.00"])
        psi4_keys.sort()
        self.assertEqual(["modes", "time"], psi4_keys)

        # clean up
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # create a temporary h5 file
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        relevant_filepaths = {"radiative":
            {
                _RadiativeFilenames.YLM_WEYLSCAL4_ASC:
                    {
                        "Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc": [
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc")
                        ],
                        "Ylm_WEYLSCAL4::Psi4r_l2_m2_r80.00.asc": [
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r80.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r80.00.asc")
                        ],
                        "Ylm_WEYLSCAL4::Psi4r_l3_m2_r70.00.asc": [
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l3_m2_r70.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l3_m2_r70.00.asc")
                        ],
                        "Ylm_WEYLSCAL4::Psi4r_l3_m3_r70.00.asc": [
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l3_m3_r70.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l3_m3_r70.00.asc")
                        ]
                    }
            }
        }
        _store_radiative_data(temp_h5_file, relevant_filepaths)

        # check that it created the right groups
        self.assertEqual(["radiative"], list(temp_h5_file.keys()))
        self.assertEqual(["psi4"], list(temp_h5_file["radiative"].keys()))
        radius_groups = list(temp_h5_file["radiative"]["psi4"].keys())
        radius_groups.sort()
        self.assertEqual(["radius=70.00", "radius=80.00"], radius_groups)
        psi4_keys = list(temp_h5_file["radiative"]["psi4"]["radius=70.00"])
        psi4_keys.sort()
        self.assertEqual(["modes", "time"], psi4_keys)
        r70_l_groups = list(temp_h5_file["radiative"]["psi4"]["radius=70.00"]["modes"].keys())
        r70_l_groups.sort()
        self.assertEqual(["l=2", "l=3"], r70_l_groups)
        r80_l_groups = list(temp_h5_file["radiative"]["psi4"]["radius=80.00"]["modes"].keys())
        r80_l_groups.sort()
        self.assertEqual(["l=2"], r80_l_groups)
        m_groups = list(temp_h5_file["radiative"]["psi4"]["radius=70.00"]["modes"]["l=3"].keys())
        m_groups.sort()
        self.assertEqual(["m=2", "m=3"], m_groups)

        # check that it stored the stitched data
        stitched_psi4_data = _stitch_timeseries_data(
            relevant_filepaths["radiative"][_RadiativeFilenames.YLM_WEYLSCAL4_ASC][
                "Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc"])
        self.assertTrue(
            np.all(stitched_psi4_data[:, 0] == temp_h5_file["radiative"]["psi4"]["radius=70.00"]["time"][()]))
        self.assertTrue(
            np.all(stitched_psi4_data[:, 1] ==
                   temp_h5_file["radiative"]["psi4"]["radius=70.00"]["modes"]["l=2"]["m=2"]["real"][()]))
        self.assertTrue(
            np.all(stitched_psi4_data[:, 2] ==
                   temp_h5_file["radiative"]["psi4"]["radius=70.00"]["modes"]["l=2"]["m=2"]["imaginary"][
                       ()]))

        # clean up
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # multipole

        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        relevant_filepaths = {"radiative":
            {
                _RadiativeFilenames.MP_PSI4_ASC:
                    {
                        "mp_Psi4r_l2_m2_r75.00.asc": [
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0000/mp_Psi4r_l2_m2_r75.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0001/mp_Psi4r_l2_m2_r75.00.asc"),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/outputs/output-0002/mp_Psi4r_l2_m2_r75.00.asc")
                        ]
                    }
            }
        }
        _store_radiative_data(temp_h5_file, relevant_filepaths)

        # check that it created the right groups
        self.assertEqual(["radiative"], list(temp_h5_file.keys()))
        radius_groups = list(temp_h5_file["radiative"]["psi4"].keys())
        radius_groups.sort()
        self.assertEqual(["radius=75.00"], radius_groups)
        psi4_keys = list(temp_h5_file["radiative"]["psi4"]["radius=75.00"])
        psi4_keys.sort()
        self.assertEqual(["modes", "time"], psi4_keys)
        r70_l_groups = list(temp_h5_file["radiative"]["psi4"]["radius=75.00"]["modes"].keys())
        r70_l_groups.sort()
        self.assertEqual(["l=2"], r70_l_groups)
        m_groups = list(temp_h5_file["radiative"]["psi4"]["radius=75.00"]["modes"]["l=2"].keys())
        m_groups.sort()
        self.assertEqual(["m=2"], m_groups)

        # check that it stored the stitched data
        stitched_psi4_data = _stitch_timeseries_data(
            relevant_filepaths["radiative"][_RadiativeFilenames.MP_PSI4_ASC]["mp_Psi4r_l2_m2_r75.00.asc"])
        self.assertTrue(
            np.all(stitched_psi4_data[:, 0] == temp_h5_file["radiative"]["psi4"]["radius=75.00"]["time"][()]))
        self.assertTrue(
            np.all(stitched_psi4_data[:, 1] ==
                   temp_h5_file["radiative"]["psi4"]["radius=75.00"]["modes"]["l=2"]["m=2"]["real"][()]))
        self.assertTrue(
            np.all(stitched_psi4_data[:, 2] ==
                   temp_h5_file["radiative"]["psi4"]["radius=75.00"]["modes"]["l=2"]["m=2"]["imaginary"][
                       ()]))

        # clean up
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

    def test__store_compact_object_data(self):
        from mayawaves.utils.postprocessingutils import _store_compact_object_data
        from mayawaves.utils.postprocessingutils import _CompactObjectFilenames
        from mayawaves.utils.postprocessingutils import _BH_diagnostics
        from mayawaves.utils.postprocessingutils import _Ihspin_hn
        from mayawaves.utils.postprocessingutils import _Shifttracker
        from mayawaves.compactobject import CompactObject

        relevant_filepaths = {
            "compact_object":
                {
                    _CompactObjectFilenames.SHIFTTRACKER:
                        {
                            "ShiftTracker0.asc":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/ShiftTracker0.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name),
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0001/%s/ShiftTracker0.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ],
                            "ShiftTracker1.asc":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/ShiftTracker1.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name),
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0001/%s/ShiftTracker1.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ]
                        },
                    _CompactObjectFilenames.BH_DIAGNOSTICS:
                        {
                            "BH_diagnostics.ah1.gp":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/BH_diagnostics.ah1.gp") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ],
                            "BH_diagnostics.ah2.gp":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/BH_diagnostics.ah2.gp") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ],
                            "BH_diagnostics.ah3.gp":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/BH_diagnostics.ah3.gp") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name),
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0001/%s/BH_diagnostics.ah3.gp") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ]
                        },
                    _CompactObjectFilenames.IHSPIN_HN:
                        {
                            "ihspin_hn_0.asc":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_0.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ],
                            "ihspin_hn_1.asc":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_1.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ],
                            "ihspin_hn_2.asc":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/ihspin_hn_2.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name),
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0001/%s/ihspin_hn_2.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ]
                        }
                }
        }
        parameter_file_path = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                           "resources/main_test_simulation/%s/output-0000/%s/%s.par") % (
                                  TestPostprocessingUtils.simulation_name,
                                  TestPostprocessingUtils.simulation_name,
                                  TestPostprocessingUtils.simulation_name)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parameter_file_path) as f:
            parfile_group.attrs['par_content'] = f.read()

        _store_compact_object_data(temp_h5_file, relevant_filepaths)

        compact_object_keys = list(temp_h5_file["compact_object"].keys())
        compact_object_keys.sort()
        self.assertEqual(["object=0", "object=1", "object=2"], compact_object_keys)

        self.assertEqual(52, temp_h5_file["compact_object"]["object=0"][()].shape[1])
        self.assertEqual(52, temp_h5_file["compact_object"]["object=1"][()].shape[1])
        self.assertEqual(46, temp_h5_file["compact_object"]["object=2"][()].shape[1])

        # object 0
        compact_object_data = temp_h5_file["compact_object"]["object=0"]
        header_list = compact_object_data.attrs["header"].tolist()
        # BH_diagnostics
        bh_diagnostics_column_idxs = [header_list.index(col.header_text) for col in _BH_diagnostics.column_list if
                                      col is not None]

        actual_BH_diagnostics_data = compact_object_data[
                                         ~np.isnan(compact_object_data[:, bh_diagnostics_column_idxs[-1]])][:,
                                     bh_diagnostics_column_idxs]
        expected_BH_diagnostics_data = np.loadtxt(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah1.gp") % TestPostprocessingUtils.simulation_name)
        # check info only in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 5:], actual_BH_diagnostics_data[:, 5:], atol=0.001)))
        # check position data
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 2:5], actual_BH_diagnostics_data[:, 2:5], atol=0.05)))
        # check iteration column
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 0], actual_BH_diagnostics_data[:, 0], rtol=0.001)))
        # check time column
        # the time column is rounded off in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 1], actual_BH_diagnostics_data[:, 1], atol=0.001)))

        # ihspin_hn
        ih_spin_column_idxs = [header_list.index(col.header_text) for col in _Ihspin_hn.column_list if col is not None]
        ih_spin_column_idxs.insert(0, header_list.index(CompactObject.Column.TIME.header_text))
        actual_ihspin_hn_data = compact_object_data[~np.isnan(compact_object_data[:, ih_spin_column_idxs[-1]])][:,
                                ih_spin_column_idxs]
        expected_ihspin_hn_data = np.loadtxt(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/%s/ihspin_hn_0.asc") % TestPostprocessingUtils.simulation_name)
        self.assertTrue(np.all(np.isclose(expected_ihspin_hn_data, actual_ihspin_hn_data, atol=1e-3)))

        # ShiftTracker
        shift_tracker_column_idxs = [header_list.index(col.header_text) for col in _Shifttracker.column_list if
                                     col is not None]
        actual_shifttracker_data = compact_object_data[
                                       ~np.isnan(compact_object_data[:, shift_tracker_column_idxs[-1]])][:,
                                   shift_tracker_column_idxs]
        expected_shifttracker_data = np.loadtxt(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/%s/ShiftTracker0.asc") % TestPostprocessingUtils.simulation_name)
        self.assertTrue(np.all(np.isclose(expected_shifttracker_data, actual_shifttracker_data, atol=1e-3)))

        # object 1
        compact_object_data = temp_h5_file["compact_object"]["object=1"]
        header_list = compact_object_data.attrs["header"].tolist()

        # BH_diagnostics
        bh_diagnostics_column_idxs = [header_list.index(col.header_text) for col in _BH_diagnostics.column_list if
                                      col is not None]
        actual_BH_diagnostics_data = compact_object_data[
                                         ~np.isnan(compact_object_data[:, bh_diagnostics_column_idxs[-1]])][:,
                                     bh_diagnostics_column_idxs]
        expected_BH_diagnostics_data = np.loadtxt(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah2.gp") % TestPostprocessingUtils.simulation_name)
        # check info only in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 5:], actual_BH_diagnostics_data[:, 5:], atol=0.001)))
        # check position data
        # TODO consider this atol
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 2:5], actual_BH_diagnostics_data[:, 2:5], atol=0.05)))
        # check iteration column
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 0], actual_BH_diagnostics_data[:, 0], rtol=0.001)))
        # check time column
        # the time column is rounded off in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 1], actual_BH_diagnostics_data[:, 1], atol=0.001)))

        # ihspin_hn
        ih_spin_column_idxs = [header_list.index(col.header_text) for col in _Ihspin_hn.column_list if col is not None]
        ih_spin_column_idxs.insert(0, header_list.index(CompactObject.Column.TIME.header_text))
        actual_ihspin_hn_data = compact_object_data[~np.isnan(compact_object_data[:, ih_spin_column_idxs[-1]])][:,
                                ih_spin_column_idxs]
        expected_ihspin_hn_data = np.loadtxt(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/%s/ihspin_hn_1.asc") % TestPostprocessingUtils.simulation_name)
        self.assertTrue(np.all(np.isclose(expected_ihspin_hn_data, actual_ihspin_hn_data, atol=1e-3)))

        # ShiftTracker
        shift_tracker_column_idxs = [header_list.index(col.header_text) for col in _Shifttracker.column_list if
                                     col is not None]
        actual_shifttracker_data = compact_object_data[
                                       ~np.isnan(compact_object_data[:, shift_tracker_column_idxs[-1]])][:,
                                   shift_tracker_column_idxs]
        expected_shifttracker_data = np.loadtxt(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/%s/ShiftTracker1.asc") % TestPostprocessingUtils.simulation_name)
        self.assertTrue(np.all(np.isclose(expected_shifttracker_data, actual_shifttracker_data, atol=1e-3)))

        # object 2
        compact_object_data = temp_h5_file["compact_object"]["object=2"]
        header_list = compact_object_data.attrs["header"].tolist()

        # BH_diagnostics
        bh_diagnostics_column_idxs = [header_list.index(col.header_text) for col in _BH_diagnostics.column_list if
                                      col is not None]
        actual_BH_diagnostics_data = compact_object_data[
                                         ~np.isnan(compact_object_data[:, bh_diagnostics_column_idxs[-1]])][:,
                                     bh_diagnostics_column_idxs]
        expected_BH_diagnostics_data = np.loadtxt(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/%s/BH_diagnostics.ah3.gp") % TestPostprocessingUtils.simulation_name)
        # check info only in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 5:], actual_BH_diagnostics_data[:, 5:], atol=0.001)))
        # check position data
        # TODO consider this atol
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 2:5], actual_BH_diagnostics_data[:, 2:5], atol=0.05)))
        # check iteration column
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 0], actual_BH_diagnostics_data[:, 0], rtol=0.001)))
        # check time column
        # the time column is rounded off in BH_diagnostics
        self.assertTrue(np.all(
            np.isclose(expected_BH_diagnostics_data[:, 1], actual_BH_diagnostics_data[:, 1], atol=0.001)))

        # ihspin_hn
        ih_spin_column_idxs = [header_list.index(col.header_text) for col in _Ihspin_hn.column_list if col is not None]
        ih_spin_column_idxs.insert(0, header_list.index(CompactObject.Column.TIME.header_text))
        actual_ihspin_hn_data = compact_object_data[~np.isnan(compact_object_data[:, ih_spin_column_idxs[-1]])][:,
                                ih_spin_column_idxs]
        expected_ihspin_hn_data = np.loadtxt(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/%s/ihspin_hn_2.asc") % TestPostprocessingUtils.simulation_name)
        self.assertTrue(np.all(np.isclose(expected_ihspin_hn_data, actual_ihspin_hn_data, atol=1e-3)))

        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

    def test__get_data_from_columns(self):
        from mayawaves.utils.postprocessingutils import _get_data_from_columns
        from mayawaves.compactobject import CompactObject

        header = [CompactObject.Column.TIME.header_text, CompactObject.Column.SX.header_text,
                  CompactObject.Column.AX.header_text, CompactObject.Column.M_IRREDUCIBLE.header_text]
        full_data = np.array([[1, np.nan, 3, np.nan], [10, 20, 30, np.nan], [100, 200, 300, np.nan]])

        # one column with all the data
        columns = [CompactObject.Column.AX]
        expected_data = np.array([3, 30, 300])
        actual_data = _get_data_from_columns(full_data, header, columns)
        self.assertTrue(np.all(expected_data == actual_data))

        # one column missing some data
        columns = [CompactObject.Column.SX]
        expected_data = np.array([20, 200])
        actual_data = _get_data_from_columns(full_data, header, columns)
        self.assertTrue(np.all(expected_data == actual_data))

        # multiple columns with all data
        columns = [CompactObject.Column.TIME, CompactObject.Column.AX]
        expected_data = np.array([[1, 3], [10, 30], [100, 300]])
        actual_data = _get_data_from_columns(full_data, header, columns)
        self.assertTrue(np.all(expected_data == actual_data))

        # multiple columns missing some data
        columns = [CompactObject.Column.TIME, CompactObject.Column.SX, CompactObject.Column.AX]
        expected_data = np.array([[10, 20, 30], [100, 200, 300]])
        actual_data = _get_data_from_columns(full_data, header, columns)
        self.assertTrue(np.all(expected_data == actual_data))

        # non present column
        columns = [CompactObject.Column.X]
        actual_data = _get_data_from_columns(full_data, header, columns)
        self.assertIsNone(actual_data)

        # column with only nans
        columns = [CompactObject.Column.M_IRREDUCIBLE]
        expected_data = np.array([])
        actual_data = _get_data_from_columns(full_data, header, columns)
        self.assertTrue(np.all(expected_data == actual_data))

        # multiple columns with one all nans
        columns = [CompactObject.Column.TIME, CompactObject.Column.M_IRREDUCIBLE]
        expected_data = np.empty((0, 2))
        actual_data = _get_data_from_columns(full_data, header, columns)
        self.assertTrue(np.all(expected_data == actual_data))

    def test__get_dimensional_spin_from_parfile(self):
        from mayawaves.utils.postprocessingutils import _get_dimensional_spin_from_parfile

        # no parameter file data
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        actual_spins = _get_dimensional_spin_from_parfile(parfile_group)
        self.assertIsNone(actual_spins)
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # rpar and par data
        rparfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                "resources/main_test_simulation/%s/output-0000/%s.rpar") % (
                       TestPostprocessingUtils.simulation_name, TestPostprocessingUtils.simulation_name)
        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/%s/output-0000/%s.par") % (
                      TestPostprocessingUtils.simulation_name, TestPostprocessingUtils.simulation_name)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(rparfile) as f:
            rpar_content = f.read()
        parfile_group.attrs['rpar_content'] = rpar_content
        with open(parfile) as f:
            par_content = f.read()
        parfile_group.attrs['par_content'] = par_content
        actual_spins = _get_dimensional_spin_from_parfile(parfile_group)
        expected_spins = ([0, 0, 0], [0, 0, 0])
        self.assertTrue(np.all(np.isclose(expected_spins[0], actual_spins[0], atol=1e-4)))
        self.assertTrue(np.all(np.isclose(expected_spins[1], actual_spins[1], atol=1e-4)))
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # par data only
        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/main_test_simulation/%s/output-0000/%s.par") % (
                      TestPostprocessingUtils.simulation_name, TestPostprocessingUtils.simulation_name)
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parfile) as f:
            par_content = f.read()
        parfile_group.attrs['par_content'] = par_content
        actual_spins = _get_dimensional_spin_from_parfile(parfile_group)
        expected_spins = ([0, 0, 0], [0, 0, 0])
        self.assertTrue(np.all(np.isclose(expected_spins[0], actual_spins[0], atol=1e-4)))
        self.assertTrue(np.all(np.isclose(expected_spins[1], actual_spins[1], atol=1e-4)))
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/parfiles/D11_q5_a1_0_0_0_a2_0_0_-0.4_m302.4.par")
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parfile) as f:
            par_content = f.read()
        parfile_group.attrs['par_content'] = par_content
        actual_spins = _get_dimensional_spin_from_parfile(parfile_group)
        expected_spins = ([0, 0, 0], [0, 0, -0.011111111111111])
        self.assertTrue(np.all(np.isclose(expected_spins[0], actual_spins[0], atol=1e-4)))
        self.assertTrue(np.all(np.isclose(expected_spins[1], actual_spins[1], atol=1e-4)))
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/parfiles/D9_q1.277_a0.644_0.676_theta_1.33_1.60_m240.par")
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parfile) as f:
            par_content = f.read()
        parfile_group.attrs['par_content'] = par_content
        actual_spins = _get_dimensional_spin_from_parfile(parfile_group)
        expected_spins = ([-0.084257801444721, 0.020024065762755, 0.189357759132091],
                          [-0.059836322976587, -0.001958417738138, -0.115910893880460])
        self.assertTrue(np.all(np.isclose(expected_spins[0], actual_spins[0], atol=1e-4)))
        self.assertTrue(np.all(np.isclose(expected_spins[1], actual_spins[1], atol=1e-4)))
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

    def test__get_initial_dimensional_spins(self):
        from mayawaves.utils.postprocessingutils import _get_initial_dimensional_spins

        # all the data is in the h5 file
        h5_file = h5py.File(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5"))
        actual_dimensional_spin_0, actual_dimensional_spin_1 = _get_initial_dimensional_spins(h5_file)
        expected_dimensional_spin_0 = [0, 0, 0]
        expected_dimensional_spin_1 = [0, 0, 0]
        self.assertTrue(np.all(np.isclose(expected_dimensional_spin_0, actual_dimensional_spin_0, atol=1e-3)))
        self.assertTrue(np.all(np.isclose(expected_dimensional_spin_1, actual_dimensional_spin_1, atol=1e-3)))

        # the mass data is missing from the h5 file
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        compact_object = temp_h5_file.create_group("compact_object")
        object_0_data = np.full((10, 52), np.nan)
        object_1_data = np.full((10, 52), np.nan)
        header = ["itt", "time",
                  "x", "y", "z",
                  "vx", "vy", "vz",
                  "ax", "ay", "az",
                  "Sx", "Sy", "Sz",
                  "Px", "Py", "Pz",
                  "min radius", "max radius", "mean radius",
                  "quadrupole_xx", "quadrupole_xy", "quadrupole_xz",
                  "quadrupole_yy", "quadrupole_yz", "quadrupole_zz",
                  "min x", "max x", "min y", "max y", "min z", "max z",
                  "xy-plane circumference", "xz-plane circumference", "yz-plane circumference",
                  "ratio of xz/xy-plane circumferences", "ratio of yz/xy-plane circumferences",
                  "area",
                  "m_irreducible", "areal radius", "expansion Theta_(l)",
                  "inner expansion Theta_(n)",
                  "product of the expansions", "mean curvature", "gradient of the areal radius",
                  "gradient of the expansion Theta_(l)",
                  "gradient of the inner expansion Theta_(n)",
                  "gradient of the product of the expansions", "gradient of the mean curvature",
                  "minimum  of the mean curvature", "maximum  of the mean curvature",
                  "integral of the mean curvature"]
        object_0 = compact_object.create_dataset("object=0", data=object_0_data)
        object_0.attrs["header"] = header
        object_1 = compact_object.create_dataset("object=1", data=object_1_data)
        object_1.attrs["header"] = header
        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/parfiles/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par")
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parfile) as f:
            parfile_group.attrs['par_content'] = f.read()
        actual_dimensional_spin_0, actual_dimensional_spin_1 = _get_initial_dimensional_spins(temp_h5_file)
        expected_dimensional_spin_0 = [0, 0, 0]
        expected_dimensional_spin_1 = [0, 0, 0]

        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        self.assertTrue(np.all(np.isclose(expected_dimensional_spin_0, actual_dimensional_spin_0, atol=1e-3)))
        self.assertTrue(np.all(np.isclose(expected_dimensional_spin_1, actual_dimensional_spin_1, atol=1e-3)))

        h5_file.close()

    def test__get_initial_dimensionless_spins(self):
        from mayawaves.utils.postprocessingutils import _get_initial_dimensionless_spins

        dimensional_spin_0 = [0, 0, 0]
        dimensional_spin_1 = [0, 0, 0]
        horizon_mass_0 = 0.5
        horizon_mass_1 = 0.5
        actual_dimensionless_spins = _get_initial_dimensionless_spins(dimensional_spin_0, dimensional_spin_1,
                                                                      horizon_mass_0, horizon_mass_1)
        expected_dimensionless_spins = ([0, 0, 0], [0, 0, 0])
        self.assertTrue(np.all(np.isclose(expected_dimensionless_spins[0], actual_dimensionless_spins[0], atol=1e-4)))
        self.assertTrue(np.all(np.isclose(expected_dimensionless_spins[1], actual_dimensionless_spins[1], atol=1e-4)))

        dimensional_spin_0 = [0.01, 0.02, 0.05]
        dimensional_spin_1 = [0.02, 0.05, 0.1]
        horizon_mass_0 = 0.3
        horizon_mass_1 = 0.6
        actual_dimensionless_spins = _get_initial_dimensionless_spins(dimensional_spin_0, dimensional_spin_1,
                                                                      horizon_mass_0, horizon_mass_1)
        expected_dimensionless_spins = ([0.111111, 0.222222, 0.555556], [0.0555556, 0.138889, 0.277778])
        self.assertTrue(np.all(np.isclose(expected_dimensionless_spins[0], actual_dimensionless_spins[0], atol=1e-4)))
        self.assertTrue(np.all(np.isclose(expected_dimensionless_spins[1], actual_dimensionless_spins[1], atol=1e-4)))

    def test__get_spin_configuration(self):
        from mayawaves.utils.postprocessingutils import _get_spin_configuration

        expected = "non-spinning"
        actual = _get_spin_configuration([0, 0, 0], [0, 0, 0])
        self.assertEqual(expected, actual)

        expected = "non-spinning"
        actual = _get_spin_configuration([1e-5, 1e-6, 1e-8], [1e-5, 1e-7, 1e-12])
        self.assertEqual(expected, actual)

        expected = "aligned-spins"
        actual = _get_spin_configuration([1e-5, 1e-6, 0.8], [1e-5, 1e-7, -0.2])
        self.assertEqual(expected, actual)

        expected = "aligned-spins"
        actual = _get_spin_configuration([0, 0, 0.8], [0, 0, 0])
        self.assertEqual(expected, actual)

        expected = "aligned-spins"
        actual = _get_spin_configuration([0, 0, 0], [0, 0, 0.2])
        self.assertEqual(expected, actual)

        expected = "precessing"
        actual = _get_spin_configuration([0, 0, 0], [0.1, 0, 0])
        self.assertEqual(expected, actual)

        expected = "precessing"
        actual = _get_spin_configuration([0, 0, 0], [0, 0.1, 0])
        self.assertEqual(expected, actual)

        expected = "precessing"
        actual = _get_spin_configuration([0, 0.1, 0], [0, 0, 0])
        self.assertEqual(expected, actual)

        expected = "precessing"
        actual = _get_spin_configuration([0.1, 0, 0], [0, 0, 0])
        self.assertEqual(expected, actual)

        expected = "precessing"
        actual = _get_spin_configuration([0, 0.2, 0.8], [0.1, 0, 0.4])
        self.assertEqual(expected, actual)

    def test__irreducible_mass_to_horizon_mass(self):
        from mayawaves.utils.postprocessingutils import _irreducible_mass_to_horizon_mass

        expected = 0.5
        actual = _irreducible_mass_to_horizon_mass(0.5, [0, 0, 0])
        self.assertEqual(expected, actual)

        expected = 0.8
        actual = _irreducible_mass_to_horizon_mass(0.8, [0, 0, 0])
        self.assertEqual(expected, actual)

        expected = 0.343188
        actual = _irreducible_mass_to_horizon_mass(0.3, [0.1, 0, 0])
        self.assertTrue(np.isclose(expected, actual, atol=1e-4))

        expected = 0.65532
        actual = _irreducible_mass_to_horizon_mass(0.6, [0.1, 0, 0.3])
        self.assertTrue(np.isclose(expected, actual, atol=1e-4))

    def test__horizon_mass_to_irreducible_mass(self):
        from mayawaves.utils.postprocessingutils import _horizon_mass_to_irreducible_mass

        expected = 0.5
        actual = _horizon_mass_to_irreducible_mass(0.5, [0, 0, 0])
        self.assertEqual(expected, actual)

        expected = 0.8
        actual = _horizon_mass_to_irreducible_mass(0.8, [0, 0, 0])
        self.assertEqual(expected, actual)

        expected = 0.594067
        actual = _horizon_mass_to_irreducible_mass(0.6, [0.1, 0, 0])
        self.assertTrue(np.isclose(expected, actual, atol=1e-4))

        expected = 0.51577
        actual = _horizon_mass_to_irreducible_mass(0.6, [0.1, 0, 0.3])
        self.assertTrue(np.isclose(expected, actual, atol=1e-4))

    def test__get_masses_from_out_file(self):
        from mayawaves.utils.postprocessingutils import _get_masses_from_out_file

        relevant_output_files = {"D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out": [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out"),
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out")]}
        dimensional_spin_0 = [0, 0, 0]
        dimensional_spin_1 = [0, 0, 0]
        actual_irreducible_mass_0, actual_irreducible_mass_1, actual_horizon_mass_0, actual_horizon_mass_1 = _get_masses_from_out_file(
            relevant_output_files, dimensional_spin_0, dimensional_spin_1)
        expected_horizon_mass_0 = 0.516817
        expected_horizon_mass_1 = 0.516817
        expected_irreducible_mass_0 = 0.516817
        expected_irreducible_mass_1 = 0.516817
        self.assertTrue(np.isclose(expected_horizon_mass_0, actual_horizon_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_horizon_mass_1, actual_horizon_mass_1, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_0, actual_irreducible_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_1, actual_irreducible_mass_1, atol=1e-4))

        relevant_output_files = {"D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out": [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out"),
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out")]}
        dimensional_spin_0 = [0, 0.1, 0]
        dimensional_spin_1 = [0.05, 0.1, 0]
        actual_irreducible_mass_0, actual_irreducible_mass_1, actual_horizon_mass_0, actual_horizon_mass_1 = _get_masses_from_out_file(
            relevant_output_files, dimensional_spin_0, dimensional_spin_1)
        expected_horizon_mass_0 = 0.516817
        expected_horizon_mass_1 = 0.516817
        expected_irreducible_mass_0 = 0.507333
        expected_irreducible_mass_1 = 0.504814
        self.assertTrue(np.isclose(expected_horizon_mass_0, actual_horizon_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_horizon_mass_1, actual_horizon_mass_1, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_0, actual_irreducible_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_1, actual_irreducible_mass_1, atol=1e-4))

        relevant_output_files = {"D9_q1.277_a0.644_0.676_theta_1.33_1.60_m240_modified.out": [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/out_files/D9_q1.277_a0.644_0.676_theta_1.33_1.60_m240_modified.out")],
            "stdout": [os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/out_files/stdout")]}
        dimensional_spin_0 = [0.1, 0.1, 0]
        dimensional_spin_1 = [0.1, 0, 0]
        actual_irreducible_mass_0, actual_irreducible_mass_1, actual_horizon_mass_0, actual_horizon_mass_1 = _get_masses_from_out_file(
            relevant_output_files, dimensional_spin_0, dimensional_spin_1)
        expected_horizon_mass_0 = 0.560826
        expected_horizon_mass_1 = 0.439174
        expected_irreducible_mass_0 = 0.545648
        expected_irreducible_mass_1 = 0.422965
        self.assertTrue(np.isclose(expected_horizon_mass_0, actual_horizon_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_horizon_mass_1, actual_horizon_mass_1, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_0, actual_irreducible_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_1, actual_irreducible_mass_1, atol=1e-4))

    def test__get_initial_masses(self):
        from mayawaves.utils.postprocessingutils import _get_initial_masses

        # all the data is in the h5 file
        h5_file = h5py.File(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5"))
        dimensional_spin_0 = [0, 0.1, 0]
        dimensional_spin_1 = [0.05, 0.1, 0]
        relevant_output_files = {"D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out": [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out"),
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out")]}
        actual_irreducible_mass_0, actual_irreducible_mass_1, actual_horizon_mass_0, actual_horizon_mass_1 = _get_initial_masses(
            h5_file, dimensional_spin_0, dimensional_spin_1, relevant_output_files)
        expected_irreducible_mass_0 = 0.5167970311
        expected_irreducible_mass_1 = 0.5167970311
        expected_horizon_mass_0 = 0.525775
        expected_horizon_mass_1 = 0.527996
        self.assertTrue(np.isclose(expected_horizon_mass_0, actual_horizon_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_horizon_mass_1, actual_horizon_mass_1, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_0, actual_irreducible_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_1, actual_irreducible_mass_1, atol=1e-4))

        # the mass data is missing from the h5 file
        # all the data is in the h5 file
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        compact_object = temp_h5_file.create_group("compact_object")
        object_0_data = np.full((10, 52), np.nan)
        object_1_data = np.full((10, 52), np.nan)
        header = ["itt", "time",
                  "x", "y", "z",
                  "vx", "vy", "vz",
                  "ax", "ay", "az",
                  "Sx", "Sy", "Sz",
                  "Px", "Py", "Pz",
                  "min radius", "max radius", "mean radius",
                  "quadrupole_xx", "quadrupole_xy", "quadrupole_xz",
                  "quadrupole_yy", "quadrupole_yz", "quadrupole_zz",
                  "min x", "max x", "min y", "max y", "min z", "max z",
                  "xy-plane circumference", "xz-plane circumference", "yz-plane circumference",
                  "ratio of xz/xy-plane circumferences", "ratio of yz/xy-plane circumferences",
                  "area",
                  "m_irreducible", "areal radius", "expansion Theta_(l)",
                  "inner expansion Theta_(n)",
                  "product of the expansions", "mean curvature", "gradient of the areal radius",
                  "gradient of the expansion Theta_(l)",
                  "gradient of the inner expansion Theta_(n)",
                  "gradient of the product of the expansions", "gradient of the mean curvature",
                  "minimum  of the mean curvature", "maximum  of the mean curvature",
                  "integral of the mean curvature"]
        object_0 = compact_object.create_dataset("object=0", data=object_0_data)
        object_0.attrs["header"] = header
        object_1 = compact_object.create_dataset("object=1", data=object_1_data)
        object_1.attrs["header"] = header
        dimensional_spin_0 = [0, 0, 0]
        dimensional_spin_1 = [0, 0, 0]
        relevant_output_files = {"D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out": [
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out"),
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/output-0001/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.out")]}
        actual_irreducible_mass_0, actual_irreducible_mass_1, actual_horizon_mass_0, actual_horizon_mass_1 = _get_initial_masses(
            temp_h5_file, dimensional_spin_0, dimensional_spin_1, relevant_output_files)
        expected_irreducible_mass_0 = 0.5167970311
        expected_irreducible_mass_1 = 0.5167970311
        expected_horizon_mass_0 = 0.5167970311
        expected_horizon_mass_1 = 0.5167970311

        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        self.assertTrue(np.isclose(expected_horizon_mass_0, actual_horizon_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_horizon_mass_1, actual_horizon_mass_1, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_0, actual_irreducible_mass_0, atol=1e-4))
        self.assertTrue(np.isclose(expected_irreducible_mass_1, actual_irreducible_mass_1, atol=1e-4))

        h5_file.close()

    def test__get_initial_separation_from_parfile(self):
        from mayawaves.utils.postprocessingutils import _get_initial_separation_from_parfile

        # if no parfile information, should return None
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        actual_initial_separation = _get_initial_separation_from_parfile(parfile_group)
        self.assertIsNone(actual_initial_separation)
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # if rpar and par
        rparfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                "resources/parfiles/D11_q5_a1_0_0_0_a2_0_0_-0.4_m302.4.rpar")
        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/parfiles/D11_q5_a1_0_0_0_a2_0_0_-0.4_m302.4.par")
        expected_initial_separation = 11
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(rparfile) as f:
            rpar_content = f.read()
        parfile_group.attrs['rpar_content'] = rpar_content
        with open(parfile) as f:
            par_content = f.read()
        parfile_group.attrs['par_content'] = par_content
        actual_initial_separation = _get_initial_separation_from_parfile(parfile_group)
        self.assertEqual(expected_initial_separation, actual_initial_separation)

        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # if only par
        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/parfiles/D11_q5_a1_0_0_0_a2_0_0_-0.4_m302.4.par")
        expected_initial_separation = 11
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parfile) as f:
            par_content = f.read()
        parfile_group.attrs['par_content'] = par_content
        actual_initial_separation = _get_initial_separation_from_parfile(parfile_group)
        self.assertEqual(expected_initial_separation, actual_initial_separation)

        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

    def test__get_initial_separation(self):
        from mayawaves.utils.postprocessingutils import _get_initial_separation

        h5_file = h5py.File(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/%s.h5") % TestPostprocessingUtils.simulation_name,
            'r')
        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/parfiles/%s.par") % TestPostprocessingUtils.simulation_name
        actual_initial_separation = _get_initial_separation(h5_file)
        expected_initial_separation = 2.337285746
        self.assertTrue(np.isclose(expected_initial_separation, actual_initial_separation, atol=1e-3))
        h5_file.close()

        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        compact_object = temp_h5_file.create_group("compact_object")
        object_0_data = np.full((10, 52), np.nan)
        object_1_data = np.full((10, 52), np.nan)
        header = ["itt", "time",
                  "x", "y", "z",
                  "vx", "vy", "vz",
                  "ax", "ay", "az",
                  "Sx", "Sy", "Sz",
                  "Px", "Py", "Pz",
                  "min radius", "max radius", "mean radius",
                  "quadrupole_xx", "quadrupole_xy", "quadrupole_xz",
                  "quadrupole_yy", "quadrupole_yz", "quadrupole_zz",
                  "min x", "max x", "min y", "max y", "min z", "max z",
                  "xy-plane circumference", "xz-plane circumference", "yz-plane circumference",
                  "ratio of xz/xy-plane circumferences", "ratio of yz/xy-plane circumferences",
                  "area",
                  "m_irreducible", "areal radius", "expansion Theta_(l)",
                  "inner expansion Theta_(n)",
                  "product of the expansions", "mean curvature", "gradient of the areal radius",
                  "gradient of the expansion Theta_(l)",
                  "gradient of the inner expansion Theta_(n)",
                  "gradient of the product of the expansions", "gradient of the mean curvature",
                  "minimum  of the mean curvature", "maximum  of the mean curvature",
                  "integral of the mean curvature"]
        object_0 = compact_object.create_dataset("object=0", data=object_0_data)
        object_0.attrs["header"] = header
        object_1 = compact_object.create_dataset("object=1", data=object_1_data)
        object_1.attrs["header"] = header
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parfile) as f:
            par_content = f.read()
        parfile_group.attrs['par_content'] = par_content
        actual_initial_separation = _get_initial_separation(temp_h5_file)
        expected_initial_separation = 2.337285746
        self.assertTrue(np.isclose(expected_initial_separation, actual_initial_separation, atol=1e-4))
        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

    def test__get_initial_orbital_frequency(self):
        from mayawaves.utils.postprocessingutils import _get_initial_orbital_frequency

        h5_file = h5py.File(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/%s.h5") % TestPostprocessingUtils.simulation_name,
            'r')
        actual_initial_orbital_frequency = _get_initial_orbital_frequency(h5_file)
        expected_initial_orbital_frequency = 0.43
        self.assertTrue(np.isclose(expected_initial_orbital_frequency, actual_initial_orbital_frequency, atol=1e-2))

    def test__store_meta_data(self):
        from mayawaves.utils.postprocessingutils import _store_meta_data
        from mayawaves.utils.postprocessingutils import _stitch_timeseries_data
        from mayawaves.utils.postprocessingutils import _MiscDataFilenames

        original_h5_file = h5py.File(
            os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5"), 'r')
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        original_h5_file.copy('compact_object', temp_h5_file)

        relevant_data_filepaths = {
            "misc":
                {
                    _MiscDataFilenames.RUNSTATS:
                        {
                            "runstats.asc":
                                [
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0000/%s/runstats.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name),
                                    os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/%s/output-0001/%s/runstats.asc") % (
                                        TestPostprocessingUtils.simulation_name,
                                        TestPostprocessingUtils.simulation_name)
                                ]
                        }
                }
        }

        relevant_output_filepaths = {
            ".out":
                {
                    "%s.out" % (TestPostprocessingUtils.simulation_name):
                        [
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/%s/output-0000/%s.out") % (
                                TestPostprocessingUtils.simulation_name, TestPostprocessingUtils.simulation_name),
                            os.path.join(TestPostprocessingUtils.CURR_DIR,
                                         "resources/main_test_simulation/%s/output-0001/%s.out") % (
                                TestPostprocessingUtils.simulation_name, TestPostprocessingUtils.simulation_name),
                        ]
                }
        }
        parfile = os.path.join(TestPostprocessingUtils.CURR_DIR,
                               "resources/parfiles/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.par")
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parfile) as f:
            parfile_group.attrs['par_content'] = f.read()

        _store_meta_data(temp_h5_file, relevant_data_filepaths, relevant_output_filepaths,
                         TestPostprocessingUtils.simulation_name)

        expected_attributes = ["name",
                               "dimensional spin 0",
                               "dimensional spin 1",
                               "irreducible mass 0",
                               "irreducible mass 1",
                               "horizon mass 0",
                               "horizon mass 1",
                               "mass ratio",
                               "dimensionless spin 0",
                               "dimensionless spin 1",
                               "spin configuration",
                               "initial separation",
                               "initial orbital frequency"]
        expected_attributes.sort()
        generated_attributes = list(temp_h5_file.attrs.keys())
        generated_attributes.sort()
        self.assertTrue(np.all(expected_attributes == generated_attributes))
        for attribute in temp_h5_file.attrs.keys():
            if type(temp_h5_file.attrs[attribute]) != str:
                self.assertFalse(np.all(np.isnan(temp_h5_file.attrs[attribute])))
            else:
                self.assertTrue(temp_h5_file.attrs[attribute] != "" and temp_h5_file.attrs[attribute] is not None)

        self.assertTrue("runstats" in temp_h5_file)
        expected_header_array = ["iteration", "coord_time", "wall_time", "speed (hours^-1)", "period (minutes)",
                                 "cputime (cpu hours)"]
        self.assertTrue(np.all(expected_header_array == temp_h5_file["runstats"].attrs["header"]))
        expected_runstat_data = _stitch_timeseries_data(
            relevant_data_filepaths["misc"][_MiscDataFilenames.RUNSTATS]["runstats.asc"])
        self.assertTrue(np.all(expected_runstat_data == temp_h5_file["runstats"][()]))

        temp_h5_file.close()
        os.remove(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"))

        # test storing catalog id
        temp_h5_file = h5py.File(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5"), 'w')
        original_h5_file.copy('compact_object', temp_h5_file)
        parfile_group = temp_h5_file.create_group('parfile')
        with open(parfile) as f:
            parfile_group.attrs['par_content'] = f.read()
        _store_meta_data(temp_h5_file, relevant_data_filepaths, relevant_output_filepaths,
                         TestPostprocessingUtils.simulation_name, catalog_id='catalog0000')
        expected_attributes = ["name",
                               "catalog id",
                               "dimensional spin 0",
                               "dimensional spin 1",
                               "irreducible mass 0",
                               "irreducible mass 1",
                               "horizon mass 0",
                               "horizon mass 1",
                               "mass ratio",
                               "dimensionless spin 0",
                               "dimensionless spin 1",
                               "spin configuration",
                               "initial separation",
                               "initial orbital frequency"]
        expected_attributes.sort()
        generated_attributes = list(temp_h5_file.attrs.keys())
        generated_attributes.sort()
        self.assertEqual(expected_attributes, generated_attributes)
        self.assertEqual("catalog0000", temp_h5_file.attrs['catalog id'])

        original_h5_file.close()

    def test_create_h5_from_simulation(self):
        from mayawaves.utils.postprocessingutils import _Shifttracker, _Ihspin_hn, _BH_diagnostics
        from mayawaves.compactobject import CompactObject

        # if path doesn't exist
        h5_filename = pputils.create_h5_from_simulation('/not/a/directory', TestPostprocessingUtils.output_directory)
        self.assertFalse(
            os.path.isfile(os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output/directory.h5")))

        # trailing slash
        h5_filename = pputils.create_h5_from_simulation(TestPostprocessingUtils.raw_directory + '/',
                                                        TestPostprocessingUtils.output_directory)

        self.assertEqual(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                      "resources/test_output/%s.h5") % TestPostprocessingUtils.simulation_name,
                         h5_filename)
        self.assertTrue(os.path.isfile(h5_filename))
        os.remove(h5_filename)

        # raw unstitched simulation
        h5_filename = pputils.create_h5_from_simulation(TestPostprocessingUtils.raw_directory,
                                                        TestPostprocessingUtils.output_directory)

        self.assertTrue(os.path.isfile(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                    "resources/test_output/%s.h5") % TestPostprocessingUtils.simulation_name))
        print("test_create_h5_from_simualation -> h5 file is created")

        # open file to be tested
        file_to_test = h5py.File(h5_filename)

        # test rpar file is correct
        self.assertTrue('parfile' in file_to_test.keys())
        parfile_group = file_to_test['parfile']
        self.assertTrue('rpar_content' in parfile_group.attrs)
        expected_rpar_file = os.path.join(TestPostprocessingUtils.stitched_data_directory,
                                          "%s.rpar" % TestPostprocessingUtils.simulation_name)
        with open(expected_rpar_file) as f:
            expected_content = f.read()
        self.assertEqual(expected_content, parfile_group.attrs['rpar_content'])
        temp_parameter_file = os.path.join(TestPostprocessingUtils.CURR_DIR, 'resources/temp_parfile.rpar')
        with open(temp_parameter_file, 'w') as f:
            f.write(parfile_group.attrs['rpar_content'])
        self.assertTrue(filecmp.cmp(expected_rpar_file, temp_parameter_file))
        self.assertTrue('par_content' in parfile_group.attrs)
        os.remove(temp_parameter_file)

        print("test_create_h5_from_simualation -> rpar file was correctly stored")

        # test psi4 data
        for extraction_radius_group_name in file_to_test["radiative"]["psi4"]:
            extraction_radius = extraction_radius_group_name[7:]

            # modes
            for l_group_name in file_to_test["radiative"]["psi4"][extraction_radius_group_name]["modes"]:
                l_value = l_group_name[2:]
                for m_group_name in file_to_test["radiative"]["psi4"][extraction_radius_group_name]["modes"][
                    l_group_name]:
                    m_value = m_group_name[2:]

                    # reconstruct generated psi4 data
                    generated_psi4_time = file_to_test["radiative"]["psi4"][extraction_radius_group_name]["time"][()]
                    generated_psi4_real = \
                        file_to_test["radiative"]["psi4"][extraction_radius_group_name]["modes"][l_group_name][
                            m_group_name]["real"][()]
                    generated_psi4_imaginary = \
                        file_to_test["radiative"]["psi4"][extraction_radius_group_name]["modes"][l_group_name][
                            m_group_name]["imaginary"][()]
                    generated_psi4_data = np.empty((generated_psi4_time.shape[0], 3))
                    generated_psi4_data[:, 0] = generated_psi4_time
                    generated_psi4_data[:, 1] = generated_psi4_real
                    generated_psi4_data[:, 2] = generated_psi4_imaginary

                    # test monotonically increasing time
                    self.assertTrue(np.all(np.diff(generated_psi4_time) > 0))

                    # check consistent with previous stitching methods
                    expected_psi4_data = np.loadtxt(
                        os.path.join(TestPostprocessingUtils.stitched_data_directory,
                                     "Ylm_WEYLSCAL4::Psi4r_l%s_m%s_r%s.asc" % (l_value, m_value, extraction_radius)))

                    self.assertTrue(
                        (len(expected_psi4_data) - len(generated_psi4_data)) / len(expected_psi4_data) < 0.01)

                    common_times, indices_generated, indices_expected = np.intersect1d(generated_psi4_data[:, 0],
                                                                                       expected_psi4_data[:, 0],
                                                                                       assume_unique=True,
                                                                                       return_indices=True)
                    self.assertTrue(np.all(common_times == generated_psi4_data[:, 0]))
                    expected_psi4_data = expected_psi4_data[indices_expected]

                    self.assertTrue(np.all(generated_psi4_data == expected_psi4_data))

        print("test_create_h5_from_simualation -> psi4 data correctly stored")

        # test compact object data
        for compact_object_name in file_to_test["compact_object"].keys():
            compact_object_num = int(compact_object_name[7:])
            print("test_create_h5_from_simualation -> checking data for compact object %d" % (compact_object_num))
            generated_data = file_to_test["compact_object"][compact_object_name]
            header_list = generated_data.attrs["header"].tolist()

            # check shifttracker data
            if not os.path.isfile(os.path.join(TestPostprocessingUtils.stitched_data_directory,
                                               "ShiftTracker%d.asc" % compact_object_num)):
                self.assertFalse('vx' in header_list)
            else:
                # test monotonically increasing time
                shifttracker_column_idxs = [header_list.index(col.header_text) for col in _Shifttracker.column_list if
                                            col is not None]
                generated_shifttracker_data = generated_data[
                                                  ~np.isnan(generated_data[:, shifttracker_column_idxs[-1]])][:,
                                              shifttracker_column_idxs]
                self.assertTrue(np.all(np.diff(generated_shifttracker_data[:, 1]) > 0))

                # test consistency with previous stitch method
                expected_shifttracker_data = np.loadtxt(
                    os.path.join(TestPostprocessingUtils.stitched_data_directory,
                                 "ShiftTracker%d.asc" % compact_object_num))
                self.assertTrue(np.all(np.isclose(expected_shifttracker_data, generated_shifttracker_data, rtol=0.001)))

            print("test_create_h5_from_simualation -> shifttracker data correctly stored")

            # check ihspin_hn data
            if not os.path.isfile(os.path.join(TestPostprocessingUtils.stitched_data_directory,
                                               "ihspin_hn_%d.asc" % compact_object_num)):
                self.assertFalse('Sx' in header_list)
            else:
                # test monotonically increasing time
                ih_spin_column_idxs = [header_list.index(col.header_text) for col in _Ihspin_hn.column_list if
                                       col is not None]
                ih_spin_column_idxs.insert(0, header_list.index(CompactObject.Column.TIME.header_text))
                generated_ihspin_hn_data = generated_data[~np.isnan(generated_data[:, ih_spin_column_idxs[-1]])][:,
                                           ih_spin_column_idxs]

                self.assertTrue(np.all(np.diff(generated_ihspin_hn_data[:, 0]) > 0))

                # test consistency with previous stitch method
                expected_ihspin_hn_data = np.loadtxt(
                    os.path.join(TestPostprocessingUtils.stitched_data_directory,
                                 "ihspin_hn_%d.asc" % compact_object_num))
                self.assertTrue(np.all(np.isclose(expected_ihspin_hn_data, generated_ihspin_hn_data, rtol=0.001)))

            print("test_create_h5_from_simualation -> ihspin_hn data correctly stored")

            # check bh_diagnostics data
            if not os.path.isfile(
                    os.path.join(TestPostprocessingUtils.stitched_data_directory,
                                 "BH_diagnostics.ah%d.gp" % (compact_object_num + 1))):
                self.assertFalse('area' in header_list)
            else:
                # test monotonically increasing time
                bh_diagnostics_column_idxs = [header_list.index(col.header_text) for col in _BH_diagnostics.column_list
                                              if col is not None]
                generated_bh_diagnostics_data = generated_data[
                                                    ~np.isnan(generated_data[:, bh_diagnostics_column_idxs[-1]])][:,
                                                bh_diagnostics_column_idxs]
                self.assertTrue(np.all(np.diff(generated_bh_diagnostics_data[:, 1]) > 0))

                expected_bh_diagnostics_data = np.loadtxt(
                    os.path.join(TestPostprocessingUtils.stitched_data_directory,
                                 "BH_diagnostics.ah%d.gp" % (compact_object_num + 1)))
                # check info only in BH_diagnostics
                self.assertTrue(np.all(
                    np.isclose(expected_bh_diagnostics_data[:, 5:], generated_bh_diagnostics_data[:, 5:], atol=0.001)))
                # check position data
                self.assertTrue(np.all(
                    np.isclose(expected_bh_diagnostics_data[:, 2:5], generated_bh_diagnostics_data[:, 2:5], atol=0.05)))
                # check iteration column
                self.assertTrue(np.all(
                    np.isclose(expected_bh_diagnostics_data[:, 0], generated_bh_diagnostics_data[:, 0], rtol=0.001)))
                # check time column
                # the time column is rounded off in BH_diagnostics
                self.assertTrue(np.all(
                    np.isclose(expected_bh_diagnostics_data[:, 1], generated_bh_diagnostics_data[:, 1], atol=0.001)))

            print("test_create_h5_from_simualation -> bh_diagnostics data correctly stored")
            print(
                "test_create_h5_from_simualation -> data for compact object %d correctly stored" % (compact_object_num))
        print("test_create_h5_from_simualation -> data for compact objects correctly stored")

        # test runstats
        if "runstats" in file_to_test.keys():
            expected_runstats_data = np.loadtxt(
                os.path.join(TestPostprocessingUtils.stitched_data_directory, "runstats.asc"))
            generated_runstats_data = file_to_test["runstats"][()]
            self.assertTrue(np.all(expected_runstats_data == generated_runstats_data))

            # test monotonically increasing time
            self.assertTrue(np.all(np.diff(generated_runstats_data[:, 1]) > 0))

        print("test_create_h5_from_simualation -> runstats data correctly stored")

        # name
        self.assertEqual(file_to_test.attrs["name"], TestPostprocessingUtils.simulation_name)
        # catalog id is not stored
        self.assertFalse('catalog id' in file_to_test.attrs)
        # initial spins
        self.assertTrue(np.all(np.isclose(
            file_to_test.attrs["dimensional spin 0"], [0, 0, 0], atol=1e-3)))
        self.assertTrue(np.all(np.isclose(
            file_to_test.attrs["dimensional spin 1"], [0, 0, 0], atol=1e-3)))
        self.assertTrue(np.all(np.isclose(
            file_to_test.attrs["dimensionless spin 0"],
            [0, 0, 0], atol=1e-3)))
        self.assertTrue(np.all(np.isclose(
            file_to_test.attrs["dimensionless spin 1"],
            [0, 0, 0], atol=1e-3)))
        # spin configuration
        self.assertEqual(file_to_test.attrs["spin configuration"], "non-spinning")
        # initial masses
        self.assertTrue(np.isclose(file_to_test.attrs["irreducible mass 0"], 0.5167970311, atol=1e-3))
        self.assertTrue(np.isclose(file_to_test.attrs["irreducible mass 1"], 0.5167970311, atol=1e-3))
        self.assertTrue(np.isclose(file_to_test.attrs["horizon mass 0"], 0.5167970311, atol=1e-3))
        self.assertTrue(np.isclose(file_to_test.attrs["horizon mass 1"], 0.5167970311, atol=1e-3))
        # mass ratio
        self.assertTrue(np.isclose(file_to_test.attrs["mass ratio"], 1, atol=0.01))
        # initial separation
        self.assertTrue(np.isclose(file_to_test.attrs["initial separation"], 2.33, atol=0.01))
        # initial orbital frequency
        self.assertTrue(np.isclose(file_to_test.attrs["initial orbital frequency"], 0.43, atol=0.01))

        file_to_test.close()

        # test prestitched simulation

        stitched_h5_directory = TestPostprocessingUtils.output_directory + "/stitched_version"
        os.mkdir(stitched_h5_directory)
        stitched_h5_filename = pputils.create_h5_from_simulation(
            os.path.join(TestPostprocessingUtils.CURR_DIR,
                         "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"),
            stitched_h5_directory)

        raw_h5_version = h5py.File(h5_filename, 'r')
        stitched_h5_version = h5py.File(stitched_h5_filename, 'r')

        # compare attributes
        raw_attrs_keys = sorted(list(raw_h5_version.attrs.keys()))
        stitched_attrs_keys = sorted(list(stitched_h5_version.attrs.keys()))
        self.assertEqual(raw_attrs_keys, stitched_attrs_keys)

        for key in raw_attrs_keys:
            self.assertTrue(np.all(raw_h5_version.attrs[key] == stitched_h5_version.attrs[key]))

        # compare dictionaries
        def compare_h5_content(raw, stitched):
            raw_keys = sorted(list(raw.keys()))
            stitched_keys = sorted(list(stitched.keys()))
            self.assertTrue(np.all(raw_keys == stitched_keys))
            for key in raw_keys:
                if type(raw[key]) == h5py._hl.group.Group:
                    compare_h5_content(raw[key], stitched[key])
                else:
                    raw_data = raw[key][()]
                    stitched_data = stitched[key][()]

                    # compare location of nans
                    raw_not_nan_indices = ~np.isnan(raw_data)
                    stitched_non_nan_indices = ~np.isnan(stitched_data)
                    self.assertTrue(np.all(raw_not_nan_indices == stitched_non_nan_indices))

                    # compare all non nan data
                    raw_data = raw_data[raw_not_nan_indices]
                    stitched_data = stitched_data[raw_not_nan_indices]
                    self.assertTrue(np.all(raw_data == stitched_data))

        compare_h5_content(raw_h5_version, stitched_h5_version)
        os.remove(h5_filename)

        # test including a catalog id
        h5_filename = pputils.create_h5_from_simulation(TestPostprocessingUtils.raw_directory,
                                                        TestPostprocessingUtils.output_directory, 'catalog0000')
        file_to_test = h5py.File(h5_filename)
        self.assertEqual(os.path.join(TestPostprocessingUtils.output_directory, 'catalog0000.h5'), h5_filename)
        # name
        self.assertEqual(file_to_test.attrs["name"], TestPostprocessingUtils.simulation_name)
        # catalog id is stored
        self.assertTrue('catalog id' in file_to_test.attrs)
        self.assertEqual('catalog0000', file_to_test.attrs['catalog id'])

        os.remove(h5_filename)

    def test_get_stitched_data(self):
        raw_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                     "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")
        filename = "Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc"
        stitched_data = pputils.get_stitched_data(raw_directory=raw_directory, filename=filename)

        expected_stitched_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                  "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc")
        expected_stitched_data = np.loadtxt(expected_stitched_filepath)

        self.assertTrue(np.all(np.isclose(stitched_data, expected_stitched_data, atol=1e-4)))

        # test stitching one file
        raw_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                     "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_different_name")
        filename = "ShiftTracker0.asc"
        stitched_data = pputils.get_stitched_data(raw_directory=raw_directory, filename=filename)

        expected_stitched_data = np.loadtxt(os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                         "resources/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_different_name/output-0000/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67/ShiftTracker0.asc"))

        self.assertTrue(np.all(np.isclose(stitched_data, expected_stitched_data, atol=1e-4)))

        # directory doesn't exist
        raw_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/non_existing_directory")
        filename = "Ylm_WEYLSCAL4::Psi4r_l2_m2_r70.00.asc"
        stitched_data = pputils.get_stitched_data(raw_directory=raw_directory, filename=filename)
        self.assertTrue(stitched_data is None)

        # filename doesn't exist
        raw_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                     "resources/main_test_simulation/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")
        filename = "non_existing_filename.txt"
        stitched_data = pputils.get_stitched_data(raw_directory=raw_directory, filename=filename)
        self.assertTrue(stitched_data is None)

    def test__nr_frequency_to_physical(self):
        from mayawaves.utils.postprocessingutils import _nr_frequency_to_physical
        nr_frequency = 0.1
        mass = 100
        expected_physical_frequency = 203.025443517
        actual_physical_frequency = _nr_frequency_to_physical(nr_frequency, mass)
        self.assertTrue(np.isclose(expected_physical_frequency, actual_physical_frequency, rtol=0.01))

        nr_frequency = 0.2
        mass = 50
        expected_physical_frequency = 812.101774068
        actual_physical_frequency = _nr_frequency_to_physical(nr_frequency, mass)
        self.assertTrue(np.isclose(expected_physical_frequency, actual_physical_frequency, rtol=0.01))

        nr_frequency = 0.01
        mass = 100
        expected_physical_frequency = 20.3025443517
        actual_physical_frequency = _nr_frequency_to_physical(nr_frequency, mass)
        self.assertTrue(np.isclose(expected_physical_frequency, actual_physical_frequency, rtol=0.01))

    def test__crop_between_times(self):
        from mayawaves.utils.postprocessingutils import _crop_between_times

        # data with single column, merge time within time array, start time of 0
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 1)
        start_time = 0
        end_time = 7
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertTrue(np.all(time[:8] == time_cropped))
        self.assertTrue(np.all(data[:8] == data_cropped))

        # data with single column, merge time greater than last time, start time of 0
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 1)
        start_time = 0
        end_time = 12
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertTrue(np.all(time == time_cropped))
        self.assertTrue(np.all(data == data_cropped))

        # data with single column, merge time before first time, start time of 0
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 1)
        start_time = 0
        end_time = -3
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertEqual((0,), time_cropped.shape)
        self.assertEqual((0, 1), data_cropped.shape)

        # data with multiple columns, merge time within time array, start time of 0
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 3)
        start_time = 0
        end_time = 7
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertTrue(np.all(time[:8] == time_cropped))
        self.assertTrue(np.all(data[:8] == data_cropped))

        # data with multiple columns, merge time greater than last time, start time of 0
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 3)
        start_time = 0
        end_time = 12
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertTrue(np.all(time == time_cropped))
        self.assertTrue(np.all(data == data_cropped))

        # data with multiple columns, merge time before first time, start time of 0
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 3)
        start_time = 0
        end_time = -3
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertEqual((0,), time_cropped.shape)
        self.assertEqual((0, 3), data_cropped.shape)

        # data with multiple columns, merge time within time array, start time within time array
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 3)
        start_time = 2
        end_time = 7
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertTrue(np.all(time[2:8] == time_cropped))
        self.assertTrue(np.all(data[2:8] == data_cropped))

        # data with multiple columns, merge time greater than last time, start time within time array
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 3)
        start_time = 2
        end_time = 12
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertTrue(np.all(time[2:] == time_cropped))
        self.assertTrue(np.all(data[2:] == data_cropped))

        # data with multiple columns, merge time before first time, start time within time array
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 3)
        start_time = 2
        end_time = -3
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertEqual((0,), time_cropped.shape)
        self.assertEqual((0, 3), data_cropped.shape)

        # data with multiple columns, merge time greater than last time, start time greater than last time
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 3)
        start_time = 11
        end_time = 12
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertEqual((0,), time_cropped.shape)
        self.assertEqual((0, 3), data_cropped.shape)

        # data with multiple columns, merge time greater than last time, start time less than start time
        time = np.arange(0, 10, 1)
        data = np.random.randn(len(time), 3)
        start_time = -3
        end_time = 12
        time_cropped, data_cropped = _crop_between_times(time, data, start_time, end_time)
        self.assertTrue(np.all(time == time_cropped))
        self.assertTrue(np.all(data == data_cropped))

    def test__find_first_significant_gap_time(self):
        time = np.concatenate([np.linspace(0, 10, 100), np.linspace(20, 100, 100)])
        gap_time = pputils._find_first_significant_gap_time(time)
        self.assertEqual(10, gap_time)

        time = np.concatenate([np.linspace(0, 10, 100), np.linspace(11, 21, 100)])
        gap_time = pputils._find_first_significant_gap_time(time)
        self.assertEqual(10, gap_time)

        time = np.concatenate([np.linspace(0, 10, 100), np.linspace(10.1, 20.1, 100)])
        gap_time = pputils._find_first_significant_gap_time(time)
        self.assertEqual(time[-1], gap_time)

        time = np.concatenate([np.linspace(0, 15, 150), np.linspace(15.7, 30.7, 150)])
        gap_time = pputils._find_first_significant_gap_time(time)
        self.assertEqual(15, gap_time)

    def test__find_last_significant_gap(self):
        time = np.concatenate([np.linspace(0, 10, 100), np.linspace(20, 100, 100)])
        gap_time = pputils._find_last_significant_gap(time)
        self.assertEqual(20, gap_time)

        time = np.concatenate([np.linspace(0, 10, 100), np.linspace(11, 21, 100)])
        gap_time = pputils._find_last_significant_gap(time)
        self.assertEqual(11, gap_time)

        time = np.concatenate([np.linspace(0, 10, 100), np.linspace(10.1, 20.1, 100)])
        gap_time = pputils._find_last_significant_gap(time)
        self.assertEqual(0, gap_time)

        time = np.concatenate([np.linspace(0, 15, 150), np.linspace(15.7, 30.7, 150)])
        gap_time = pputils._find_last_significant_gap(time)
        self.assertEqual(15.7, gap_time)

    def test__store_compact_object_timeseries_data(self):
        from mayawaves.utils.postprocessingutils import _store_compact_object_timeseries_data

        initial_horizon_time = 75
        time_shift = 50

        # format 1
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/format_test/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_format_1.h5")
        lvc_format = 1
        coalescence = Coalescence(h5_filename)

        merge_time = coalescence.merge_time

        temp_lal_h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
        temp_lal_h5 = h5py.File(temp_lal_h5_filename, 'w')

        _store_compact_object_timeseries_data(coalescence, temp_lal_h5, lvc_format,
                                              initial_horizon_time=initial_horizon_time, time_shift=time_shift)

        expected_groups = set()
        actual_groups = set(temp_lal_h5.keys())

        self.assertEqual(expected_groups, actual_groups)

        coalescence.close()
        temp_lal_h5.close()
        os.remove(temp_lal_h5_filename)

        # format 2
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/format_test/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_format_2.h5")
        lvc_format = 2
        coalescence = Coalescence(h5_filename)

        temp_lal_h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
        temp_lal_h5 = h5py.File(temp_lal_h5_filename, 'w')

        _store_compact_object_timeseries_data(coalescence, temp_lal_h5, lvc_format,
                                              initial_horizon_time=initial_horizon_time, time_shift=time_shift)

        expected_groups = {'mass1-vs-time', 'mass2-vs-time', 'spin1x-vs-time', 'spin1y-vs-time', 'spin1z-vs-time',
                           'spin2x-vs-time', 'spin2y-vs-time', 'spin2z-vs-time', 'position1x-vs-time',
                           'position1y-vs-time', 'position1z-vs-time', 'position2x-vs-time', 'position2y-vs-time',
                           'position2z-vs-time', 'LNhatx-vs-time', 'LNhaty-vs-time', 'LNhatz-vs-time', 'Omega-vs-time'}
        actual_groups = set(temp_lal_h5.keys())

        self.assertEqual(expected_groups, actual_groups)

        primary_object = coalescence.primary_compact_object

        time, expected_mass1 = primary_object.horizon_mass
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_mass1 = expected_mass1[start_index:merge_index]

        spline_mass1 = romspline.readSpline(temp_lal_h5_filename, 'mass1-vs-time')
        generated_mass1 = spline_mass1(time)
        self.assertTrue(np.allclose(expected_mass1, generated_mass1, atol=1e-4))

        time, expected_spin1 = primary_object.dimensionless_spin_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_spin1 = expected_spin1[start_index:merge_index]
        expected_spin1x = expected_spin1[:, 0]
        expected_spin1y = expected_spin1[:, 1]
        expected_spin1z = expected_spin1[:, 2]

        spline_spin1x = romspline.readSpline(temp_lal_h5_filename, 'spin1x-vs-time')
        generated_spin1x = spline_spin1x(time)
        self.assertTrue(np.allclose(expected_spin1x, generated_spin1x, atol=1e-4))

        spline_spin1y = romspline.readSpline(temp_lal_h5_filename, 'spin1y-vs-time')
        generated_spin1y = spline_spin1y(time)
        self.assertTrue(np.allclose(expected_spin1y, generated_spin1y, atol=1e-4))

        spline_spin1z = romspline.readSpline(temp_lal_h5_filename, 'spin1z-vs-time')
        generated_spin1z = spline_spin1z(time)
        self.assertTrue(np.allclose(expected_spin1z, generated_spin1z, atol=1e-4))

        time, expected_position1 = primary_object.position_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_position1 = expected_position1[start_index:merge_index]
        expected_position1x = expected_position1[:, 0]
        expected_position1y = expected_position1[:, 1]
        expected_position1z = expected_position1[:, 2]

        spline_position1x = romspline.readSpline(temp_lal_h5_filename, 'position1x-vs-time')
        generated_position1x = spline_position1x(time)
        self.assertTrue(np.allclose(expected_position1x, generated_position1x, atol=1e-4))

        spline_position1y = romspline.readSpline(temp_lal_h5_filename, 'position1y-vs-time')
        generated_position1y = spline_position1y(time)
        self.assertTrue(np.allclose(expected_position1y, generated_position1y, atol=1e-4))

        spline_position1z = romspline.readSpline(temp_lal_h5_filename, 'position1z-vs-time')
        generated_position1z = spline_position1z(time)
        self.assertTrue(np.allclose(expected_position1z, generated_position1z, atol=1e-4))

        secondary_object = coalescence.secondary_compact_object

        time, expected_mass2 = secondary_object.horizon_mass
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_mass2 = expected_mass2[start_index:merge_index]

        spline_mass2 = romspline.readSpline(temp_lal_h5_filename, 'mass2-vs-time')
        generated_mass2 = spline_mass2(time)
        self.assertTrue(np.allclose(expected_mass2, generated_mass2, atol=1e-4))

        time, expected_spin2 = secondary_object.dimensionless_spin_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_spin2 = expected_spin2[start_index:merge_index]
        expected_spin2x = expected_spin2[:, 0]
        expected_spin2y = expected_spin2[:, 1]
        expected_spin2z = expected_spin2[:, 2]

        spline_spin2x = romspline.readSpline(temp_lal_h5_filename, 'spin2x-vs-time')
        generated_spin2x = spline_spin2x(time)
        self.assertTrue(np.allclose(expected_spin2x, generated_spin2x, atol=1e-4))

        spline_spin2y = romspline.readSpline(temp_lal_h5_filename, 'spin2y-vs-time')
        generated_spin2y = spline_spin2y(time)
        self.assertTrue(np.allclose(expected_spin2y, generated_spin2y, atol=1e-4))

        spline_spin2z = romspline.readSpline(temp_lal_h5_filename, 'spin2z-vs-time')
        generated_spin2z = spline_spin2z(time)
        self.assertTrue(np.allclose(expected_spin2z, generated_spin2z, atol=1e-4))

        time, expected_position2 = secondary_object.position_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_position2 = expected_position2[start_index:merge_index]
        expected_position2x = expected_position2[:, 0]
        expected_position2y = expected_position2[:, 1]
        expected_position2z = expected_position2[:, 2]

        spline_position2x = romspline.readSpline(temp_lal_h5_filename, 'position2x-vs-time')
        generated_position2x = spline_position2x(time)
        self.assertTrue(np.allclose(expected_position2x, generated_position2x, atol=1e-4))

        spline_position2y = romspline.readSpline(temp_lal_h5_filename, 'position2y-vs-time')
        generated_position2y = spline_position2y(time)
        self.assertTrue(np.allclose(expected_position2y, generated_position2y, atol=1e-4))

        spline_position2z = romspline.readSpline(temp_lal_h5_filename, 'position2z-vs-time')
        generated_position2z = spline_position2z(time)
        self.assertTrue(np.allclose(expected_position2z, generated_position2z, atol=1e-4))

        time, expected_lhat = coalescence.orbital_angular_momentum_unit_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_lhat = expected_lhat[start_index:merge_index]
        expected_lhatx = expected_lhat[:, 0]
        expected_lhaty = expected_lhat[:, 1]
        expected_lhatz = expected_lhat[:, 2]

        spline_lhatx = romspline.readSpline(temp_lal_h5_filename, 'LNhatx-vs-time')
        generated_lhatx = spline_lhatx(time)
        self.assertTrue(np.allclose(expected_lhatx, generated_lhatx, atol=1e-4))

        spline_lhaty = romspline.readSpline(temp_lal_h5_filename, 'LNhaty-vs-time')
        generated_lhaty = spline_lhaty(time)
        self.assertTrue(np.allclose(expected_lhaty, generated_lhaty, atol=1e-4))

        spline_lhatz = romspline.readSpline(temp_lal_h5_filename, 'LNhatz-vs-time')
        generated_lhatz = spline_lhatz(time)
        self.assertTrue(np.allclose(expected_lhatz, generated_lhatz, atol=1e-4))

        time, expected_omega = coalescence.orbital_frequency
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_omega = expected_omega[start_index:merge_index]
        spline_omega = romspline.readSpline(temp_lal_h5_filename, 'Omega-vs-time')
        generated_omega = spline_omega(time)
        self.assertTrue(np.allclose(expected_omega, generated_omega, atol=1e-4))

        coalescence.close()
        temp_lal_h5.close()
        os.remove(temp_lal_h5_filename)

        # format 3
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/format_test/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_format_3.h5")
        lvc_format = 3
        coalescence = Coalescence(h5_filename)

        temp_lal_h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
        temp_lal_h5 = h5py.File(temp_lal_h5_filename, 'w')

        _store_compact_object_timeseries_data(coalescence, temp_lal_h5, lvc_format,
                                              initial_horizon_time=initial_horizon_time, time_shift=time_shift)

        expected_groups = {'mass1-vs-time', 'mass2-vs-time', 'spin1x-vs-time', 'spin1y-vs-time', 'spin1z-vs-time',
                           'spin2x-vs-time', 'spin2y-vs-time', 'spin2z-vs-time', 'position1x-vs-time',
                           'position1y-vs-time', 'position1z-vs-time', 'position2x-vs-time', 'position2y-vs-time',
                           'position2z-vs-time', 'LNhatx-vs-time', 'LNhaty-vs-time', 'LNhatz-vs-time', 'Omega-vs-time',
                           'remnant-mass-vs-time', 'remnant-spinx-vs-time', 'remnant-spiny-vs-time',
                           'remnant-spinz-vs-time', 'remnant-positionx-vs-time', 'remnant-positiony-vs-time',
                           'remnant-positionz-vs-time'}
        actual_groups = set(temp_lal_h5.keys())

        self.assertEqual(expected_groups, actual_groups)

        primary_object = coalescence.primary_compact_object

        time, expected_mass1 = primary_object.horizon_mass
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_mass1 = expected_mass1[start_index:merge_index]

        spline_mass1 = romspline.readSpline(temp_lal_h5_filename, 'mass1-vs-time')
        generated_mass1 = spline_mass1(time)
        self.assertTrue(np.allclose(expected_mass1, generated_mass1, atol=1e-4))

        time, expected_spin1 = primary_object.dimensionless_spin_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - 50
        expected_spin1 = expected_spin1[start_index:merge_index]
        expected_spin1x = expected_spin1[:, 0]
        expected_spin1y = expected_spin1[:, 1]
        expected_spin1z = expected_spin1[:, 2]

        spline_spin1x = romspline.readSpline(temp_lal_h5_filename, 'spin1x-vs-time')
        generated_spin1x = spline_spin1x(time)
        self.assertTrue(np.allclose(expected_spin1x, generated_spin1x, atol=1e-4))

        spline_spin1y = romspline.readSpline(temp_lal_h5_filename, 'spin1y-vs-time')
        generated_spin1y = spline_spin1y(time)
        self.assertTrue(np.allclose(expected_spin1y, generated_spin1y, atol=1e-4))

        spline_spin1z = romspline.readSpline(temp_lal_h5_filename, 'spin1z-vs-time')
        generated_spin1z = spline_spin1z(time)
        self.assertTrue(np.allclose(expected_spin1z, generated_spin1z, atol=1e-4))

        time, expected_position1 = primary_object.position_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_position1 = expected_position1[start_index:merge_index]
        expected_position1x = expected_position1[:, 0]
        expected_position1y = expected_position1[:, 1]
        expected_position1z = expected_position1[:, 2]

        spline_position1x = romspline.readSpline(temp_lal_h5_filename, 'position1x-vs-time')
        generated_position1x = spline_position1x(time)
        self.assertTrue(np.allclose(expected_position1x, generated_position1x, atol=1e-4))

        spline_position1y = romspline.readSpline(temp_lal_h5_filename, 'position1y-vs-time')
        generated_position1y = spline_position1y(time)
        self.assertTrue(np.allclose(expected_position1y, generated_position1y, atol=1e-4))

        spline_position1z = romspline.readSpline(temp_lal_h5_filename, 'position1z-vs-time')
        generated_position1z = spline_position1z(time)
        self.assertTrue(np.allclose(expected_position1z, generated_position1z, atol=1e-4))

        secondary_object = coalescence.secondary_compact_object

        time, expected_mass2 = secondary_object.horizon_mass
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_mass2 = expected_mass2[start_index:merge_index]

        spline_mass2 = romspline.readSpline(temp_lal_h5_filename, 'mass2-vs-time')
        generated_mass2 = spline_mass2(time)
        self.assertTrue(np.allclose(expected_mass2, generated_mass2, atol=1e-4))

        time, expected_spin2 = secondary_object.dimensionless_spin_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_spin2 = expected_spin2[start_index:merge_index]
        expected_spin2x = expected_spin2[:, 0]
        expected_spin2y = expected_spin2[:, 1]
        expected_spin2z = expected_spin2[:, 2]

        spline_spin2x = romspline.readSpline(temp_lal_h5_filename, 'spin2x-vs-time')
        generated_spin2x = spline_spin2x(time)
        self.assertTrue(np.allclose(expected_spin2x, generated_spin2x, atol=1e-4))

        spline_spin2y = romspline.readSpline(temp_lal_h5_filename, 'spin2y-vs-time')
        generated_spin2y = spline_spin2y(time)
        self.assertTrue(np.allclose(expected_spin2y, generated_spin2y, atol=1e-4))

        spline_spin2z = romspline.readSpline(temp_lal_h5_filename, 'spin2z-vs-time')
        generated_spin2z = spline_spin2z(time)
        self.assertTrue(np.allclose(expected_spin2z, generated_spin2z, atol=1e-4))

        time, expected_position2 = secondary_object.position_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_position2 = expected_position2[start_index:merge_index]
        expected_position2x = expected_position2[:, 0]
        expected_position2y = expected_position2[:, 1]
        expected_position2z = expected_position2[:, 2]

        spline_position2x = romspline.readSpline(temp_lal_h5_filename, 'position2x-vs-time')
        generated_position2x = spline_position2x(time)
        self.assertTrue(np.allclose(expected_position2x, generated_position2x, atol=1e-4))

        spline_position2y = romspline.readSpline(temp_lal_h5_filename, 'position2y-vs-time')
        generated_position2y = spline_position2y(time)
        self.assertTrue(np.allclose(expected_position2y, generated_position2y, atol=1e-4))

        spline_position2z = romspline.readSpline(temp_lal_h5_filename, 'position2z-vs-time')
        generated_position2z = spline_position2z(time)
        self.assertTrue(np.allclose(expected_position2z, generated_position2z, atol=1e-4))

        time, expected_lhat = coalescence.orbital_angular_momentum_unit_vector
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_lhat = expected_lhat[start_index:merge_index]
        expected_lhatx = expected_lhat[:, 0]
        expected_lhaty = expected_lhat[:, 1]
        expected_lhatz = expected_lhat[:, 2]

        spline_lhatx = romspline.readSpline(temp_lal_h5_filename, 'LNhatx-vs-time')
        generated_lhatx = spline_lhatx(time)
        self.assertTrue(np.allclose(expected_lhatx, generated_lhatx, atol=1e-4))

        spline_lhaty = romspline.readSpline(temp_lal_h5_filename, 'LNhaty-vs-time')
        generated_lhaty = spline_lhaty(time)
        self.assertTrue(np.allclose(expected_lhaty, generated_lhaty, atol=1e-4))

        spline_lhatz = romspline.readSpline(temp_lal_h5_filename, 'LNhatz-vs-time')
        generated_lhatz = spline_lhatz(time)
        self.assertTrue(np.allclose(expected_lhatz, generated_lhatz, atol=1e-4))

        time, expected_omega = coalescence.orbital_frequency
        merge_index = np.argmax(time > merge_time)
        start_index = np.argmax(time >= initial_horizon_time)
        time = time[start_index:merge_index] - time_shift
        expected_omega = expected_omega[start_index:merge_index]
        spline_omega = romspline.readSpline(temp_lal_h5_filename, 'Omega-vs-time')
        generated_omega = spline_omega(time)
        self.assertTrue(np.allclose(expected_omega, generated_omega, atol=1e-4))

        remnant_object = coalescence.final_compact_object

        time, expected_remnant_mass = remnant_object.horizon_mass
        merge_index = np.argmax(time > merge_time)
        time = time[:merge_index] - time_shift
        expected_remnant_mass = expected_remnant_mass[:merge_index]

        spline_remnant_mass = romspline.readSpline(temp_lal_h5_filename, 'remnant-mass-vs-time')
        generated_remnant_mass = spline_remnant_mass(time)
        self.assertTrue(np.allclose(expected_remnant_mass, generated_remnant_mass, atol=1e-4))

        time, expected_remnant_spin = remnant_object.dimensionless_spin_vector
        merge_index = np.argmax(time > merge_time)
        time = time[:merge_index] - time_shift
        expected_remnant_spin = expected_remnant_spin[:merge_index]
        expected_remnant_spinx = expected_remnant_spin[:, 0]
        expected_remnant_spiny = expected_remnant_spin[:, 1]
        expected_remnant_spinz = expected_remnant_spin[:, 2]

        spline_remnant_spinx = romspline.readSpline(temp_lal_h5_filename, 'remnant-spinx-vs-time')
        generated_remnant_spinx = spline_remnant_spinx(time)
        self.assertTrue(np.allclose(expected_remnant_spinx, generated_remnant_spinx, atol=1e-4))

        spline_remnant_spiny = romspline.readSpline(temp_lal_h5_filename, 'remnant-spiny-vs-time')
        generated_remnant_spiny = spline_remnant_spiny(time)
        self.assertTrue(np.allclose(expected_remnant_spiny, generated_remnant_spiny, atol=1e-4))

        spline_remnant_spinz = romspline.readSpline(temp_lal_h5_filename, 'remnant-spinz-vs-time')
        generated_remnant_spinz = spline_remnant_spinz(time)
        self.assertTrue(np.allclose(expected_remnant_spinz, generated_remnant_spinz, atol=1e-4))

        time, expected_remnant_position = remnant_object.position_vector
        merge_index = np.argmax(time > merge_time)
        time = time[:merge_index] - time_shift
        expected_remnant_position = expected_remnant_position[:merge_index]
        expected_remnant_positionx = expected_remnant_position[:, 0]
        expected_remnant_positiony = expected_remnant_position[:, 1]
        expected_remnant_positionz = expected_remnant_position[:, 2]

        spline_remnant_positionx = romspline.readSpline(temp_lal_h5_filename, 'remnant-positionx-vs-time')
        generated_remnant_positionx = spline_remnant_positionx(time)
        self.assertTrue(np.allclose(expected_remnant_positionx, generated_remnant_positionx, atol=1e-4))

        spline_remnant_positiony = romspline.readSpline(temp_lal_h5_filename, 'remnant-positiony-vs-time')
        generated_remnant_positiony = spline_remnant_positiony(time)
        self.assertTrue(np.allclose(expected_remnant_positiony, generated_remnant_positiony, atol=1e-4))

        spline_remnant_positionz = romspline.readSpline(temp_lal_h5_filename, 'remnant-positionz-vs-time')
        generated_remnant_positionz = spline_remnant_positionz(time)
        self.assertTrue(np.allclose(expected_remnant_positionz, generated_remnant_positionz, atol=1e-4))

        coalescence.close()
        temp_lal_h5.close()
        os.remove(temp_lal_h5_filename)

    def test__store_lal_metadata(self):
        from mayawaves.utils.postprocessingutils import _store_lal_metadata

        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)

        temp_h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
        temp_lal_h5 = h5py.File(temp_h5_filename, 'w')
        name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        alternative_names = []
        initial_time_horizon = 75
        omega_22_nr = 0.2

        lvc_format = 1

        _store_lal_metadata(coalescence, temp_lal_h5, name, alternative_names,
                            initial_time_horizon, omega_22_nr, lvc_format, NR_group='UT Austin', NR_code='MAYA',
                            bibtex_keys='Jani:2016wkt', contact_email='deirdre.shoemaker@austin.utexas.edu')

        expected_attributes = {"Format", "type", "name", "alternative-names", "NR-group", "NR-code",
                               "modification-date", "point-of-contact-email", "simulation-type", "INSPIRE-bibtex-keys",
                               "license", "Lmax", "NR-techniques", "files-in-error-series", "comparable-simulation",
                               "production-run", "object1", "object2", "mass1", "mass2", "eta", "f_lower_at_1MSUN",
                               "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "LNhatx", "LNhaty", "LNhatz",
                               "nhatx", "nhaty", "nhatz", "Omega", "eccentricity", "mean_anomaly", "Warning",
                               "PN_approximant"}
        actual_attributes = set(temp_lal_h5.attrs)

        self.assertEqual(expected_attributes, actual_attributes)

        self.assertTrue(temp_lal_h5.attrs["Format"] in [1, 2, 3])
        self.assertEqual('NRinjection', temp_lal_h5.attrs["type"])
        self.assertEqual(name, temp_lal_h5.attrs["name"])
        self.assertEqual(alternative_names, list(temp_lal_h5.attrs["alternative-names"]))
        self.assertEqual("UT Austin", temp_lal_h5.attrs["NR-group"])
        self.assertEqual("MAYA", temp_lal_h5.attrs["NR-code"])
        self.assertEqual(date.today().strftime("%Y-%m-%d"), temp_lal_h5.attrs["modification-date"])

        if temp_lal_h5.attrs["spin1x"] == 0 and temp_lal_h5.attrs["spin1y"] == 0 and temp_lal_h5.attrs[
            "spin2x"] == 0 and temp_lal_h5.attrs["spin2y"] == 0:
            if temp_lal_h5.attrs["spin1z"] == 0 and temp_lal_h5.attrs["spin2z"] == 0:
                self.assertEqual("non-spinning", temp_lal_h5.attrs["simulation-type"])
            else:
                self.assertEqual("aligned-spins", temp_lal_h5.attrs["simulation-type"])
        else:
            self.assertEqual("precessing", temp_lal_h5.attrs["simulation-type"])

        self.assertTrue(temp_lal_h5.attrs["license"] in ["LVC-internal", "public"])
        self.assertTrue(temp_lal_h5.attrs["Lmax"] >= 2)

        self.assertTrue(len(temp_lal_h5.attrs["NR-techniques"].split(',')) == 6)

        self.assertTrue(temp_lal_h5.attrs["production-run"] in [0, 1])

        self.assertTrue(temp_lal_h5.attrs["object1"] in ["BH", "NS"])
        self.assertTrue(temp_lal_h5.attrs["object2"] in ["BH", "NS"])

        self.assertTrue('frame' not in temp_lal_h5['auxiliary-info'].attrs)

        coalescence.close()
        temp_lal_h5.close()
        os.remove(temp_h5_filename)

        # test setting frame
        # com corrected frame
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor <= 10:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)

            temp_h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
            temp_lal_h5 = h5py.File(temp_h5_filename, 'w')
            name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
            alternative_names = []
            initial_time_horizon = 75
            omega_22_nr = 0.2

            lvc_format = 1

            coalescence.set_radiation_frame(center_of_mass_corrected=True)

            _store_lal_metadata(coalescence, temp_lal_h5, name, alternative_names,
                                initial_time_horizon, omega_22_nr, lvc_format, NR_group='UT Austin', NR_code='MAYA',
                                bibtex_keys='Jani:2016wkt', contact_email='deirdre.shoemaker@austin.utexas.edu')

            self.assertTrue('frame' in temp_lal_h5['auxiliary-info'].attrs)
            self.assertEqual('Center of mass drift corrected', temp_lal_h5['auxiliary-info'].attrs['frame'])

            coalescence.close()
            temp_lal_h5.close()
            os.remove(temp_h5_filename)

            # failed com corrected frame
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)

            temp_h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
            temp_lal_h5 = h5py.File(temp_h5_filename, 'w')
            name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
            alternative_names = []
            initial_time_horizon = 75
            omega_22_nr = 0.2

            lvc_format = 1

            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(None, None)) as mock_center_of_mass:
                coalescence.set_radiation_frame(center_of_mass_corrected=True)
                _store_lal_metadata(coalescence, temp_lal_h5, name, alternative_names,
                                    initial_time_horizon, omega_22_nr, lvc_format, NR_group='UT Austin', NR_code='MAYA',
                                    bibtex_keys='Jani:2016wkt', contact_email='deirdre.shoemaker@austin.utexas.edu', )

            self.assertTrue('frame' not in temp_lal_h5['auxiliary-info'].attrs)

            coalescence.close()
            temp_lal_h5.close()
            os.remove(temp_h5_filename)

        # change metadata

        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)

        temp_h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
        temp_lal_h5 = h5py.File(temp_h5_filename, 'w')
        name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        alternative_names = []
        initial_time_horizon = 75
        omega_22_nr = 0.2

        lvc_format = 1

        _store_lal_metadata(coalescence, temp_lal_h5, name, alternative_names,
                            initial_time_horizon, omega_22_nr, lvc_format, NR_group='UT Austin', NR_code='MAYA',
                            bibtex_keys='Jani:2016wkt', contact_email='deirdre.shoemaker@austin.utexas.edu',
                            license_type='public', nr_techniques='some techniques', comparable_simulation='sim',
                            files_in_error_series='sim1, sim2')

        expected_attributes = {"Format", "type", "name", "alternative-names", "NR-group", "NR-code",
                               "modification-date", "point-of-contact-email", "simulation-type", "INSPIRE-bibtex-keys",
                               "license", "Lmax", "NR-techniques", "files-in-error-series", "comparable-simulation",
                               "production-run", "object1", "object2", "mass1", "mass2", "eta", "f_lower_at_1MSUN",
                               "spin1x", "spin1y", "spin1z", "spin2x", "spin2y", "spin2z", "LNhatx", "LNhaty", "LNhatz",
                               "nhatx", "nhaty", "nhatz", "Omega", "eccentricity", "mean_anomaly", "Warning",
                               "PN_approximant"}
        actual_attributes = set(temp_lal_h5.attrs)

        self.assertEqual(expected_attributes, actual_attributes)

        self.assertTrue(temp_lal_h5.attrs["Format"] in [1, 2, 3])
        self.assertEqual('NRinjection', temp_lal_h5.attrs["type"])
        self.assertEqual(name, temp_lal_h5.attrs["name"])
        self.assertEqual(alternative_names, list(temp_lal_h5.attrs["alternative-names"]))
        self.assertEqual("UT Austin", temp_lal_h5.attrs["NR-group"])
        self.assertEqual("MAYA", temp_lal_h5.attrs["NR-code"])
        self.assertEqual("public", temp_lal_h5.attrs["license"])
        self.assertEqual("some techniques", temp_lal_h5.attrs["NR-techniques"])
        self.assertEqual("sim", temp_lal_h5.attrs["comparable-simulation"])
        self.assertEqual("sim1, sim2", temp_lal_h5.attrs["files-in-error-series"])
        self.assertEqual(date.today().strftime("%Y-%m-%d"), temp_lal_h5.attrs["modification-date"])

        if temp_lal_h5.attrs["spin1x"] == 0 and temp_lal_h5.attrs["spin1y"] == 0 and temp_lal_h5.attrs[
            "spin2x"] == 0 and temp_lal_h5.attrs["spin2y"] == 0:
            if temp_lal_h5.attrs["spin1z"] == 0 and temp_lal_h5.attrs["spin2z"] == 0:
                self.assertEqual("non-spinning", temp_lal_h5.attrs["simulation-type"])
            else:
                self.assertEqual("aligned-spins", temp_lal_h5.attrs["simulation-type"])
        else:
            self.assertEqual("precessing", temp_lal_h5.attrs["simulation-type"])

        self.assertTrue(temp_lal_h5.attrs["license"] in ["LVC-internal", "public"])
        self.assertTrue(temp_lal_h5.attrs["Lmax"] >= 2)

        self.assertTrue(temp_lal_h5.attrs["production-run"] in [0, 1])

        self.assertTrue(temp_lal_h5.attrs["object1"] in ["BH", "NS"])
        self.assertTrue(temp_lal_h5.attrs["object2"] in ["BH", "NS"])

        self.assertTrue('frame' not in temp_lal_h5['auxiliary-info'].attrs)

        coalescence.close()
        temp_lal_h5.close()
        os.remove(temp_h5_filename)

    def test__get_max_time_all_strain_modes(self):
        from mayawaves.utils.postprocessingutils import _get_max_time_all_strain_modes
        strain_modes = {}
        time = np.arange(-1000, 1000, 1)
        amp = 1e8 - 20 * (time - 5) * (time - 5)
        strain_modes[0] = (time, amp)
        expected_max_time = 5
        actual_max_time = _get_max_time_all_strain_modes(strain_modes)
        self.assertTrue(np.isclose(expected_max_time, actual_max_time, rtol=1e-4))

        strain_modes = {}
        time = np.arange(-1000, 1000, 1)
        amp = 1e8 - 20 * (time - 10) * (time - 10)
        strain_modes[0] = (time, amp)
        expected_max_time = 10
        actual_max_time = _get_max_time_all_strain_modes(strain_modes)
        self.assertTrue(np.isclose(expected_max_time, actual_max_time, rtol=1e-4))

        strain_modes = {}
        time = np.arange(-1000, 1000, 1)
        amp = 1e8 - 20 * (time - 5) * (time - 5)
        strain_modes[0] = (time, amp)
        amp = 1e8 - 18 * (time - 10) * (time - 10)
        strain_modes[1] = (time, amp)
        expected_max_time = 7.4
        actual_max_time = _get_max_time_all_strain_modes(strain_modes)
        self.assertTrue(np.isclose(expected_max_time, actual_max_time, atol=1))

    def test__get_omega_at_time(self):
        from mayawaves.utils.postprocessingutils import _get_omega_at_time

        time = np.arange(0, 1000, 1)
        phase = -5 * time
        initial_time = 500
        expected_omega = 5
        actual_omega = _get_omega_at_time(time, phase, initial_time)
        self.assertEqual(expected_omega, actual_omega)

        time = np.arange(0, 1000, 1)
        phase = -5 * time * time
        initial_time = 100
        expected_omega = 10 * (initial_time + 1)
        actual_omega = _get_omega_at_time(time, phase, initial_time)
        self.assertEqual(expected_omega, actual_omega)

        time = np.arange(0, 1000, 1)
        phase = -5 * time * time
        initial_time = 600
        expected_omega = 10 * (initial_time + 1)
        actual_omega = _get_omega_at_time(time, phase, initial_time)
        self.assertEqual(expected_omega, actual_omega)

        time = np.arange(0, 1000, 1)
        phase = -5 * np.power(time, 3)
        initial_time = 100
        expected_omega = 15 * (initial_time + 1) ** 2
        actual_omega = _get_omega_at_time(time, phase, initial_time)
        self.assertTrue(np.isclose(expected_omega, actual_omega, rtol=1e-4))

    def test_low_pass_filter(self):
        t = np.linspace(0, 10000, 1000)
        low_freq = np.sin(0.01 * t)
        high_freq = 0.1 * np.sin(1 * t)
        signal = low_freq + high_freq
        filtered = pputils.low_pass_filter(t, signal, low_pass_freq_cutoff=0.1)
        self.assertTrue(np.allclose(low_freq[10:-10], filtered[10:-10], rtol=1e-3))

        t = np.linspace(0, 10000, 1000)
        low_freq = np.sin(0.01 * t)
        high_freq = 0.1 * np.sin(1 * t)
        signal = low_freq + high_freq
        filtered = pputils.low_pass_filter(t, signal, low_pass_freq_cutoff=0.05)
        self.assertTrue(np.allclose(low_freq[30:-30], filtered[30:-30], rtol=1e-3))

    def test_determine_lvc_format(self):
        from mayawaves.utils.postprocessingutils import determine_lvc_format

        initial_horizon_time = 75

        # format 1
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/format_test/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_format_1.h5")
        expected_format = 1
        coalescence = Coalescence(h5_filename)
        lvc_format = determine_lvc_format(coalescence, initial_horizon_time=initial_horizon_time)
        self.assertEqual(expected_format, lvc_format)

        # format 2
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/format_test/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_format_2.h5")
        expected_format = 2
        coalescence = Coalescence(h5_filename)
        lvc_format = determine_lvc_format(coalescence, initial_horizon_time=initial_horizon_time)
        self.assertEqual(expected_format, lvc_format)

        # format 3
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/format_test/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_format_3.h5")
        expected_format = 3
        coalescence = Coalescence(h5_filename)
        lvc_format = determine_lvc_format(coalescence, initial_horizon_time=initial_horizon_time)
        self.assertEqual(expected_format, lvc_format)

    def test__put_data_in_lal_compatible_format(self):
        from mayawaves.utils.postprocessingutils import _put_data_in_lal_compatible_format

        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)

        temp_lal_h5_file_name = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
        name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
        alternative_names = []
        extraction_radius = 70
        _put_data_in_lal_compatible_format(coalescence, temp_lal_h5_file_name, name, alternative_names,
                                           extraction_radius, NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                           contact_email='deirdre.shoemaker@austin.utexas.edu')

        temp_lal_h5 = h5py.File(temp_lal_h5_file_name, 'r')

        format = temp_lal_h5.attrs["Format"]
        self.assertTrue(format in [1, 2, 3])

        if format == 2 or format == 3:
            self.assertTrue("mass1-vs-time" in temp_lal_h5)
            self.assertTrue("mass2-vs-time" in temp_lal_h5)
            self.assertTrue("spin1x-vs-time" in temp_lal_h5)
            self.assertTrue("spin1y-vs-time" in temp_lal_h5)
            self.assertTrue("spin1z-vs-time" in temp_lal_h5)
            self.assertTrue("spin2x-vs-time" in temp_lal_h5)
            self.assertTrue("spin2y-vs-time" in temp_lal_h5)
            self.assertTrue("spin2z-vs-time" in temp_lal_h5)
            self.assertTrue("position1x-vs-time" in temp_lal_h5)
            self.assertTrue("position1y-vs-time" in temp_lal_h5)
            self.assertTrue("position1z-vs-time" in temp_lal_h5)
            self.assertTrue("position2x-vs-time" in temp_lal_h5)
            self.assertTrue("position2y-vs-time" in temp_lal_h5)
            self.assertTrue("position2z-vs-time" in temp_lal_h5)
            self.assertTrue("LNhatx-vs-time" in temp_lal_h5)
            self.assertTrue("LNhaty-vs-time" in temp_lal_h5)
            self.assertTrue("LNhatz-vs-time" in temp_lal_h5)
            self.assertTrue("Omega-vs-time" in temp_lal_h5)

        elif format == 3:
            self.assertTrue("remnant-mass-vs-time" in temp_lal_h5)
            self.assertTrue("remnant-spinx-vs-time" in temp_lal_h5)
            self.assertTrue("remnant-spiny-vs-time" in temp_lal_h5)
            self.assertTrue("remnant-spinz-vs-time" in temp_lal_h5)
            self.assertTrue("remnant-positionx-vs-time" in temp_lal_h5)
            self.assertTrue("remnant-positiony-vs-time" in temp_lal_h5)
            self.assertTrue("remnant-positionz-vs-time" in temp_lal_h5)

        time = temp_lal_h5["NRtimes"][()]
        self.assertTrue(len(time) > 0)

        lmax = temp_lal_h5.attrs["Lmax"]
        for l in range(2, lmax + 1):
            for m in range(-l, l + 1):
                amp_group_name = "amp_l%d_m%d" % (l, m)
                phase_group_name = "phase_l%d_m%d" % (l, m)
                self.assertTrue(amp_group_name in temp_lal_h5)
                self.assertTrue(phase_group_name in temp_lal_h5)
                raw_time, raw_amp, raw_phase = coalescence.strain_amp_phase_for_mode(l, m, extraction_radius)

                cut_index = np.argmax(raw_time > (75 + extraction_radius))
                expected_time = raw_time[cut_index:]
                expected_amp = raw_amp[cut_index:]
                expected_phase = raw_phase[cut_index:]

                self.assertEqual(len(time), len(expected_time))
                spline_amp = romspline.readSpline(temp_lal_h5_file_name, amp_group_name)
                spline_phase = romspline.readSpline(temp_lal_h5_file_name, phase_group_name)
                generated_amp = spline_amp(time)
                generated_phase = spline_phase(time)
                if m != 0:
                    self.assertTrue(np.all(np.isclose(expected_amp, generated_amp, atol=2e-3)))
                    self.assertTrue(np.all(np.isclose(expected_phase, generated_phase, atol=1e-2)))
                else:
                    self.assertTrue(np.all(generated_amp == 0))
                    self.assertTrue(np.all(generated_phase == 0))

        temp_lal_h5 = h5py.File(temp_lal_h5_file_name, 'r')
        self.assertTrue('frame' not in temp_lal_h5['auxiliary-info'].attrs)

        coalescence.close()
        os.remove(temp_lal_h5_file_name)

        # com corrected case
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)

            temp_lal_h5_file_name = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
            name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
            alternative_names = []
            extraction_radius = 70
            try:
                _put_data_in_lal_compatible_format(coalescence, temp_lal_h5_file_name, name, alternative_names,
                                                   extraction_radius, center_of_mass_correction=True,
                                                   NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                   contact_email='deirdre.shoemaker@austin.utexas.edu')
                self.fail()
            except ImportError:
                pass
            coalescence.close()

        else:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)

            temp_lal_h5_file_name = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
            name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
            alternative_names = []
            extraction_radius = 70
            _put_data_in_lal_compatible_format(coalescence, temp_lal_h5_file_name, name, alternative_names,
                                               extraction_radius, center_of_mass_correction=True, NR_group='UT Austin',
                                               NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                               contact_email='deirdre.shoemaker@austin.utexas.edu')

            temp_lal_h5 = h5py.File(temp_lal_h5_file_name, 'r')
            self.assertTrue('frame' in temp_lal_h5['auxiliary-info'].attrs)
            self.assertEqual('Center of mass drift corrected', temp_lal_h5['auxiliary-info'].attrs['frame'])

            coalescence.close()
            temp_lal_h5.close()
            os.remove(temp_lal_h5_file_name)

        # failed com corrected case
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)

            temp_lal_h5_file_name = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
            name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
            alternative_names = []
            extraction_radius = 70
            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(None, None)) as mock_center_of_mass:
                try:
                    _put_data_in_lal_compatible_format(coalescence, temp_lal_h5_file_name, name, alternative_names,
                                                       extraction_radius, center_of_mass_correction=True,
                                                       NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                       contact_email='deirdre.shoemaker@austin.utexas.edu')
                    self.fail()
                except ImportError:
                    pass
            coalescence.close()

        else:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)

            temp_lal_h5_file_name = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp.h5")
            name = "D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67"
            alternative_names = []
            extraction_radius = 70
            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(None, None)) as mock_center_of_mass:
                _put_data_in_lal_compatible_format(coalescence, temp_lal_h5_file_name, name, alternative_names,
                                                   extraction_radius, center_of_mass_correction=True,
                                                   NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                   contact_email='deirdre.shoemaker@austin.utexas.edu')

            temp_lal_h5 = h5py.File(temp_lal_h5_file_name, 'r')
            self.assertTrue('frame' not in temp_lal_h5['auxiliary-info'].attrs)

            coalescence.close()
            temp_lal_h5.close()
            os.remove(temp_lal_h5_file_name)

    def test_export_to_ascii(self):
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)
        output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output")
        pputils.export_to_ascii(coalescence, output_directory)

        expected_output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")
        actual_output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                               "resources/test_output/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")

        self.assertTrue(os.path.isdir(actual_output_directory))

        expected_filenames = set(os.listdir(expected_output_directory))
        actual_filenames = set(os.listdir(actual_output_directory))

        par_expected = False
        for filename in expected_filenames:
            if filename.endswith('.par'):
                par_expected = True
        if not par_expected:
            parfile_name = None
            for filename in actual_filenames:
                if filename.endswith('.par'):
                    parfile_name = filename
            if parfile_name is not None:
                actual_filenames.remove(parfile_name)
        self.assertEqual(expected_filenames, actual_filenames)

        for filename in expected_filenames:
            if not filename.endswith("par"):
                expected_values = np.loadtxt(os.path.join(expected_output_directory, filename))
                actual_values = np.loadtxt(os.path.join(actual_output_directory, filename))
                if "BH_diagnostics" not in filename:
                    self.assertTrue(np.all(np.isclose(expected_values, actual_values, atol=1e-4)))
                    if "Ylm_WEYLSCAL4" in filename:
                        with open(os.path.join(actual_output_directory, filename)) as f:
                            for i in range(4):
                                line = f.readline()
                            self.assertFalse(line.startswith('#'))
                else:
                    # check info only in BH_diagnostics
                    self.assertTrue(np.all(
                        np.isclose(expected_values[:, 5:], actual_values[:, 5:],
                                   atol=0.001)))
                    # check position data
                    self.assertTrue(np.all(
                        np.isclose(expected_values[:, 2:5], actual_values[:, 2:5],
                                   atol=0.05)))
                    # check iteration column
                    self.assertTrue(np.all(
                        np.isclose(expected_values[:, 0], actual_values[:, 0],
                                   rtol=0.001)))
                    # check time column
                    # the time column is rounded off in BH_diagnostics
                    self.assertTrue(np.all(
                        np.isclose(expected_values[:, 1], actual_values[:, 1],
                                   atol=0.001)))
            else:
                expected_file = os.path.join(expected_output_directory, filename)
                actual_file = os.path.join(actual_output_directory, filename)
                self.assertTrue(filecmp.cmp(expected_file, actual_file))

        if os.path.exists(actual_output_directory):
            shutil.rmtree(actual_output_directory)

        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)
        output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output")
        pputils.export_to_ascii(coalescence, output_directory, center_of_mass_correction=True)

        expected_output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                                 "resources/main_test_simulation/stitched/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")
        actual_output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                               "resources/test_output/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67")

        self.assertTrue(os.path.isdir(actual_output_directory))

        expected_filenames = set(os.listdir(expected_output_directory))
        actual_filenames = set(os.listdir(actual_output_directory))

        par_expected = False
        for filename in expected_filenames:
            if filename.endswith('.par'):
                par_expected = True
        if not par_expected:
            parfile_name = None
            for filename in actual_filenames:
                if filename.endswith('.par'):
                    parfile_name = filename
            if parfile_name is not None:
                actual_filenames.remove(parfile_name)
        self.assertEqual(expected_filenames, actual_filenames)

        for filename in expected_filenames:
            if not filename.endswith("par"):
                expected_values = np.loadtxt(os.path.join(expected_output_directory, filename))
                actual_values = np.loadtxt(os.path.join(actual_output_directory, filename))
                if "BH_diagnostics" not in filename:
                    self.assertTrue(np.all(np.isclose(expected_values, actual_values, atol=1e-4)))
                    if "Ylm_WEYLSCAL4" in filename:
                        with open(os.path.join(actual_output_directory, filename)) as f:
                            for i in range(4):
                                line = f.readline()
                            self.assertEqual("# Corrected for center of mass drift\n", line)
                else:
                    # check info only in BH_diagnostics
                    self.assertTrue(np.all(
                        np.isclose(expected_values[:, 5:], actual_values[:, 5:],
                                   atol=0.001)))
                    # check position data
                    self.assertTrue(np.all(
                        np.isclose(expected_values[:, 2:5], actual_values[:, 2:5],
                                   atol=0.05)))
                    # check iteration column
                    self.assertTrue(np.all(
                        np.isclose(expected_values[:, 0], actual_values[:, 0],
                                   rtol=0.001)))
                    # check time column
                    # the time column is rounded off in BH_diagnostics
                    self.assertTrue(np.all(
                        np.isclose(expected_values[:, 1], actual_values[:, 1],
                                   atol=0.001)))
            else:
                expected_file = os.path.join(expected_output_directory, filename)
                actual_file = os.path.join(actual_output_directory, filename)
                self.assertTrue(filecmp.cmp(expected_file, actual_file))

    def test__export_ascii_file(self):
        from mayawaves.utils.postprocessingutils import _export_ascii_file, _Ylm_WEYLSCAL4_ASC
        filename = "Ylm_WEYLSCAL4_temp.asc"
        data = np.linspace((1, 10, 100), (5, 50, 500), 50)
        coalescence_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources")
        l_value = 4
        m_value = -3
        radius = 80.00
        header = _Ylm_WEYLSCAL4_ASC.get_header(l_value, m_value, radius)

        _export_ascii_file(filename, data, coalescence_directory, header=header)
        filepath = os.path.join(coalescence_directory, filename)
        self.assertTrue(os.path.exists(filepath))

        generated_data = np.loadtxt(filepath)
        self.assertTrue(np.all(data == generated_data))

        header_string = ""
        with open(filepath) as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line.startswith('#'):
                    break
                header_string = header_string + line

        expected_header = header.replace("\n", "\n# ")
        expected_header = "# " + expected_header + "\n"
        self.assertEqual(expected_header, header_string)

        os.remove(filepath)

    @mock.patch("mayawaves.coalescence.Coalescence.spin_configuration", new_callable=PropertyMock)
    def test_export_to_lvcnr_catalog(self, mock_coalescence_spin_configuration):
        from mayawaves.radiation import Frame
        mock_coalescence_spin_configuration.return_value = "non-spinning"
        output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output")
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)
        coalescence.radiationbundle._RadiationBundle__frame = Frame.RAW
        pputils.export_to_lvcnr_catalog(coalescence, output_directory, NR_group='UT Austin', NR_code='MAYA',
                                        bibtex_keys='Jani:2016wkt',
                                        contact_email='deirdre.shoemaker@austin.utexas.edu')

        lal_h5_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/test_output/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        self.assertTrue(os.path.exists(lal_h5_filepath))

        # regression test
        expected_h5_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                            "resources/lvc_file/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")

        expected_file = h5py.File(expected_h5_filepath, 'r')
        actual_file = h5py.File(lal_h5_filepath, 'r')

        expected_attributes = set(expected_file.attrs)
        actual_attributes = set(actual_file.attrs)

        self.assertEqual(expected_attributes, actual_attributes)

        for attr in actual_attributes:
            if attr != 'modification-date':
                self.assertEqual(expected_file.attrs[attr], actual_file.attrs[attr])

        expected_groups = set(expected_file.keys())
        actual_groups = set(actual_file.keys())

        self.assertEqual(expected_groups, actual_groups)

        actual_time = actual_file["NRtimes"][()]
        expected_time = expected_file["NRtimes"][()]

        self.assertTrue(np.all(expected_time == actual_time))

        for group in expected_groups:
            if group != "NRtimes" and group != 'auxiliary-info':
                actual_spline = romspline.readSpline(lal_h5_filepath, group)
                actual_data = actual_spline(actual_time)
                expected_spline = romspline.readSpline(expected_h5_filepath, group)
                expected_data = expected_spline(expected_time)
                self.assertTrue(np.all(np.isclose(expected_data, actual_data, atol=1e-2)))

        actual_file.close()
        expected_file.close()
        os.remove(lal_h5_filepath)

        # com corrected case
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output")
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)
            try:
                pputils.export_to_lvcnr_catalog(coalescence, output_directory, center_of_mass_correction=True,
                                                NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                contact_email='deirdre.shoemaker@austin.utexas.edu')
                self.fail()
            except ImportError:
                pass

            coalescence.close()

        else:
            output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output")
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)
            pputils.export_to_lvcnr_catalog(coalescence, output_directory, center_of_mass_correction=True,
                                            NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                            contact_email='deirdre.shoemaker@austin.utexas.edu')

            lal_h5_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            self.assertTrue(os.path.exists(lal_h5_filepath))

            temp_lal_h5 = h5py.File(lal_h5_filepath, 'r')
            self.assertTrue('frame' in temp_lal_h5['auxiliary-info'].attrs)
            self.assertEqual('Center of mass drift corrected', temp_lal_h5['auxiliary-info'].attrs['frame'])

            coalescence.close()
            temp_lal_h5.close()
            os.remove(lal_h5_filepath)

        # failed com corrected case
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)
            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(None, None)) as mock_center_of_mass:
                try:
                    pputils.export_to_lvcnr_catalog(coalescence, output_directory, center_of_mass_correction=True,
                                                NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                contact_email='deirdre.shoemaker@austin.utexas.edu')
                    self.fail()
                except ImportError:
                    pass
                coalescence.close()

        else:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)
            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(None, None)) as mock_center_of_mass:
                pputils.export_to_lvcnr_catalog(coalescence, output_directory, center_of_mass_correction=True,
                                                NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                contact_email='deirdre.shoemaker@austin.utexas.edu')

            lal_h5_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                           "resources/test_output/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            self.assertTrue(os.path.exists(lal_h5_filepath))

            temp_lal_h5 = h5py.File(lal_h5_filepath, 'r')
            self.assertTrue('frame' not in temp_lal_h5['auxiliary-info'].attrs)

            coalescence.close()
            temp_lal_h5.close()
            os.remove(lal_h5_filepath)

        # if precessing without enough spin data, this should fail
        coalescence = Coalescence(h5_filename)
        mock_coalescence_spin_configuration.return_value = "precessing"
        pputils.export_to_lvcnr_catalog(coalescence, output_directory, NR_group='UT Austin', NR_code='MAYA',
                                        bibtex_keys='Jani:2016wkt',
                                        contact_email='deirdre.shoemaker@austin.utexas.edu')
        self.assertFalse(os.path.exists(lal_h5_filepath))

    @mock.patch("mayawaves.coalescence.Coalescence.spin_configuration", new_callable=PropertyMock)
    def test_export_to_lal_compatible_format(self, mock_coalescence_spin_configuration):
        mock_coalescence_spin_configuration.return_value = "non-spinning"
        output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output")
        h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                   "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        coalescence = Coalescence(h5_filename)
        pputils.export_to_lal_compatible_format(coalescence=coalescence, output_directory=output_directory,
                                                NR_group='UT Austin', NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                contact_email='deirdre.shoemaker@austin.utexas.edu', extraction_radius=70)

        lal_h5_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/test_output/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
        self.assertTrue(os.path.exists(lal_h5_filepath))

        # regression test
        expected_h5_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                            "resources/lal_file/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")

        expected_file = h5py.File(expected_h5_filepath, 'r')
        actual_file = h5py.File(lal_h5_filepath, 'r')

        expected_attributes = set(expected_file.attrs)
        actual_attributes = set(actual_file.attrs)

        for attr in actual_attributes:
            if attr != 'modification-date':
                if type(expected_file.attrs[attr]) == str:
                    self.assertEqual(expected_file.attrs[attr], actual_file.attrs[attr])
                else:
                    self.assertTrue(np.isclose(expected_file.attrs[attr], actual_file.attrs[attr], atol=1e-6))

        self.assertEqual(expected_attributes, actual_attributes)

        expected_groups = set(expected_file.keys())
        actual_groups = set(actual_file.keys())

        self.assertEqual(expected_groups, actual_groups)

        actual_time = actual_file["NRtimes"][()]
        expected_time = expected_file["NRtimes"][()]

        self.assertTrue(np.all(expected_time == actual_time))

        for group in expected_groups:
            if group != "NRtimes" and group != 'auxiliary-info':
                actual_spline = romspline.readSpline(lal_h5_filepath, group)
                actual_data = actual_spline(actual_time)
                expected_spline = romspline.readSpline(expected_h5_filepath, group)
                expected_data = expected_spline(expected_time)
                self.assertTrue(np.all(np.isclose(expected_data, actual_data, atol=1e-2)))

        actual_file.close()
        expected_file.close()
        os.remove(lal_h5_filepath)

        # com corrected case
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output")
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)
            try:
                pputils.export_to_lal_compatible_format(coalescence, output_directory, extraction_radius=70,
                                                        center_of_mass_correction=True, NR_group='UT Austin',
                                                        NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                        contact_email='deirdre.shoemaker@austin.utexas.edu')
                self.fail()
            except ImportError:
                pass

            coalescence.close()

        else:
            output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/test_output")
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)
            pputils.export_to_lal_compatible_format(coalescence, output_directory, extraction_radius=70,
                                                    center_of_mass_correction=True, NR_group='UT Austin',
                                                    NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                    contact_email='deirdre.shoemaker@austin.utexas.edu')

            lal_h5_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                           "resources/test_output/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            self.assertTrue(os.path.exists(lal_h5_filepath))

            temp_lal_h5 = h5py.File(lal_h5_filepath, 'r')
            self.assertTrue('frame' in temp_lal_h5['auxiliary-info'].attrs)
            self.assertEqual('Center of mass drift corrected', temp_lal_h5['auxiliary-info'].attrs['frame'])

            coalescence.close()
            temp_lal_h5.close()
            os.remove(lal_h5_filepath)

        # failed com corrected case
        import sys
        if sys.version_info.major == 3 and sys.version_info.minor > 10:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)
            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(None, None)) as mock_center_of_mass:
                try:
                    pputils.export_to_lal_compatible_format(coalescence, output_directory, extraction_radius=70,
                                                            center_of_mass_correction=True, NR_group='UT Austin',
                                                            NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                            contact_email='deirdre.shoemaker@austin.utexas.edu')
                    self.fail()
                except ImportError:
                    pass

                coalescence.close()

        else:
            h5_filename = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                       "resources/temp/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            coalescence = Coalescence(h5_filename)
            with patch.object(Coalescence, 'center_of_mass', new_callable=PropertyMock,
                              return_value=(None, None)) as mock_center_of_mass:
                pputils.export_to_lal_compatible_format(coalescence, output_directory, extraction_radius=70,
                                                        center_of_mass_correction=True, NR_group='UT Austin',
                                                        NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                        contact_email='deirdre.shoemaker@austin.utexas.edu')

            lal_h5_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                           "resources/test_output/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67.h5")
            self.assertTrue(os.path.exists(lal_h5_filepath))

            temp_lal_h5 = h5py.File(lal_h5_filepath, 'r')
            self.assertTrue('frame' not in temp_lal_h5['auxiliary-info'].attrs)

            coalescence.close()
            temp_lal_h5.close()
            os.remove(lal_h5_filepath)

        # if precessing without enough spin data, this should just warn
        coalescence = Coalescence(h5_filename)
        mock_coalescence_spin_configuration.return_value = "precessing"
        pputils.export_to_lal_compatible_format(coalescence, output_directory, NR_group='UT Austin',
                                                NR_code='MAYA', bibtex_keys='Jani:2016wkt',
                                                contact_email='deirdre.shoemaker@austin.utexas.edu',)
        self.assertTrue(os.path.exists(lal_h5_filepath))
        os.remove(lal_h5_filepath)

    @mock.patch("matplotlib.pyplot.show")
    def test_summarize_coalescence(self, mock_show):
        self.maxDiff = None
        # full simulation with all the data
        simulation_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                           "resources/format_test/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_format_3.h5")

        coalescence = Coalescence(simulation_filepath)
        output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp")
        pputils.summarize_coalescence(coalescence, output_directory)

        # Check if directory was created in correct location to store data.
        simulation_output_directory = os.path.join(output_directory, "summary_" + coalescence.name)
        self.assertTrue(os.path.exists(simulation_output_directory))
        # Check if correct summary information was saved to this directory.
        path_to_summary_file = os.path.join(simulation_output_directory, 'summary.txt')
        self.assertTrue(os.path.exists(path_to_summary_file))
        path_to_separation_plot = os.path.join(simulation_output_directory, 'separation.png')
        self.assertTrue(os.path.exists(path_to_separation_plot))
        path_to_position_plot = os.path.join(simulation_output_directory, 'position.png')
        self.assertTrue(os.path.exists(path_to_position_plot))
        path_to_psi4_plot = os.path.join(simulation_output_directory, 'psi4_22.png')
        self.assertTrue(os.path.exists(path_to_psi4_plot))
        path_to_strain_plot = os.path.join(simulation_output_directory, 'strain_22.png')
        self.assertTrue(os.path.exists(path_to_strain_plot))

        # Check contents of summary.txt
        expected_output = f"""Summary for D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_format_3

Initial Values (t=0 M):
----------------------------------------------------------------------
mass ratio:				2.0000
primary horizon mass:			0.6665
secondary horizon mass:			0.3333
primary dimensionless spin:		[0.0000, 0.0000, 0.4001]
secondary dimensionless spin:		[0.0000, 0.0000, 0.4000]
separation:				11.0001 M

After Junk Radiation (t=75 M):
----------------------------------------------------------------------
primary horizon mass:			0.6667
secondary horizon mass:			0.3334
primary dimensionless spin:		[-0.0000, 0.0000, 0.4007]
secondary dimensionless spin:		[-0.0000, 0.0000, 0.4020]
separation:				10.7785 M
separation unit vector:			[0.1005, -0.9949, -0.0000]
orbital frequency:			0.0243
orbital angular momentum unit vector:	[0.0000, 0.0000, 1.0000]
eccentricity:				0.0034

Remnant Values (Last available data):
----------------------------------------------------------------------
horizon mass:				0.9492
dimensionless spin:			[-0.0000, 0.0000, 0.7701]"""

        summary_file = open(path_to_summary_file, "r")
        actual_output = summary_file.read()
        self.assertEqual(expected_output, actual_output)
        summary_file.close()

        # Remove test output directory after checking.
        shutil.rmtree(simulation_output_directory)

        # Run without specifying a directory and check what is displayed.
        with mock.patch('sys.stdout', new_callable=io.StringIO) as stdout_mock:
            pputils.summarize_coalescence(coalescence)
            actual_output = stdout_mock.getvalue()
            self.assertEqual(expected_output + '\n', actual_output)
            self.assertEqual(4, mock_show.call_count)
            mock_show.reset_mock()

        coalescence.close()

        # full simulation with missing horizon data
        #   with output directory

        simulation_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                           "resources/format_test/D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_format_1.h5")

        coalescence = Coalescence(simulation_filepath)
        output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp")
        pputils.summarize_coalescence(coalescence, output_directory)

        # Check if directory was created in correct location to store data.
        simulation_output_directory = os.path.join(output_directory, "summary_" + coalescence.name)
        self.assertTrue(os.path.exists(simulation_output_directory))
        # Check if correct summary information was saved to this directory.
        path_to_summary_file = os.path.join(simulation_output_directory, 'summary.txt')
        self.assertTrue(os.path.exists(path_to_summary_file))
        path_to_separation_plot = os.path.join(simulation_output_directory, 'separation.png')
        self.assertTrue(os.path.exists(path_to_separation_plot))
        path_to_position_plot = os.path.join(simulation_output_directory, 'position.png')
        self.assertTrue(os.path.exists(path_to_position_plot))
        path_to_psi4_plot = os.path.join(simulation_output_directory, 'psi4_22.png')
        self.assertTrue(os.path.exists(path_to_psi4_plot))
        path_to_strain_plot = os.path.join(simulation_output_directory, 'strain_22.png')
        self.assertTrue(os.path.exists(path_to_strain_plot))

        # Check contents of summary.txt
        expected_output = f"""Summary for D2.33_q1_a1_0_0_0_a2_0_0_0_m42.67_format_1

Initial Values (t=0 M):
----------------------------------------------------------------------
mass ratio:				1.0000
primary horizon mass:			0.5168
secondary horizon mass:			0.5168
primary dimensionless spin:		[0.0000, 0.0000, 0.0006]
secondary dimensionless spin:		[0.0000, 0.0000, 0.0006]
separation:				2.3363 M

After Junk Radiation (t=75 M):
----------------------------------------------------------------------
primary horizon mass:			NaN
secondary horizon mass:			NaN
primary dimensionless spin:		NaN
secondary dimensionless spin:		NaN
separation:				0.0000 M
separation unit vector:			[-0.1809, -0.9835, 0.0000]
orbital frequency:			0.4345
orbital angular momentum unit vector:	[0.0000, 0.0000, 1.0000]
eccentricity:				0.0243
**A 'NaN' value represents data that was not being tracked at t=75 M.**

Remnant Values (Last available data):
----------------------------------------------------------------------
horizon mass:				0.9771
dimensionless spin:			[-0.0000, 0.0000, 0.6763]"""

        summary_file = open(path_to_summary_file, "r")
        actual_output = summary_file.read()
        self.assertEqual(expected_output, actual_output)
        summary_file.close()

        # Remove test output directory after checking.
        shutil.rmtree(simulation_output_directory)

        # Run without specifying a directory and check what is displayed.
        with mock.patch('sys.stdout', new_callable=io.StringIO) as stdout_mock:
            pputils.summarize_coalescence(coalescence)
            actual_output = stdout_mock.getvalue()
            self.assertEqual('There is not enough data to crop to four orbits\n' + expected_output + '\n',
                             actual_output)
            self.assertEqual(4, mock_show.call_count)
            mock_show.reset_mock()

        coalescence.close()

        # unmerged simulation

        simulation_filepath = os.path.join(TestPostprocessingUtils.CURR_DIR,
                                           "resources/D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_unmerged.h5")

        coalescence = Coalescence(simulation_filepath)
        output_directory = os.path.join(TestPostprocessingUtils.CURR_DIR, "resources/temp")
        pputils.summarize_coalescence(coalescence, output_directory)

        # Check if directory was created in correct location to store data.
        simulation_output_directory = os.path.join(output_directory, "summary_" + coalescence.name)
        self.assertTrue(os.path.exists(simulation_output_directory))
        # Check if correct summary information was saved to this directory.
        path_to_summary_file = os.path.join(simulation_output_directory, 'summary.txt')
        self.assertTrue(os.path.exists(path_to_summary_file))
        path_to_separation_plot = os.path.join(simulation_output_directory, 'separation.png')
        self.assertTrue(os.path.exists(path_to_separation_plot))
        path_to_position_plot = os.path.join(simulation_output_directory, 'position.png')
        self.assertTrue(os.path.exists(path_to_position_plot))
        path_to_psi4_plot = os.path.join(simulation_output_directory, 'psi4_22.png')
        self.assertTrue(os.path.exists(path_to_psi4_plot))
        path_to_strain_plot = os.path.join(simulation_output_directory, 'strain_22.png')
        self.assertTrue(os.path.exists(path_to_strain_plot))

        # Check contents of summary.txt
        expected_output = f"""Summary for D11_q2_a1_0.0_0.0_0.4_a2_0.0_0.0_0.4_m282.35_unmerged

Initial Values (t=0 M):
----------------------------------------------------------------------
mass ratio:				2.0000
primary horizon mass:			0.6665
secondary horizon mass:			0.3333
primary dimensionless spin:		[0.0000, 0.0000, 0.4001]
secondary dimensionless spin:		[0.0000, 0.0000, 0.4000]
separation:				11.0001 M

After Junk Radiation (t=75 M):
----------------------------------------------------------------------
primary horizon mass:			0.6667
secondary horizon mass:			0.3334
primary dimensionless spin:		[-0.0000, 0.0000, 0.4007]
secondary dimensionless spin:		[-0.0000, 0.0000, 0.4020]
separation:				10.7785 M
separation unit vector:			[0.1005, -0.9949, -0.0000]
orbital frequency:			0.0243
orbital angular momentum unit vector:	[0.0000, 0.0000, 1.0000]
eccentricity:				0.0001

Current Values (Last available data):
----------------------------------------------------------------------
current time:				499.9133
separation:				10.2176
separation unit vector:			[-10.2141, 0.2670, 0.0000]
orbital frequency:			0.0270
orbital angular momentum unit vector:	[0.0000, -0.0000, 1.0000]"""

        summary_file = open(path_to_summary_file, "r")
        actual_output = summary_file.read()
        self.assertEqual(expected_output, actual_output)
        summary_file.close()

        # Remove test output directory after checking.
        shutil.rmtree(simulation_output_directory)

        # Run without specifying a directory and check what is displayed.
        with mock.patch('sys.stdout', new_callable=io.StringIO) as stdout_mock:
            pputils.summarize_coalescence(coalescence)
            actual_output = stdout_mock.getvalue()
            self.assertEqual('There is not enough data to crop to four orbits\n' + expected_output + '\n',
                             actual_output)
            self.assertEqual(4, mock_show.call_count)
            mock_show.reset_mock()

        coalescence.close()
