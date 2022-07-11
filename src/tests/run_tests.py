import unittest

from tests.unit.utils.test_util_functions import TestUtilFunctions
from tests.unit.data_generator.test_batch_generator_functions import TestBatchGeneratorFunctions
from tests.unit.data_generator.test_batch_generator import TestBatchGenerator

def run_tests():
    test_classes = [TestUtilFunctions, TestBatchGeneratorFunctions, TestBatchGenerator]
    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    test_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    _ = runner.run(test_suite)

if __name__ == '__main__':
    run_tests()


