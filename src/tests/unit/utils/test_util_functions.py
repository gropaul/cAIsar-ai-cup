import unittest
import numpy as np
from utils.util_functions import convert_mask_to_cup_format
from utils.util_functions import shift_array


class TestUtilFunctions(unittest.TestCase):

    def test_convert_mask_to_cup_format(self):
        '''
        test the convert_mask_to_cup_format() - method
        '''

        mask = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1])
        # [8, 8] is ignored as it is not valid
        converted_true = [[2, 3], [5, 6], [10, 11]]
        converted_calc = convert_mask_to_cup_format(mask)
        assert (converted_calc == converted_true), f'Calculated format must match.\ncalculated: {converted_calc}\nactual: {converted_true}'

        mask = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        converted_true = [[0, 1], [3, 4], [8, 11]]
        converted_calc = convert_mask_to_cup_format(mask)
        assert (converted_calc == converted_true), f'Calculated format must match.\ncalculated: {converted_calc}\nactual: {converted_true}'
        
        mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        converted_true = [[0, 11]]
        converted_calc = convert_mask_to_cup_format(mask)
        assert (converted_calc == converted_true), f'Calculated format must match.\ncalculated: {converted_calc}\nactual: {converted_true}'
        
        mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        converted_true = []
        converted_calc = convert_mask_to_cup_format(mask)
        assert (converted_calc == converted_true), f'Calculated format must match.\ncalculated: {converted_calc}\nactual: {converted_true}'

    def test_shift_array(self):
        '''
        test the shift_array() - method
        '''
        
        arr = np.array([1, 2, 3, 4, 5, 6])
        shift = 2
        fill = 0
        shifted_calc = shift_array(arr=arr, num=shift, fill=fill)
        shifted_true = np.array([0, 0, 1, 2, 3, 4])
        assert (shifted_calc == shifted_true).all(), f'Shifted arrays must match.\ncalculated {shifted_calc}\nactual: {shifted_true}'
        
        arr = np.array([1, 2, 3, 4, 5, 6])
        shift = -2
        fill = 0
        shifted_calc = shift_array(arr=arr, num=shift, fill=fill)
        shifted_true = np.array([3, 4, 5, 6, 0, 0])
        assert (shifted_calc == shifted_true).all(), f'Shifted arrays must match.\ncalculated {shifted_calc}\nactual: {shifted_true}'
        
        arr = np.array([1, 2, 3, 4, 5, 6])
        shift = 0
        fill = 0
        shifted_calc = shift_array(arr=arr, num=shift, fill=fill)
        shifted_true = np.array([1, 2, 3, 4, 5, 6])
        assert (shifted_calc == shifted_true).all(), f'Shifted arrays must match.\ncalculated {shifted_calc}\nactual: {shifted_true}'
        
        arr = np.array([1, 2, 3, 4, 5, 6])
        shift = 9
        fill = 0
        shifted_calc = shift_array(arr=arr, num=shift, fill=fill)
        shifted_true = np.array([0, 0, 0, 0, 0, 0])
        assert (shifted_calc == shifted_true).all(), f'Shifted arrays must match.\ncalculated {shifted_calc}\nactual: {shifted_true}'
        
if __name__ == '__main__':
    unittest.main()