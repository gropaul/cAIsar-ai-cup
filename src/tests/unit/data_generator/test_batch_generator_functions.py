import unittest
import numpy as np
import pandas as pd
from data_generator.batch_generator_functions import pad_df,concat_ts_partials

class TestBatchGeneratorFunctions(unittest.TestCase):
    def test_func_concat_ts_partials(self):
        '''
        test the concat_ts_partials() - method
        '''
        
        # Case 1:
        # test on a time-series of length 32
        # sampled at a frequency of 8; length 16 
        center_offset = 4
        center = 8
        full_arr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        partials = [
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]),
        ]

        concatenated_arr = concat_ts_partials(partials=partials,center=center,center_offset=center_offset)

        assert (full_arr == concatenated_arr).all(), 'Arrays should match element-wise.'
        del center_offset
        del center

        # Case 1:
        # test on a time-series of length 32
        # sampled at a frequency of 8; length 16 
        self.center_offset = 4
        self.center = 8
        full_arr = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        partials = [
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]),
        ]

        concatenated_arr = concat_ts_partials(self, partials=partials)

        assert (full_arr == concatenated_arr).all(), 'Arrays should match element-wise.'
        del self.center_offset
        del self.center


    def test_pad_df(self):


        # Case 1:
        # test on a sample data frame that needs *no* padding
        sample_offset = 0
        sample_freq = 8
        length = 16

        df = pd.DataFrame({
            'A' : np.arange(32),
            'B' : np.arange(32),
        })
        padded_df = pad_df(df=df,sample_offset=sample_offset,length=length,sample_freq=sample_freq, fill=0)

        assert df.equals(padded_df), 'Data frames should be equal as no padding is required.'

        # Case 2:
        # test on a sample data frame that needs padding
        sample_offset = 0
        sample_freq = 8
        length = 16

        df = pd.DataFrame({
            'A' : np.arange(34),
            'B' : np.arange(34),
        })
        padding = pd.DataFrame({
            'A' : np.zeros(6, dtype=np.uint),
            'B' : np.zeros(6, dtype=np.uint),
            'index' : np.arange(34, 40)
        })
        padding.set_index('index', inplace=True)

        manually_padded_df = pd.concat([df, padding])
        auto_padded_df = pad_df(df=df,sample_offset=sample_offset,length=length,sample_freq=sample_freq, fill=0)

        assert manually_padded_df.equals(auto_padded_df), 'Data frames should be padded equally.'

if __name__ == '__main__':
    unittest.main()