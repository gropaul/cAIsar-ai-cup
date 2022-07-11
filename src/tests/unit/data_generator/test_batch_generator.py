from math import ceil
from operator import le
import re
import unittest
from typing import List, Tuple
import numpy as np
import pandas as pd
import os
from data_generator.batch_generator import BatchGenerator
from data_generator.batch_generator_functions import pad_df,concat_ts_partials

from utils.util_functions import save_index, printc
from utils.SYSCONFIG import DATA_PATH, PLATFORM,TEST_DATA_PATH
"""
todo:
[x] create test for simple config
    [x] create test data 
    [x] run bg with test data
    [x] check bg results
[ ] create custom configs 
[ ] adapt test outcome expectations by config 
[ ] create test for full pipeline: Calculate f-1-score from training data -> must be 1 -> easy test for whole loop 
[ ] mit random numpy arrays den f-score testen
"""
class TestBatchGenerator(unittest.TestCase):

    verbose = True

    def send_message(self, message: str, **kwargs) -> None:
        if self.verbose:
            printc(source='[TestBatchGenerator]', message=message, **kwargs)
    # region Test File creation

    data_config = {
        'data_columns' : ['m_a_float', 'm_b_float', 'm_c_int', 'm_d_int'],
        'label_columns' : ['l_a_int', 'l_b_int']
    }

    data_types = {
        'm_a_float': np.float64, 
        'm_b_float': np.float64,
        'm_c_int': np.int8, 'm_c_int': np.int8,
        'l_a_int': np.int8, 'l_b_int': np.int8, 
    }
    def get_whole_timeseries(self,dfs:List[pd.DataFrame], sample_offset:int, length:int, sample_freq:int,batch_size:int, fill:int = 0):
        # only for padding=True and pad_batch = True



        for i in range(len(dfs)):
            df = dfs[i]
            print(f"Df size before pad: {df.shape[0]}")
            df = pad_df(df,sample_offset,length,sample_freq)
            print(f"Df size after pad: {df.shape[0]}")
            dfs[i] = df

        number_of_rows = pd.concat(dfs).shape[0]
        print(f"All files together add up to {number_of_rows} rows")
        # as padding is enabled, they have to fit perfectly
        number_of_segments = number_of_rows // length
        
        number_of_batches:int = number_of_segments // batch_size

        # if number of segments is not divideable by batchsize, one padding batch is appended
        if number_of_segments % batch_size != 0:
            print("As the number of segments is not divideable by the batch size, one batch is added as padding")
            number_of_batches += 1

        # how many segments we need for the next batch to be filled
        target_number_of_segments = number_of_batches * batch_size

        print(f"The data offerst {number_of_segments} segments for {number_of_batches}, but {target_number_of_segments} are needed for filling up the next batch")

        if number_of_segments < target_number_of_segments:
        
            row_target_number = target_number_of_segments * length
            padding = pd.DataFrame(fill, index=np.arange(number_of_rows,row_target_number), columns=df.columns)
            print(f"Added a padding with {padding.shape[0]} entries, as {row_target_number} rows a needed")
            dfs.append(padding)

        return pd.concat(dfs)
        


    def create_test_dataframe(self,number_rows:int)->pd.DataFrame:
        return pd.DataFrame({
            "m_a_float" : np.arange(0,10,10/number_rows),
            "m_b_float" : np.arange(0,20,20/number_rows),
            "m_c_int": np.arange(0,23,23/number_rows).astype(np.int8),
            "m_d_int": np.arange(0,number_rows,1).astype(np.int8),
            "l_a_int": np.arange(0,2,2/number_rows).astype(np.int8),
            "l_b_int": np.arange(0,2,2/number_rows).astype(np.int8)
    })

    def create_test_files(self,config:List[int])->Tuple[List[pd.DataFrame],str]: 
        """Creates testfiles an a corresponding file index

        Args:
            config (List[int]): creates one file for each entry with the entry as number of rows
        
        Returns:
            _type_: Tuple, First value is a List of Dataframes and the second value is a file path to the files index
        """
        
        dataframes:List[pd.DataFrame] = [self.create_test_dataframe(rows) for rows in config]
        file_index:List[str] = []



        # Check whether the specified path exists or not
        isExist = os.path.exists(TEST_DATA_PATH)

        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(TEST_DATA_PATH)
            self.send_message("A new test data folder was created.")
        else:
            self.send_message("Found an existing test data folder")

        # save dataframes as csv files to mimic real preconditions 
        for i in range(len(dataframes)):
            path = TEST_DATA_PATH + ('\\' if PLATFORM == 'WINDOWS' else '/') + f'test_file_{i}.csv'
            dataframes[i].to_csv(path,index=False)
            file_index.append(path)
        
        index_path =  TEST_DATA_PATH + ('\\' if PLATFORM == 'WINDOWS' else '/') + f'test_index.txt'
        
        save_index(index_path,file_index)
        self.send_message(f"Created file index containing {len(file_index)} files and saved it to {index_path}")

        return dataframes,index_path

    # endregion 
    def possible_segments_count(self,file_config:List[int],padding:bool,batch_size:int,length:int,sample_freq:int,pad_batch:bool):
        """
        Tested empirically: 
        length=8, sample_freq=4, batch_size=4, padding=True
        Number of batch_items for 22 rows: 5, max_start_index=16
        Number of batch_items for 15 rows: 3, max_start_index=8
        Number of batch_items for 32 rows: 7, max_start_index=24
        Number of batch_items for 120 rows: 29, max_start_index=112

        length=8, sample_freq=4, batch_size=4, padding=False
        Number of batch_items for 22 rows: 4, max_start_index=14
        Number of batch_items for 15 rows: 2, max_start_index=7
        Number of batch_items for 32 rows: 7, max_start_index=24
        Number of batch_items for 120 rows: 29, max_start_index=112
        """
        sum_segments = 0

        # print(f"length={length}, sample_freq={sample_freq}, batch_size={batch_size}, padding={padding}")
        
        for number_of_rows in file_config:

            max_start_index = 0 
            if padding: 
        
                # if it perfectly fits, no improvements
                if ((number_of_rows - length) % sample_freq == 0): 
                    max_start_index =  number_of_rows - length
                    
                else:
                    # number of items that fit without padding 
                    n = (number_of_rows - length) // sample_freq
                    n = n + 1
                    max_start_index = n * sample_freq

            else: max_start_index = number_of_rows - length

            # plus 1 as the last one also fits 
            number_of_segments = max_start_index // sample_freq + 1
            # print(f"Number of batch_items for {number_of_rows} rows: {number_batch_items}, max_start_index={max_start_index}")
            sum_segments += number_of_segments

        # if number of segments is divideable by batchsize, no furher adaption needed
        if sum_segments % batch_size == 0:
            return sum_segments
        else: 
            # possible number ob batches
            number_of_batches:int = sum_segments // batch_size

            # number of batches increased if pad_batches is enabled
            if pad_batch: return (number_of_batches + 1) * batch_size
            
            else: return number_of_batches * batch_size
    def flatten(self,t):

        return [item for sublist in t for item in sublist]

    def test_batch_generator(self):
        
        # region Init Batch Generator

        # len(file_config) = number of files, value at x = number of rows of file_x

        batch_size = 4
        length = 8
        sample_freq = 4
        data_columns =  self.data_config['data_columns']
        label_columns =  self.data_config['label_columns']

        padding = False
        pad_batch = False

        #file_config = [22,15,32,120]
        file_config = [8,8]
        data, index_path = self.create_test_files(config=file_config)

        config = {
            'index_file' :  index_path,
            'data_columns' : data_columns,
            'label_columns' : label_columns,
            'batch_size' : batch_size,
            'shuffle' : False,
            'dtype' : self.data_types,
            'verbose' : True,
            'table_file' : f'{TEST_DATA_PATH}\\test_table.pkl' if PLATFORM == 'WINDOWS' else f'{DATA_PATH}/test_table.pkl',
            'length' : length,
            'validation_generator' : False,
            'train_test_split' : 1.0,
            'sample_freq' : sample_freq,
            'sample_offset' : 0,
            'caching' : False,
            'padding' : padding,
            'pad_last_batch' : pad_batch
        }
        """
        
        batch_generator = BatchGenerator(**config)
        
        # endregion
        
        # region Check Batch Size, Length and Dimensions
        X, y = batch_generator.__getitem__(0)
        
        # Check formats of batches
        x_batch_size, x_batch_length, x_batch_dim = X.shape
        y_batch_size, y_batch_length, y_batch_dim = y.shape
        assert (x_batch_size == y_batch_size == batch_size), 'Batch Size must keep the same'
        assert (x_batch_length == y_batch_length == length), 'Batch Length must keep the same'
        assert (x_batch_dim == len(data_columns)), 'Dimension of x must be the same as the length of data_columns'
        assert (y_batch_dim == len(label_columns)), 'Dimension of x must be the same as the length of label_columns'

        # endregion

        # region Check if number of batches fits with data size
        
        # check number of batches with no padding enabled 

        padding = False

        _config = config.copy()
        _config['padding'] = padding
        batch_generator = BatchGenerator(**_config)
        X, y = batch_generator.get_data()
    
        # true value
        possible_batch_items_count = self.possible_segments_count(file_config,padding,batch_size,length,sample_freq,pad_batch)

        self.send_message(f"Checking the possible amount of items under the following settings: ")
        self.send_message(f"length={length}, sample_freq={sample_freq}, batch_size={batch_size}, padding={padding}")
        self.send_message(f"There should be {possible_batch_items_count} items generated out of the data set, the batch generater produces {len(X)}")

        assert (possible_batch_items_count == len(X) == len(y)), 'Number of batch items must be the same for no padding'

        padding = True

        _config = config.copy()
        _config['padding'] = padding
        batch_generator = BatchGenerator(**_config)
        X, y = batch_generator.get_data()

        possible_batch_items_count = self.possible_segments_count(file_config,padding,batch_size,length,sample_freq,pad_batch)

        self.send_message(f"Checking the possible amount of items under the following settings: ")
        self.send_message(f"length={length}, sample_freq={sample_freq}, batch_size={batch_size}, padding={padding}")
        self.send_message(f"There should be {possible_batch_items_count} items generated out of the data set, the batch generater produces {len(X)}")
        
        assert (possible_batch_items_count == len(X) == len(y)), 'Number of batch items must be the same for padding enabled'

        # endregion 
        """

        # region check if ts is concatable

        config['padding'] = True
        config['pad_last_batch'] = True
        config['shuffle'] = False

        batch_generator = BatchGenerator(**config)
        X, y = batch_generator.get_data()
        
        X = self.flatten(X)
        y = self.flatten(y)

        x_df = pd.DataFrame(X,columns=data_columns)
        y_df = pd.DataFrame(y,columns=label_columns)

        batch_res = x_df.join(y_df)

        batch_res.to_csv("batch_res.csv",index=False)

        whole_ts = self.get_whole_timeseries(data,sample_offset=0,length=length,sample_freq=sample_freq,batch_size=batch_size,fill=0)
        print(whole_ts.shape)
        whole_ts.to_csv("whole_ts.csv",index=False)

        print(batch_res.shape)

        
if __name__ == '__main__':
    TestBatchGenerator().test_batch_generator()