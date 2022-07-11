import json
import os
import random
from typing import Dict, List, Tuple
from data_generator.data_generator_constants import ENCODING_DICT, MISSING_VALUE, MISSING_VALUES_DICT, NORMALIZE_AGE_DICT, NORMALIZE_BMI_DICT, NORMALIZE_HEIGHT_DICT, NORMALIZE_WALKED_DISTANCE_DICT, NORMALIZE_WALKING_SPEED_DICT, NORMALIZE_WEIGHT_DICT, min_max_scaling_key
from data_processing.data_normalization_params import DataNormalizationParams
from data_processing.data_processing_functions import match_csv_json_pairs, match_csv_normalization_data_pairs
from data_processing.runtime_normalizer import apply_normalization
from tensorflow.keras.utils import Sequence
from os.path import exists
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import math
import pickle

from utils.SYSCONFIG import DATA_PATH, PLATFORM, PROCESSOR_CORES
from utils.util_functions import CustomBar, create_zero_frame, create_zero_meta_data, get_partial_dict, printc, parse_index, create_zero_norm_meta_data

import data_generator.batch_generator_functions as bgf
import data_generator.batch_data_path_generator as pg

class BatchGenerator(Sequence):
    '''
    data generator to produce batches,
    '''

    def __init__(self, data_subset_name: str, data_columns: List[str], label_columns: List[str], normalization_settings: dict = DataNormalizationParams.default,
            batch_size: int = 32, shuffle: bool = True, chunk_size: int = 4, dtype: dict = None, verbose: bool = False, 
            length: int = 1024, validation_generator: bool = False, 
            train_test_split: bool = 1.0, augmenter = None, sample_freq: int = 512, sample_offset: int = 0, 
            caching: bool = False, padding: bool = False, pad_last_batch: bool = False, meta: bool = False, **kwargs) -> None:
        
        # batch generator settings: general
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.train_test_split = train_test_split
        self.validation_generator = validation_generator

        # batch generator settings: implementation-specific
        self.length = length

        self.sample_freq = sample_freq
        self.sample_offset = sample_offset
        self.padding = padding
        self.pad_last_batch = pad_last_batch


        # dependent vars to concat predictions 
        self.center = int(self.length / 2)
        self.center_offset = int(self.length / 4)

        # data-related settings
        self.data_subset_name = data_subset_name
        self.data_columns = data_columns
        self.label_columns = label_columns
        self.dtype = dtype

        self.normalization_function = normalization_settings['normalization_function']
        self.normalization_mode = normalization_settings['normalization_mode']

        # helper file locations
        self.index_file = pg.get_csv_index_path(data_subset_name)      
        self.table_file = pg.get_table_name(data_subset_name)      
        self.meta_table_file = pg.get_meta_table_name(data_subset_name)
        self.json_index = pg.get_original_json_index_path(data_subset_name)     
        self.norm_meta_index_path = pg.get_norm_meta_index_path(data_subset_name)     

        # meta data and data augmentation
        self.meta = meta
        self.augmenter = augmenter

        # init sequence
        self.send_message('Started initialization ...')
        self.send_message(f'Selected dataset: {data_subset_name}')
        self.send_message(f'Selected normalization function: {self.normalization_function}')
        self.send_message(f'Selected normalization mode: {self.normalization_mode}')

        zero_path = os.path.join(DATA_PATH, 'zero.xsv')
        zero_meta_path = os.path.join(DATA_PATH, 'zero.xson')
        zero_norm_path = os.path.join(DATA_PATH, 'zero_norm.xson')
        if self.pad_last_batch:
            if not(exists(zero_path)):
                self.send_message(f'Found no zero frame, but pad_last_batch is set to True. Creating frame ...')
                create_zero_frame()
                self.send_message(f'Created zero frame at {zero_path}')    

            if not(exists(zero_norm_path)):
                self.send_message(f'Found no zero frame for normalization meta data, but pad_last_batch is set to True. Creating frame ...')
                create_zero_norm_meta_data()
                self.send_message(f'Created zero frame at {zero_path}')  

            if self.meta:
                if not(exists(zero_meta_path)):
                    self.send_message(f'Found no zero meta data, but pad_last_batch and meta are both set to True. Creating zero meta data ...')
                    create_zero_meta_data()
                    self.send_message(f'Created zero meta data at {zero_meta_path}')             
        
        self.send_message(f'Parsing and verifying index {self.index_file}')
        self.file_list = parse_index(self.index_file, verify=True)
        self.send_message(f'Found {len(self.file_list)} data files.')
        
        self.total, self.sample_table = self.generate_sample_table()
        self.split_stamp = self.determine_split()
        
        self.batches = self.group_data_by_batch()

        # init normalization meta data like std, avg, mean, ...
        key_csv_norm_json_table = match_csv_normalization_data_pairs(self.index_file,  self.norm_meta_index_path)
        self.csv_to_norm_meta_table = {}
        for key in key_csv_norm_json_table.keys():
            csv_file = key_csv_norm_json_table[key]['csv']
            json_file = key_csv_norm_json_table[key]['json']
            self.csv_to_norm_meta_table[csv_file] = json_file
        self.csv_to_norm_meta_table[zero_path] = zero_norm_path

        # init run meta data
        if self.meta:
            key_csv_json_table = match_csv_json_pairs(self.index_file, self.json_index)
            self.csv_to_json_table = {}
            for key in key_csv_json_table.keys():
                csv_file = key_csv_json_table[key]['csv']
                json_file = key_csv_json_table[key]['json']
                self.csv_to_json_table[csv_file] = json_file
            self.csv_to_json_table[zero_path] = zero_meta_path
        
        self.caching = caching
        if self.caching:
            self.send_message('Caching is turned ON.')
            self.send_message('Initializing CSV cache and pre-storing files...')
            self.cache: Dict[str, pd.DataFrame] = dict()
            for file in self.file_list:
                df = pd.read_csv(file, usecols=[*self.data_columns, *self.label_columns], dtype=self.dtype, index_col=False)
                if self.padding:
                    df = bgf.pad_df(df=df,sample_offset= self.sample_offset,length=self.length,sample_freq=self.sample_freq)
                self.cache[file] = df
            self.cache[zero_path] = pd.read_csv(zero_path, usecols=[*self.data_columns, *self.label_columns], dtype=self.dtype ,index_col=False)
            

            # Normalization Meta Data
            self.send_message('Initializing JSON for normalization meta data and pre-storing files...')
            self.norm_meta_cache: Dict[str, dict] = dict()
            for file in self.file_list:
                meta_file = self.csv_to_norm_meta_table[file]
                with open(meta_file, 'r') as f:
                    meta_data = json.load(f)
                self.norm_meta_cache[meta_file] = meta_data

            with open(zero_norm_path, 'r') as f:
                zero_dict = json.load(f)
            self.norm_meta_cache[zero_norm_path] = zero_dict
        

            # Run Meta Data
            if self.meta:
                self.send_message('Initializing JSON cache and pre-storing files...')
                self.meta_cache: Dict[str, dict] = dict()
                for file in self.file_list:
                    meta_file = self.csv_to_json_table[file]
                    with open(meta_file, 'r') as f:
                        meta_data = json.load(f)
                    meta_data = self.parse_meta_data_dict(meta_data)
                    self.meta_cache[meta_file] = meta_data

                with open(zero_meta_path, 'r') as f:
                    zero_dict = json.load(f)
                zero_meta_data = self.parse_meta_data_dict(zero_dict)
                self.meta_cache[zero_meta_path] = zero_meta_data
            
            self.send_message('Finished initialization of cache.')
            
        self.send_message('Finished initialization.')
    
    def __len__(self):
        ''' returns the number of batches per epoch'''
        return len(self.batches)

    def get_data(self, stack: bool = True) -> Tuple[np.array, np.array]:
        ''' return all data'''
        
        self.send_message('Retrieving all data...')
        collector_X = []
        collector_y = []
        if self.meta:
            collector_X_meta = []

        for i in range(len(self)):
            X, y = self.__getitem__(i)
            if self.meta:
                collector_X.append(X[0])
                collector_X_meta.append(X[1])
            else:
                collector_X.append(X)
            collector_y.append(y)

        if stack:
            np_X = np.vstack(collector_X)
            np_y = np.vstack(collector_y)
            if self.meta:
                np_X_meta = np.vstack(collector_X_meta)
        else:
            np_X = np.array(collector_X)
            np_y = np.array(collector_y)
            if self.meta:
                np_X_meta = np.array(collector_X_meta)

        if self.meta: 
            self.send_message(f'Retrieved all data.     X : {np_X.shape}; X_meta : {np_X_meta.shape}; y : {np_y.shape}')
            return (np_X, np_X_meta), np_y
        else:
            self.send_message(f'Retrieved all data.     X : {np_X.shape}; y : {np_y.shape}')
            return np_X, np_y
    
    def determine_split(self) -> int:
        '''
        determine the point in time to split the data into training and validation
        -> NOTE: leads to a train-test split along the time-axis
        '''
        
        index_count_map: Dict[int, int] = {}
        for entry in self.sample_table:
            _, _, indices = entry
            for index in indices:
                if index in index_count_map.keys():
                    index_count_map[index] += 1
                else:
                    index_count_map[index] = 1
        
        key_list = [key for key in index_count_map.keys()]
        key_list.sort()

        split_key: int = key_list[-1]
        training_count = self.total * self.train_test_split

        curr_total = 0
        for key in key_list:
            if curr_total < training_count:
                curr_total += index_count_map[key]
            else:
                split_key = key
                break

        return split_key
    
    def generate_sample_table(self) -> Tuple[int, List[Tuple[str, int, List[int]]]]:
        '''
        generates the sample table if None is existent,
        else checks whether the table is older than the files,
        if it is a new table is generated
        
        table contains entries of the shape: 
            (file_path, samples, indices of samples)
        '''

        table = []
        total = 0

        if exists(self.table_file) and exists(self.meta_table_file):
            self.send_message(f'Found existing version of table in {self.table_file}')
            self.send_message(f'Found existing meta data file in {self.meta_table_file}')
            last_modified_table = os.stat(self.table_file).st_mtime

            with open(self.meta_table_file, 'r') as f:
                meta_data = json.load(f)
            
            if meta_data == self.generate_table_meta_dict():
                self.send_message(f'Existing meta data file matches current BatchGenerator parameters.')
                
                # gets the timestamp of the most recent modified file in the data set
                most_recently_modified: float = 0
                for file in self.file_list:
                    curr_file_last_modified = os.stat(file).st_mtime
                    if curr_file_last_modified > most_recently_modified:
                        most_recently_modified = curr_file_last_modified

                # if the files are older than the table, load the table and return it
                if most_recently_modified < last_modified_table:
                    with open(self.table_file, 'rb') as f:
                        table = pickle.load(f)

                    for entry in table:
                        _, samples, _ = entry
                        total += samples
                    self.send_message(f'Found and loaded current version of table indicating {total} samples across {len(table)} files in {self.table_file}')
                    return total, table
                
                # else do not return, keep in function to reload table
                else:
                    self.send_message(f'Found an legacy version of table. Recreating the table.')
 
            else:
                self.send_message(f'Meta data at {self.meta_table_file} does not match current BatchGenerator parameters ...')

        else:
            if not(exists(self.table_file)):
                self.send_message(f'Found no version of table at {self.table_file} ...')
            if not(exists(self.meta_table_file)):
                self.send_message(f'Found no meta data at {self.meta_table_file} ...')
        
        # as the sample table could not be loaded, the table has to be (re-)generated
        self.send_message(f'Generating new sample table using {PROCESSOR_CORES} sub-processes ...')

        sections = [self.file_list[i * math.floor(len(self.file_list) / PROCESSOR_CORES) : (i + 1) * math.floor(len(self.file_list) / PROCESSOR_CORES)] for i in range(0, PROCESSOR_CORES - 1)]
        sections.append(self.file_list[(PROCESSOR_CORES - 1) * math.floor(len(self.file_list) / PROCESSOR_CORES) :])
        
        args = [(i, sections[i]) for i in range(PROCESSOR_CORES)]
        function = self.generate_partial_table

        with ProcessPoolExecutor(PROCESSOR_CORES) as ex:
            results = ex.map(function, args)

        for partial_total, partial_table in results:
            total += partial_total
            table = [*table, *partial_table]
        
        self.send_message('Finished generating the sample table.', leading='\n')
        self.send_message(f'Found {total} samples in {len(self.file_list)} files.')

        with open(self.table_file, 'wb') as f:
            pickle.dump(table, file=f)
        with open(self.meta_table_file, 'w') as f:
            meta_data = self.generate_table_meta_dict()
            json.dump(meta_data, fp=f)
        self.send_message(f'Saved newly generated table to {self.table_file}.')
        self.send_message(f'Saved corresponding meta data to {self.meta_table_file}.')

        return total, table
    
    def parse_meta_data_dict(self, meta_data: dict) -> dict:
        """parses a dictionary from a JSON file and returns a dictionary
        containing the relevant fields only

        Args:
            meta_data (dict): meta data dictionary from meta data JSON file

        Returns:
            dict: meta data dictionary with relevant fields
        """
        keys = list(meta_data.keys())

        for unwanted_key in ['RightFootActivity', 'LeftFootActivity', 'Subject', 'Trial', 'Code', 'SensorLocation']:
            if unwanted_key in keys:
                keys.remove(unwanted_key)
                
        partial_meta_dict = get_partial_dict(meta_data, keys)

        return partial_meta_dict
    
    def encode_meta_data_dict(self, meta_data: dict) -> list:
        """encodes a meta data dictionary using the ENCODING_DICT to create a 
        linear scalar represenation

        Args:
            meta_data (dict): meta data dictionary

        Returns:
            list: linear, encoded representation
        """
        
        encoding = []

        for key in meta_data:
            value = meta_data[key]
            if key in ENCODING_DICT:
                value = ENCODING_DICT[key][value]
            else:
                # replace potential 'NC' value
                if value == MISSING_VALUE:
                    value = MISSING_VALUES_DICT[key]
                if key in ['Age', 'Height', 'Weight', 'BMI', 'WalkedDistance', 'WalkingSpeed']:
                    value = min_max_scaling_key(value, key)

            if type(value) == list:
                for v in value:
                    encoding.append(v)
            else:
                encoding.append(value)
        
        return encoding

    def generate_table_meta_dict(self):
        meta = {}
        meta_keys = ['batch_size', 'shuffle', 'length', 'sample_freq', 'sample_offset', 'index_file', 'table_file', 'padding', 'file_list', 'train_test_split']
        for key in meta_keys:
            meta[key] = vars(self)[key]
        return meta

    def generate_partial_table(self, args) -> Tuple[int, List[Tuple[str, int, List[int]]]]:
        id, file_list = args
        total = 0
        table = []

        if id == 0  and self.verbose:
            bar =  CustomBar('Generating sample table', max=len(file_list) * PROCESSOR_CORES)

        for path in file_list:
            df = pd.read_csv(path, dtype=self.dtype,index_col=False)
            if self.padding:
                df = bgf.pad_df(df=df, sample_offset=self.sample_offset, length=self.length, sample_freq=self.sample_freq)
            
            sample_indices: List[int] = []
            index = self.sample_offset

            while index + self.length - 1 <= df.shape[0] - 1:
                # a slice of a data frame ranges from min=index and max=index+LENGTH-1
                # this imposes the constraint that index+LENGTH-1 must be less or equal
                # to the last index
                sample_indices.append(index)
                index += self.sample_freq

            total += len(sample_indices)
            if total > 0:
                table.append((path, len(sample_indices), sample_indices))
            
            if id == 0 and self.verbose:
                bar.next(n=PROCESSOR_CORES)
        
        return total, table

    def send_message(self, message: str, **kwargs) -> None:
        if self.verbose:
            printc(source='[BatchGenerator]', message=message, **kwargs)
    
    def group_data_by_batch(self) -> List[List[Tuple[str, int, List[int]]]]:

        '''
        create a list of batch information sets, that contains the files,
        and the indices as well as the number of batches to get from each file
        NOTE: Why is there a second list around the list wrapping the tuples?
        -> List[batch, batch, ...], where one batch may consist of one or more tuples

        Returns:
            _type_: batches - List[
                List[
                    Tuple[
                        str : The path of the file,
                        int : The number of batches in this file, 
                        List[Int]: A list of starting indices for the batches of the file
                    ]
                ]
            ]
        '''

        specifier = 'shuffled ' if self.shuffle else ' '
        self.send_message(f'Generating batch information for {specifier}batches from the data ...')
        
        batches: List[List[Tuple[str, int, List[int]]]] = []
        incomplete_batch: List[List[Tuple[str, int, List[int]]]] = []

        # NOTE: new shuffle mechanism:
        # the sample table entries are regrouped into separate entries (chunks) of 
        # at max the [chunk_size] indices per entry; after shuffling the files
        # and the entries are mixed while retaining at least [chunk_size] samples
        # in one batch

        chunk_table = []
        if self.shuffle:
            for entry in self.sample_table:
                file, count, indices = entry
                while len(indices) > self.chunk_size:
                    chunk_entry = (file, self.chunk_size, indices[:self.chunk_size])
                    chunk_table.append(chunk_entry)
                    indices = indices[self.chunk_size:]
                chunk_entry = (file, len(indices), indices)
                chunk_table.append(chunk_entry)
            
            random.shuffle(chunk_table)

        current_table = self.sample_table if not(self.shuffle) else chunk_table
        for entry in current_table:
            file, _, indices = entry

            filtered_indices: List[int] = []
            for index in indices:
                index = int(index)
                if self.validation_generator:
                    if self.train_test_split == 0.0:
                        if index >= self.split_stamp:
                            filtered_indices.append(index)
                    else:
                        if index > self.split_stamp:
                            filtered_indices.append(index)
                if not(self.validation_generator):
                    if self.train_test_split == 1.0:
                        if index <= self.split_stamp:
                            filtered_indices.append(index)
                    else:
                        if index < self.split_stamp:
                            filtered_indices.append(index)
            indices = filtered_indices

            if len(incomplete_batch) > 0:
                incomplete_length = sum([length for (_, length, _) in incomplete_batch])
                if len(indices) + incomplete_length < self.batch_size:
                    if len(indices) > 0:
                        incomplete_batch.append((file, len(indices), indices))
                        continue
                else:
                    remaining = self.batch_size - incomplete_length
                    selected_indices = indices[:remaining]
                    indices = indices[remaining:]

                    batch = [*incomplete_batch, (file, len(selected_indices), selected_indices)]
                    batches.append(batch)
                    incomplete_batch = []

            while len(indices) >= self.batch_size:
                selected_indices = indices[:self.batch_size]
                indices = indices[self.batch_size:]

                batch = [(file, self.batch_size, selected_indices)]
                batches.append(batch)

            if len(indices) > 0:
                incomplete_batch = [(file, len(indices), indices)]

        if self.pad_last_batch and len(incomplete_batch) > 0:
            # add empty samples with empty labels to allow
            # for all data to be included into the generated batches

            zero_path = f'{DATA_PATH}\\zero.xsv' if PLATFORM == 'WINDOWS' else f'{DATA_PATH}/zero.xsv'
            remaining_samples = 0
            for _, count, _ in incomplete_batch:
                remaining_samples += count

            indices = [0 for _ in range(self.batch_size - remaining_samples)]
            batch = [*incomplete_batch, (zero_path, len(indices), indices)]
            batches.append(batch)

        self.send_message(f'Generated {len(batches)} batches from the data ...')

        return batches

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:
        '''
        generate one batch of data given the index
        '''
        
        batch_information = self.batches[index]
        
        collector_X = []
        collector_y = []
        if self.meta:
            collector_X_meta = []

        for piece in batch_information:
            partial_result_X, partial_result_y = self.resolve_batch_information(piece)

            if self.meta:
                collector_X.append(partial_result_X[0])
                collector_X_meta.append(partial_result_X[1])
            else:
                collector_X.append(partial_result_X)
            collector_y.append(partial_result_y)

        X = np.vstack(collector_X)
        y = np.vstack(collector_y)
        if self.meta:
            #print(collector_X_meta[0][0])
            #print(collector_X_meta[1][0])

            X_meta = np.vstack(collector_X_meta)

        if self.augmenter != None and not(self.validation_generator):
            X, y = self.augmenter.augment(X, y)
        
        if self.meta:
            return (X, X_meta), y
        return X, y
    


    def resolve_batch_information(self, batch_information: Tuple[str, int, List[int]]) -> np.ndarray:
        '''
        resolves batch information
        '''

        file, _, indices = batch_information
            
        if self.caching:
            df = self.cache[file]
        else:
            df = pd.read_csv(file, usecols=[*self.data_columns, *self.label_columns], dtype=self.dtype,index_col=False)
            if self.padding:
                df = bgf.pad_df(df=df,sample_offset= self.sample_offset,length=self.length,sample_freq=self.sample_freq)
        
        # normalization meta data
        norm_meta_file = self.csv_to_norm_meta_table[file]
        if self.caching:
            norm_meta_data = self.norm_meta_cache[norm_meta_file]
        else:
            with open(norm_meta_file, 'r') as f:
                norm_meta_data = json.load(f)
        
        
        # print(f'File: {file}, Norm meta file: {norm_meta_file}')

        # Run meta  data
        if self.meta:
            meta_file = self.csv_to_json_table[file]
            if self.caching:
                meta_data = self.meta_cache[meta_file]
            else:
                with open(meta_file, 'r') as f:
                    meta_data = json.load(f)
                    meta_data = self.parse_meta_data_dict(meta_data)
                    
            encoded_meta_data = self.encode_meta_data_dict(meta_data)

        collector_X = []
        collector_y = []
        if self.meta:
            collector_X_meta = [np.array(encoded_meta_data) for _ in indices]
        
        for index in indices:
            X: pd.DataFrame = df.loc[index:index+self.length - 1]
            X = X[self.data_columns]

            y: pd.DataFrame = df.loc[index:index+self.length - 1]
            y = y[self.label_columns]

            np_X = np.array(X)
            np_y = np.array(y)

            np_X, np_y =  apply_normalization(
                np_X, np_y,
                norm_meta_data,
                data_columns = self.data_columns, 
                normalization_function_name=self.normalization_function
            )

            collector_X.append(np_X)
            collector_y.append(np_y)
        
        if self.meta:
            return (np.array(collector_X), np.array(collector_X_meta)), np.array(collector_y)
        return np.array(collector_X), np.array(collector_y)
    
    def on_epoch_end(self):
        '''
        updates the batch information once, if no shuffle 
        after the end of every epoch
        '''
        if self.shuffle == True or self.batches == None:
            self.batches = self.group_data_by_batch()

