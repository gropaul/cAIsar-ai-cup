from typing import Any, List

class Error(Exception):
    '''
    Base class for other custom errors
    '''
    pass


class DataFilesNotFoundError(Error):
    '''
    thrown if files are missing
    '''

    def __init__(self, files: List[str], *args: object) -> None:
        self.message = f'The following files could not be found:\n'
        for file in files:
            self.message += f'\t\t{file}\n'
        super().__init__(self.message, *args)


class NoSuchModelParametersError(Error):
    '''
    thrown if no such Model Parameters exist
    '''

    def __init__(self, model_class: str, model_class_params: str, *args: object) -> None:
        self.message = f'For the model class {model_class} no parameter configuration \
            {model_class}Params.{model_class_params} could be found.'
        super().__init__(self.message, *args)


class UnmatchedLossParamError(Error):
    '''
    thrown if an uneven number of parameters is encountered
    as CLI loss function params need to be in the order
    [[keyword] [value] ...]
    '''
    
    def __init__(self, params: List[Any], *args: object) -> None:
        self.message = f'The following parameter combination is invalid. {params}'
        super().__init__(self.message, *args)


class UnknownArchitectureError(Error):
    '''
    thrown if an unknown architecture is specified for the UeberNet
    '''
    
    def __init__(self, architecture: str, *args: object) -> None:
        self.message = f'The following architecture is unknown. \'{architecture}'
        super().__init__(self.message, *args)


class TooFewFiltersError(Error):
    '''
    thrown if the filter size is not compatible with the architecture
    '''
    
    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)


class WrongDataTypeInSearchSpaceError(Error):
    '''
    thrown if the data type of a variable in a config derived from a search space
    has an unexpected data type.
    '''

    def __init__(self, variable: str, value: Any, data_type: Any, expected_types: List[Any], *args: object) -> None:
        self.message = f'The variable {variable} contaiining the value {value} has the data type {data_type} which is not one of the expected types: {expected_types}'
        super().__init__(self.message, *args)