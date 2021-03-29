#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from typing import List, Tuple
import numpy as np

class SharedMemoryDataSetInfo:
    """Maintains all information required to connect to this data set and read its data."""

    def __init__(
        self,
        column_names: List[str],
        schema_json_str: str,
        shared_memory_name: str,
        shared_memory_np_array_nbytes: int,
        shared_memory_np_array_shape: Tuple[int, int],
        shared_memory_np_array_dtype: np.dtype
    ):
        self.column_names = column_names
        self.schema_json_str = schema_json_str
        self.shared_memory_name = shared_memory_name
        self.shared_memory_np_array_nbytes = shared_memory_np_array_nbytes
        self.shared_memory_np_array_shape = shared_memory_np_array_shape
        self.shared_memory_np_array_dtype = shared_memory_np_array_dtype
