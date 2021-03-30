#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from typing import List, Tuple
from uuid import UUID, uuid4

import numpy as np

from mlos.DataPlane.Interfaces import DataSetInfo
from mlos.Spaces import Hypergrid

class SharedMemoryDataSetInfo(DataSetInfo):
    """Maintains all information required to connect to this data set and read its data."""

    def __init__(
        self,
        column_names: List[str],
        schema_json_str: str,
        shared_memory_name: str,
        shared_memory_np_array_nbytes: int,
        shared_memory_np_array_shape: Tuple[int, int],
        shared_memory_np_array_dtype: np.dtype,
        data_set_id: UUID = None
    ):
        self.column_names = column_names
        self.schema_json_str = schema_json_str
        self.shared_memory_name = shared_memory_name
        self.shared_memory_np_array_nbytes = shared_memory_np_array_nbytes
        self.shared_memory_np_array_shape = shared_memory_np_array_shape
        self.shared_memory_np_array_dtype = shared_memory_np_array_dtype

        if data_set_id is None:
            data_set_id = uuid4()
        self._data_set_id = data_set_id

    @property
    def data_set_id(self) -> UUID:
        return self._data_set_id

    @property
    def name(self) -> str:
        return self.shared_memory_name

    @property
    def schema(self) -> Hypergrid:
        # TOOD: decide if we need this...
        return None
