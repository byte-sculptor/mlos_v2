#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import json
from typing import List, Tuple
from uuid import UUID, uuid4

import numpy as np

from mlos.DataPlane.Interfaces import DataSetInfo
from mlos.Spaces import Hypergrid
from mlos.Spaces.HypergridsJsonEncoderDecoder import HypergridJsonEncoder

class SharedMemoryDataSetInfo(DataSetInfo):
    """Maintains all information required to connect to this data set and read its data."""

    def __init__(
        self,
        schema: Hypergrid,
        column_names: List[str] = None,
        shared_memory_np_array_nbytes: int = None,
        shared_memory_np_array_shape: Tuple[int, int] = None,
        shared_memory_np_array_dtype: np.dtype = None,
        data_set_id: UUID = None
    ):
        assert (schema is not None) or (column_names is not None), f"{schema}, {column_names}"
        self._schema = schema
        self.column_names = column_names
        self.schema_json_str = json.dumps(schema, cls=HypergridJsonEncoder)
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
    def schema(self) -> Hypergrid:
        return self._schema
