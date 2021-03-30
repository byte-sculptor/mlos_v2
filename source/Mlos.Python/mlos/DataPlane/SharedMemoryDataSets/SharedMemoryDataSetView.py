#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from contextlib import contextmanager
import json
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd

from mlos.DataPlane.Interfaces import DataSetInfo, DataSetView
from mlos.DataPlane.SharedMemoryDataSets.SharedMemoryDataSetInfo import SharedMemoryDataSetInfo
from mlos.Spaces import Hypergrid
from mlos.Spaces.HypergridsJsonEncoderDecoder import HypergridJsonDecoder


@contextmanager
def attached_data_set_view(data_set_info: SharedMemoryDataSetInfo):
    """Context manager to help with resource cleanup."""
    data_set_view = SharedMemoryDataSetView(data_set_info=data_set_info)
    yield data_set_view
    data_set_view.detach()

class SharedMemoryDataSetView(DataSetView):
    """Allows to view a dataframe stored in shared memory.

    """
    def __init__(self, data_set_info: SharedMemoryDataSetInfo):
        self.data_set_info: SharedMemoryDataSetInfo = data_set_info
        self._schema = data_set_info.schema
        self.column_names = data_set_info.column_names
        self._shared_memory = None

    @property
    def schema(self) -> Hypergrid:
        return self._schema

    def get_data_set_info(self) -> SharedMemoryDataSetInfo:
        return self.data_set_info

    def detach(self):
        if self._shared_memory is not None:
            self._shared_memory.close()
            self._shared_memory = None

    def get_dataframe(self):
        self.detach()
        self._shared_memory = SharedMemory(name=str(self.data_set_info.data_set_id), create=False)
        shared_memory_np_records_array = np.recarray(
            shape=self.data_set_info.shared_memory_np_array_shape,
            dtype=self.data_set_info.shared_memory_np_array_dtype,
            buf=self._shared_memory.buf
        )
        df = pd.DataFrame.from_records(data=shared_memory_np_records_array, columns=self.column_names, index='index')
        return df
