#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from typing import Dict
from uuid import UUID

import pandas as pd

from mlos.DataPlane.Interfaces import DataSetInfo, DataSetStore
from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSetInfo, SharedMemoryDataSet, SharedMemoryDataSetView

class SharedMemoryDataSetStore(DataSetStore):
    """Provides functionality to create, view and delte DataSet instances in shared memory.

    """

    def __init__(self):
        self._data_sets_by_id: Dict[UUID, SharedMemoryDataSet] = dict()

    def create_data_set(self, data_set_info: DataSetInfo, df: pd.DataFrame) -> SharedMemoryDataSet:
        data_set = SharedMemoryDataSet(schema=data_set_info.schema, data_set_id=data_set_info.data_set_id)
        data_set.set_dataframe(df=df)
        self._data_sets_by_id[data_set_info.data_set_id] = data_set
        return data_set

    def add_data_set(self, data_set: SharedMemoryDataSet) -> None:
        ...

    def connect_to_data_set(self, data_set_info: SharedMemoryDataSetInfo) -> None:
        if data_set_info.data_set_id in self._data_sets_by_id:
            return

        data_set = SharedMemoryDataSet(
            data_set_id=data_set_info.data_set_id,
            schema=data_set_info.schema,
            shared_memory_np_array_nbytes=data_set_info.shared_memory_np_array_nbytes,
            shared_memory_np_array_shape=data_set_info.shared_memory_np_array_shape,
            shared_memory_np_array_dtype=data_set_info.shared_memory_np_array_dtype
        )
        data_set.map()
        self._data_sets_by_id[data_set_info.data_set_id] = data_set

    def get_data_set(self, data_set_info: DataSetInfo) -> SharedMemoryDataSet:
        ...

    def get_data_set_view(self, data_set_info: DataSetInfo) -> SharedMemoryDataSetView:
        ...

    def remove_data_set(self, data_set_info: DataSetInfo) -> None:
        ...
