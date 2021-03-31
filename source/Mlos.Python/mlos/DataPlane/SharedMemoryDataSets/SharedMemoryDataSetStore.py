#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from contextlib import contextmanager
from multiprocessing import RLock
from typing import Dict
from uuid import UUID

import pandas as pd

from mlos.DataPlane.Interfaces import DataSetInfo, DataSetStore
from mlos.Logger import create_logger
from .SharedMemoryDataSet import SharedMemoryDataSet
from .SharedMemoryDataSetInfo import SharedMemoryDataSetInfo
from .SharedMemoryDataSetView import SharedMemoryDataSetView

class SharedMemoryDataSetStore(DataSetStore):
    """Provides functionality to create, view and delte DataSet instances in shared memory.

    """

    def __init__(self, logger=None):
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger

        self._lock = RLock()
        self._data_sets_by_id: Dict[UUID, SharedMemoryDataSet] = dict()

    def get_stats(self) -> pd.DataFrame:
        records = []
        for data_set_id, data_set in self._data_sets_by_id.items():
            record = dict(
                data_set_id=data_set_id,
                n_bytes=data_set.shared_memory_np_array_nbytes,
                schema_name=data_set.schema.name if data_set.schema is not None else ''
            )
            records.append(record)
        return pd.DataFrame(records, columns=['data_set_id', 'n_bytes', 'schema_name'])

    def create_data_set(self, data_set_info: SharedMemoryDataSetInfo, df: pd.DataFrame = None) -> SharedMemoryDataSet:
        data_set = SharedMemoryDataSet(
            schema=data_set_info.schema,
            data_set_id=data_set_info.data_set_id,
            column_names=data_set_info.column_names,
            shared_memory_np_array_nbytes=data_set_info.shared_memory_np_array_nbytes,
            shared_memory_np_array_shape=data_set_info.shared_memory_np_array_shape,
            shared_memory_np_array_dtype=data_set_info.shared_memory_np_array_dtype
        )
        if df is None:
            data_set.create()
        else:
            data_set.set_dataframe(df=df)
        with self._lock:
            self._data_sets_by_id[data_set_info.data_set_id] = data_set
        return data_set

    def add_data_set(self, data_set: SharedMemoryDataSet) -> None:
        with self._lock:
            if data_set.data_set_id in self._data_sets_by_id:
                self.logger.info(f"Data set {data_set.data_set_id} already known.")
                return
            self._data_sets_by_id[data_set.data_set_id] = data_set

    def connect_to_data_set(self, data_set_info: SharedMemoryDataSetInfo) -> SharedMemoryDataSet:
        with self._lock:
            if data_set_info.data_set_id in self._data_sets_by_id:
                self.logger.info(f"Data set {data_set_info.data_set_id} already known.")
                return

            data_set = SharedMemoryDataSet(
                data_set_id=data_set_info.data_set_id,
                schema=data_set_info.schema,
                column_names=data_set_info.column_names,
                shared_memory_np_array_nbytes=data_set_info.shared_memory_np_array_nbytes,
                shared_memory_np_array_shape=data_set_info.shared_memory_np_array_shape,
                shared_memory_np_array_dtype=data_set_info.shared_memory_np_array_dtype
            )
            self.logger.info(f"Connecting to data set {data_set_info.data_set_id}")
            data_set.attach()
            assert data_set.attached
            self._data_sets_by_id[data_set_info.data_set_id] = data_set
            self.logger.info(f"Connected to data set {data_set_info.data_set_id}")
        return data_set

    def get_data_set(self, data_set_info: DataSetInfo) -> SharedMemoryDataSet:
        with self._lock:
            return self._data_sets_by_id[data_set_info.data_set_id]

    def get_data_set_view(self, data_set_info: DataSetInfo) -> SharedMemoryDataSetView:
        # TODO: check if its in this store...
        with self._lock:
            assert data_set_info.data_set_id in self._data_sets_by_id
        data_set_view = SharedMemoryDataSetView(data_set_info=data_set_info)
        return data_set_view

    @contextmanager
    def attached_data_set_view(self, data_set_info: DataSetInfo):
        """Can be used as a context manager to automatically detach the dataset view when done."""
        # TODO: check if its in this store...
        with self._lock:
            assert data_set_info.data_set_id in self._data_sets_by_id
        data_set_view = SharedMemoryDataSetView(data_set_info=data_set_info)
        yield data_set_view
        data_set_view.detach()

    def detach_data_set(self, data_set_info: DataSetInfo) -> None:
        """Removes the reference to the data set."""
        with self._lock:
            assert data_set_info.data_set_id in self._data_sets_by_id
            if data_set_info.data_set_id in self._data_sets_by_id:
                self.logger.info(f"Detaching data set {data_set_info.data_set_id}.")
                data_set = self._data_sets_by_id.pop(data_set_info.data_set_id)
                data_set.detach()
            else:
                self.logger.info(f"Data set {data_set_info.data_set_id} not known.")

    def unlink_data_set(self, data_set_info: DataSetInfo) -> None:
        """Removes the reference to the data_set and deallocates its memory.

        TODO: Add a semaphore here to be sure.
        """
        with self._lock:
            assert data_set_info.data_set_id in self._data_sets_by_id
            if data_set_info.data_set_id in self._data_sets_by_id:
                self.logger.info(f"Unlinking data set: {data_set_info.data_set_id}")
                data_set = self._data_sets_by_id.pop(data_set_info.data_set_id)
                data_set.unlink()
            else:
                self.logger.info(f"Data set: {data_set_info.data_set_id} not known.")


