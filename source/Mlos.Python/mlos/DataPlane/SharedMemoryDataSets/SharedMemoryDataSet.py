#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import json
from multiprocessing.shared_memory import SharedMemory
from typing import List, Tuple
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from mlos.DataPlane.Interfaces.DataSet import DataSet
from mlos.Spaces import CategoricalDimension, Hypergrid
from mlos.Spaces.HypergridsJsonEncoderDecoder import HypergridJsonEncoder

from mlos.DataPlane.SharedMemoryDataSets.SharedMemoryDataSetInfo import SharedMemoryDataSetInfo
from mlos.DataPlane.SharedMemoryDataSets.SharedMemoryDataSetView import SharedMemoryDataSetView


class SharedMemoryDataSet(DataSet):
    """Maintains a dataframe and its associated metadata in shared memory.

    """

    def __init__(
        self,
        data_set_id: UUID = None,
        column_names: List[str] = None,
        schema: Hypergrid = None,
        shared_memory_np_array_nbytes: int = None,
        shared_memory_np_array_shape: Tuple[int, int] = None,
        shared_memory_np_array_dtype: np.dtype = None
    ):
        assert (column_names is not None) or (schema is not None), "Either column_names or schema must be provided."

        if column_names is None:
            column_names = schema.dimension_names
        self.column_names = column_names

        self._schema = schema
        if schema is not None:
            # Let's make sure that all categorical dimensions are numeric.
            # TODO: make it part of the Dimension interface
            for dimension in schema.dimensions:
                if isinstance(dimension, CategoricalDimension):
                    assert dimension.is_numeric

        if data_set_id is None:
            data_set_id = uuid4()
        self.data_set_id = data_set_id
        self._shared_memory = None
        self._df: pd.DataFrame = None

        self._shared_memory_np_array_nbytes = shared_memory_np_array_nbytes
        self._shared_memory_np_array_shape = shared_memory_np_array_shape
        self._shared_memory_np_array_dtype = shared_memory_np_array_dtype

    @property
    def schema(self) -> Hypergrid:
        return self._schema

    def unlink(self):
        if self._shared_memory is not None:
            self._shared_memory.unlink()
            self._shared_memory = None

    def map(self):
        if self._shared_memory is not None:
            self._shared_memory = SharedMemory(name=str(self.data_set_id), create=False)

    def get_data_set_info(self) -> SharedMemoryDataSetInfo:
        return SharedMemoryDataSetInfo(
            column_names=self.column_names,
            schema=self.schema,
            shared_memory_np_array_nbytes=self._shared_memory_np_array_nbytes,
            shared_memory_np_array_shape=self._shared_memory_np_array_shape,
            shared_memory_np_array_dtype=self._shared_memory_np_array_dtype,
            data_set_id=self.data_set_id
        )

    def get_dataframe(self) -> pd.DataFrame:
        if self._shared_memory is None:
            self._shared_memory = SharedMemory(name=str(self.data_set_id), create=False)

        shared_memory_np_records_array = np.recarray(
            shape=self._shared_memory_np_array_shape,
            dtype=self._shared_memory_np_array_dtype,
            buf=self._shared_memory.buf
        )
        df = pd.DataFrame.from_records(data=shared_memory_np_records_array, columns=self.schema.dimension_names, index='index')
        return df

    def set_dataframe(self, df: pd.DataFrame) -> None:
        if self.schema is not None:
            assert df in self.schema
        self._df = df

        # Now let's put it in shared memory. We can attempt two ways:
        #   1. using the to_numpy call on the entire df - causes a few copies
        #   2. trying the to_numpy call on each column separately - not tested yet, but should result in fewer copies down the road
        #
        # Both options should allow us to preallocate a bunch more memory than needed so that appending new rows doesn't require
        # copying all the old ones. This really only applies to training observations, as dataframes produced by utility functions
        # don't grow. On the other hand, utility functions regularly pose dataframes of the same shape to the models, so it would
        # be good to reuse memory once allocated instead of allocating new memory blocks for each
        #
        # So let's start with the first one to get it working and we will unlink and create new shared memory every time. We can
        # optimize all of this later.

        if self._shared_memory is not None:
            # TODO: put a lock around this just to be sure!
            #
            # For now let's unlink any previous version of this dataframe - this should return the memory to the allocator. Later: reuse!
            self.unlink()

        np_records_array = df.to_records(index=True)

        self._shared_memory_np_array_nbytes = np_records_array.nbytes
        self._shared_memory_np_array_shape = np_records_array.shape
        self._shared_memory_np_array_dtype = np_records_array.dtype
        self._shared_memory = SharedMemory(name=str(self.data_set_id), create=True, size=self._shared_memory_np_array_nbytes)
        shared_memory_np_array = np.recarray(shape=self._shared_memory_np_array_shape, dtype=self._shared_memory_np_array_dtype, buf=self._shared_memory.buf)
        np.copyto(dst=shared_memory_np_array, src=np_records_array)

    def validate(self) -> None:
        # Validates that the dataframe in the shared memory is an exact copy of the dataframe in cache.
        # This is useful to ensure that none of the clients is accidentally modifying the df.
        #
        data_set_view = SharedMemoryDataSetView(data_set_info=self.get_data_set_info())
        df = data_set_view.get_dataframe()
        assert df.equals(self._df)
