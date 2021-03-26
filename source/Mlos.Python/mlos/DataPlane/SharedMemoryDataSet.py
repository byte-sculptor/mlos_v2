#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import json
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple

import numpy as np
import pandas as pd

from mlos.DataPlane.DataSetInterface import DataSetInterface
from mlos.Spaces import CategoricalDimension, Hypergrid
from mlos.Spaces.HypergridsJsonEncoderDecoder import HypergridJsonDecoder, HypergridJsonEncoder

class SharedMemoryDataSetInfo:
    """Maintains all information required to connect to this data set and read its data."""

    def __init__(
        self,
        schema: Hypergrid,
        shared_memory_name: str,
        shared_memory_np_array_nbytes: int,
        shared_memory_np_array_shape: Tuple[int, int],
        shared_memory_np_array_dtype: np.dtype
    ):
        self.schema = schema
        self.shared_memory_name = shared_memory_name
        self.shared_memory_np_array_nbytes = shared_memory_np_array_nbytes
        self.shared_memory_np_array_shape = shared_memory_np_array_shape
        self.shared_memory_np_array_dtype = shared_memory_np_array_dtype



class SharedMemoryDataSet(DataSetInterface):
    """Maintains a dataframe and its associated metadata in shared memory.

    """

    def __init__(
        self,
        schema: Hypergrid,
        shared_memory_name: str,
        shared_memory_np_array_nbytes: int = None,
        shared_memory_np_array_shape: Tuple[int, int] = None,
        shared_memory_np_array_dtype: np.dtype = None
    ):

        # Let's make sure that all categorical dimensions are numeric.
        # TODO: make it part of the Dimension interface
        for dimension in schema.dimensions:
            if isinstance(dimension, CategoricalDimension):
                assert dimension.is_numeric

        self.schema = schema
        self._shared_memory_name = shared_memory_name
        self._shared_memory = None
        self._df: pd.DataFrame = None

        self._shared_memory_np_array_nbytes = shared_memory_np_array_nbytes
        self._shared_memory_np_array_shape = shared_memory_np_array_shape
        self._shared_memory_np_array_dtype = shared_memory_np_array_dtype

    @classmethod
    def create_from_shared_memory_data_set_info(cls, data_set_info: SharedMemoryDataSetInfo):
        # TODO: move this to a DataSetView
        #
        shared_memory_data_set = SharedMemoryDataSet(
            schema=json.loads(data_set_info.schema, cls=HypergridJsonDecoder),
            shared_memory_name=data_set_info.shared_memory_name,
            shared_memory_np_array_nbytes=data_set_info.shared_memory_np_array_nbytes,
            shared_memory_np_array_shape=data_set_info.shared_memory_np_array_shape,
            shared_memory_np_array_dtype=data_set_info.shared_memory_np_array_dtype
        )
        return shared_memory_data_set

    def get_data_set_info(self):
        return SharedMemoryDataSetInfo(
            schema=json.dumps(self.schema, cls=HypergridJsonEncoder),
            shared_memory_name=self._shared_memory_name,
            shared_memory_np_array_nbytes=self._shared_memory_np_array_nbytes,
            shared_memory_np_array_shape=self._shared_memory_np_array_shape,
            shared_memory_np_array_dtype=self._shared_memory_np_array_dtype
        )

    def get_dataframe(self):
        if self._shared_memory is None:
            self._shared_memory = SharedMemory(name=self._shared_memory_name, create=False)

        shared_memory_np_records_array = np.recarray(
            shape=self._shared_memory_np_array_shape,
            dtype=self._shared_memory_np_array_dtype,
            buf=self._shared_memory.buf
        )
        df = pd.DataFrame.from_records(data=shared_memory_np_records_array, columns=self.schema.dimension_names, index='index')
        return df


    def set_dataframe(self, df: pd.DataFrame):
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
            self._shared_memory.unlink()

        np_records_array = df.to_records(index=True)

        self._shared_memory_np_array_nbytes = np_records_array.nbytes
        self._shared_memory_np_array_shape = np_records_array.shape
        self._shared_memory_np_array_dtype = np_records_array.dtype
        self._shared_memory = SharedMemory(name=self._shared_memory_name, create=True, size=self._shared_memory_np_array_nbytes)
        shared_memory_np_array = np.recarray(shape=self._shared_memory_np_array_shape, dtype=self._shared_memory_np_array_dtype, buf=self._shared_memory.buf)
        np.copyto(dst=shared_memory_np_array, src=np_records_array)









