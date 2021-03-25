#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd

from mlos.DataPlane.DataSetInterface import DataSetInterface
from mlos.Spaces import CategoricalDimension, Hypergrid


class SharedMemoryDataSet(DataSetInterface):
    """Maintains a dataframe and its associated metadata in shared memory.

    """

    def __init__(self, schema: Hypergrid, shared_memory_name: str):

        # Let's make sure that all categorical dimensions are numeric.
        # TODO: make it part of the Dimension interface
        for dimension in schema.dimensions:
            if isinstance(dimension, CategoricalDimension):
                assert dimension.is_numeric

        self.schema = schema
        self._shared_memory_name = shared_memory_name
        self._shared_memory = None
        self._source_df: pd.DataFrame = None

        self._shared_memory_np_array_nbytes = None
        self._shared_memory_np_array_shape = None
        self._shared_memory_np_array_dtype = None

    def set_dataframe(self, df: pd.DataFrame):
        assert df in self.schema
        self._source_df = df

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









