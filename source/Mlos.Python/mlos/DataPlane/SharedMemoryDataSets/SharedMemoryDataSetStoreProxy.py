#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from contextlib import contextmanager
from multiprocessing import connection

import pandas as pd

from mlos.DataPlane.Interfaces import DataSet, DataSetStore, DataSetInfo
from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSet, SharedMemoryDataSetInfo, SharedMemoryDataSetStore, SharedMemoryDataSetView
from mlos.DataPlane.SharedMemoryDataSets.Messages import Request, UnlinkDataSetRequest, CreateDataSetRequest, CreateDataSetResponse
from mlos.Logger import create_logger

class SharedMemoryDataSetStoreProxy(DataSetStore):
    """SharedMemoryDataSetStore clients use this proxy to access and manipulate the DataSets.

    """
    def __init__(self, service_connection: connection, logger=None):
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger

        # A connection to communicate with the authoritative DataSetStore.
        #
        self._service_connection: connection = service_connection

        # A local data store to create and access data sets.
        #
        self._data_set_store = SharedMemoryDataSetStore()

        # Since we are leaking the data sets anyway thanks to bug https://bugs.python.org/issue40882 in Python,
        # we might as well keep track of their info so that we know what they are and how much memory we have leaked so far.
        # We can use this information to exit this process once the memory leaked goes over some threshold.
        #
        # TODO: remove this entire mechanism once that bug is fixed and on non-Windows platforms.
        self._leaked_data_sets_info = []
        self._total_leaked_bytes = 0

    @property
    def total_leaked_bytes(self):
        return self._total_leaked_bytes

    def _send_request_and_get_response(self, request: Request):
        """Sends a message to the service and waits for the response.

        If response.success is false, raises the exception contained in the response.
        """
        self._service_connection.send(request)
        try:
            # Let's wait for the service to acknowledge that it succeeded.
            #
            response = self._service_connection.recv()
            if not response.success:
                raise response.exception
        except EOFError:
            # TODO: handle it better. For now, let it bubble up.
            raise

        return response

    def create_data_set(self, data_set_info: SharedMemoryDataSetInfo, df: pd.DataFrame) -> DataSet:
        """Requests that the Data Set Service allocates shared memory for us to use.

        The catch is that python has a bug:
            https://bugs.python.org/issue40882

        And an unmerged fix:
            https://github.com/python/cpython/pull/20684/files

        The bug is that when you call SharedMemory.open(..., create=False) Python leaks a handle to the memory mapping and
        consequently will never release that memory. This leads to out of memory errors.

            One of the workarounds (admittedly rather ugly) is to only call that leaky API only from worker processes and killing the
        worker processes either after every request or after n requests. When a worker process is killed, the ref-count on the
        handle will be decremented and Windows can free that memory.

            Consequently, we need the Service process to allocate the SharedMemory and a worker process to connect to it.
            To do that, we need the worker process to compute the size of the memory buffer required.
        """

        np_records_array = df.to_records(index=True)


        shared_memory_data_set_info = SharedMemoryDataSetInfo(
            schema=data_set_info.schema,
            column_names=data_set_info.column_names,
            data_set_id=data_set_info.data_set_id,
            shared_memory_np_array_nbytes=np_records_array.nbytes,
            shared_memory_np_array_shape=np_records_array.shape,
            shared_memory_np_array_dtype=np_records_array.dtype
        )

        request = CreateDataSetRequest(data_set_info=shared_memory_data_set_info)
        response = self._send_request_and_get_response(request=request)
        assert response.success
        assert response.data_set_info.data_set_id == shared_memory_data_set_info.data_set_id
        data_set = self._data_set_store.connect_to_data_set(data_set_info=response.data_set_info)
        data_set.set_dataframe(np_records_array=np_records_array)
        return data_set

    def add_data_set(self, data_set: SharedMemoryDataSet) -> None:
        self._data_set_store.add_data_set(data_set=data_set)

    def get_data_set(self, data_set_info: DataSetInfo) -> DataSet:
        return self._data_set_store.get_data_set(data_set_info=data_set_info)

    def get_data_set_view(self, data_set_info: DataSetInfo) -> SharedMemoryDataSetView:
        return self._data_set_store.get_data_set_view(data_set_info=data_set_info)

    def detach_data_set(self, data_set_info: SharedMemoryDataSetInfo) -> None:
        self._data_set_store.detach_data_set(data_set_info)
        self._leaked_data_sets_info.append(data_set_info)
        self._total_leaked_bytes += data_set_info.shared_memory_np_array_nbytes

    def unlink_data_set(self, data_set_info: SharedMemoryDataSetInfo):
        request = UnlinkDataSetRequest(data_set_info=data_set_info)
        _ = self._send_request_and_get_response(request=request)
        self._leaked_data_sets_info.append(data_set_info)
        self._total_leaked_bytes += data_set_info.shared_memory_np_array_nbytes
        return

    @contextmanager
    def attached_data_set_view(self, data_set_info: DataSetInfo):
        data_set_view = self.get_data_set_view(data_set_info=data_set_info)
        yield data_set_view
        data_set_view.detach()
