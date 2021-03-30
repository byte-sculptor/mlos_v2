#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing import connection

import pandas as pd

from mlos.DataPlane.Interfaces import DataSet, DataSetStore, DataSetInfo
from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSet, SharedMemoryDataSetStore, SharedMemoryDataSetView
from mlos.DataPlane.SharedMemoryDataSets.Messages import Request, TakeDataSetOwnershipRequest, UnlinkDataSetRequest

class SharedMemoryDataSetStoreProxy(DataSetStore):
    """SharedMemoryDataSetStore clients use this proxy to access and manipulate the DataSets.

    """
    def __init__(self, service_connection: connection):

        # A connection to communicate with the authoritative DataSetStore.
        #
        self._service_connection: connection = service_connection

        # A local data store to create and access data sets.
        #
        self._data_set_store = SharedMemoryDataSetStore()

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

    def create_data_set(self, data_set_info: DataSetInfo, df: pd.DataFrame) -> DataSet:
        """Creates a DataSet and ensures that the DataStoreService takes ownership of it."""
        data_set = self._data_set_store.create_data_set(data_set_info=data_set_info, df=df)
        shared_memory_data_set_info = data_set.get_data_set_info()

        # Let's request that the service maps this dataset into its own memory.
        #
        request = TakeDataSetOwnershipRequest(data_set_info=shared_memory_data_set_info)
        _ = self._send_request_and_get_response(request=request)

        # By the time we get the response we know that the service now holds a copy to the
        # data set in shared memory.
        return data_set

    def add_data_set(self, data_set: SharedMemoryDataSet) -> None:
        self._data_set_store.add_data_set(data_set=data_set)

    def get_data_set(self, data_set_info: DataSetInfo) -> DataSet:
        return self._data_set_store.get_data_set(data_set_info=data_set_info)

    def get_data_set_view(self, data_set_info: DataSetInfo) -> SharedMemoryDataSetView:
        return self._data_set_store.get_data_set_view(data_set_info=data_set_info)

    def detach_data_set(self, data_set_info: DataSetInfo) -> None:
        self._data_set_store.detach_data_set(data_set_info)

    def unlink_data_set(self, data_set_info: DataSetInfo):
        request = UnlinkDataSetRequest(data_set_info=data_set_info)
        _ = self._send_request_and_get_response(request=request)
        return
