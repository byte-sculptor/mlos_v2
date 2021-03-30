#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import time

from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSetService, SharedMemoryDataSetStoreProxy

class TestSharedMemoryDataSets:
    """Tests if the SharedMemoryDataSets behave as expected.

    """

    def test_transferring_data_set_ownership(self):
        """Tests whether we can successfully transfer a dataset ownership from proxy to service.

        """
        service = SharedMemoryDataSetService()
        proxy_connection = service.get_new_proxy_connection()
        time.sleep(1)
        service.launch()
        time.sleep(5)
        service.stop()

