#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from uuid import UUID, uuid4
from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSetInfo

class Request:
    """Base class for all requests."""
    def __init__(self, request_id: UUID = None):
        if request_id is None:
            request_id = uuid4()
        self.request_id = request_id


class Response:
    """Base class for all responses."""
    def __init__(self, request_id: UUID = None, success: bool = True, exception: Exception = None):
        assert (request_id is not None) and isinstance(request_id, UUID)
        self.request_id = request_id
        self.success = success
        self.exception = exception


class CreateDataSetRequest(Request):
    def __init__(self, data_set_info: SharedMemoryDataSetInfo):
        Request.__init__(self)
        self.data_set_info = data_set_info

class CreateDataSetResponse(Response):
    def __init__(self, request_id: UUID, data_set_info: SharedMemoryDataSetInfo):
        Response.__init__(self, request_id=request_id, success=True, exception=None)
        self.data_set_info = data_set_info


class TakeDataSetOwnershipRequest(Request):
    def __init__(self, data_set_info: SharedMemoryDataSetInfo):
        Request.__init__(self)
        self.data_set_info = data_set_info


class UnlinkDataSetRequest(Request):
    def __init__(self, data_set_info: SharedMemoryDataSetInfo):
        Request.__init__(self)
        self.data_set_info = data_set_info
