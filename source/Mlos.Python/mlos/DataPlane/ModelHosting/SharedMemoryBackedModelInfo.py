#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#

class SharedMemoryBackedModelInfo:
    """Maintains all information required to retrieve a regression model from shared memory."""

    def __init__(self, shared_memory_name: str):
        self.model_id: str = shared_memory_name
