#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing import Queue, Event
from multiprocessing.shared_memory import SharedMemory
import os
import pickle
from queue import Empty
from typing import Dict

import pandas as pd

from mlos.DataPlane.ModelHosting.ModelHostMessages import PredictRequest
from mlos.DataPlane.SharedMemoryDataSetInfo import SharedMemoryDataSetInfo
from mlos.DataPlane.SharedMemoryDataSetView import SharedMemoryDataSetView
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel

class SharedMemoryModelHost:
    """A worker responsible for scoring models held in shared memory.

    Objective:
        The goal is to parallelize model scoring (calls to model.predict(...)) so as to have many trees evaluating simultaneously
        in shared memory. This should provide a considerable speed boost for producing suggestions, tomograph interactions, and
        lastly model training.


    Strategy:
        Each instance of shared memory model executor will listen for requests on a queue. Once a request object is received,
        the executor will check its cache to see if it already has the requisite model deserialized, if it doesn't, it will locate
        the model in shared memory and deserialize it.

        It will then locate the features_df in shared memory and invoke the model's predict() method. Lastly, it will stash the
        prediction's dataframe in shared memory and send back the message informing the client how to find it.

        Finally, once all of the above works, we can extend the Host to also fit the models and place them in shared memory.


    Additional Notes:
        Tracing shows that currently, serial execution of models is one of the biggest bottlenecks in both registering an observation,
        and - more importantly - producing a suggestion. Parallelizing inference is the single biggest low-hanging fruit we have.

        Since Python is not big on multi-threading (GIL is here to stay), the preferred parallelization method is to use multiple
        processes. This creates the question of inter-process communication. We have a ton of options:
            * shared memory
            * pipes
            * sockets
            * ...
        But only shared memory lets us avoid unnecessary copying and serializing/deserializing of our (relatively large) dataframes
        and models. So we will leverage Python's multiprocessing.SharedMemory module to stash our data and models in shared memory.

    """

    def __init__(
        self,
        request_queue: Queue,
        response_queue: Queue,
        shutdown_event: Event

    ):
        self.request_queue: Queue = request_queue
        self.response_queue: Queue = response_queue
        self.shutdown_event: Event = shutdown_event
        self._model_cache: Dict[str, RegressionModel] = dict()

    def run(self):
        print(f'{os.getpid()} running')
        timeout_s = 5
        while not self.shutdown_event.is_set():
            try:
                request = self.request_queue.get(block=True, timeout=timeout_s)
            except Empty:
                continue

            if isinstance(request, PredictRequest):
                print(f"{os.getpid()} Got request of type: {type(request)}")
                self.process_predict_request(request)
            else:
                raise RuntimeError(f"Unknown request type: {type(request)}")

        print(f"{os.getpid()} shutting down")

    def process_predict_request(self, request: PredictRequest):
        if request.model_id in self._model_cache:
            model = self._model_cache[request.model_id]
        else:
            model_shared_memory = SharedMemory(name=request.model_id, create=False)
            model = pickle.loads(model_shared_memory.buf)
            assert isinstance(model, RegressionModel)
            self._model_cache[request.model_id] = model

        features_data_set_view = SharedMemoryDataSetView(data_set_info=request.data_set_info)
        features_df = features_data_set_view.get_dataframe()
        pd.set_option('max_columns', None)
        print(features_df)

        prediction = model.predict(feature_values_pandas_frame=features_df, include_only_valid_rows=True)
        print(prediction.get_dataframe())

        # TODO: put predictions in shared memory and send the response.
        features_data_set_view.detach()




def start_shared_memory_model_host(request_queue: Queue, response_queue: Queue, shutdown_event: Event):
    model_host = SharedMemoryModelHost(request_queue=request_queue, response_queue=response_queue, shutdown_event=shutdown_event)
    model_host.run()
