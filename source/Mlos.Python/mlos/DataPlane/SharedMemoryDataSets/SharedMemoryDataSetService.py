#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from contextlib import contextmanager
from functools import wraps
from multiprocessing import connection, Event, Pipe, RLock
from threading import Thread
from typing import List

from mlos.Logger import create_logger

from .SharedMemoryDataSetStore import SharedMemoryDataSetStore
from .SharedMemoryDataSetStoreProxy import SharedMemoryDataSetStoreProxy
from .Messages import Response, Request, TakeDataSetOwnershipRequest, UnlinkDataSetRequest, CreateDataSetRequest, CreateDataSetResponse


def request_handler():
    def request_handler_decorator(wrapped_function):
        @wraps(wrapped_function)
        def wrapper(*args, **kwargs):
            request_id = 0
            try:
                request = kwargs['request']
                request_id = request.request_id

                return wrapped_function(*args, **kwargs)
            except Exception as e:
                self = args[0]
                self.logger.error(f"Failed to process request:  {request_id}", exc_info=True)
                return Response(request_id=request_id, success=False, exception=e)

        return wrapper
    return request_handler_decorator


class SharedMemoryDataSetService:
    """Maintains an authoritative SharedMemoryDataSetStore and allows proxies to access and manipulate it.

    """

    def __init__(self, logger=None):
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger
        self._proxy_connections_lock = RLock()
        self.data_set_store: SharedMemoryDataSetStore = SharedMemoryDataSetStore()
        self._proxy_connections: List[connection] = []
        # TODO: maybe have several threads here as these requests are often blocking on syscalls
        #
        self._service_thread: Thread = None
        self._shutdown_event: Event = Event()

    def get_new_proxy_connection(self):
        client_connection, service_connection = Pipe()
        self._proxy_connections.append(client_connection)
        return service_connection

    def get_new_proxy(self):
        proxy_connection = self.get_new_proxy_connection()
        proxy = SharedMemoryDataSetStoreProxy(service_connection=proxy_connection)
        return proxy

    def launch(self):
        self._service_thread = Thread(target=self._serve, args=())
        self._service_thread.daemon = True
        self._service_thread.start()

    def stop(self):
        self.logger.info("Setting the shutdown event.")
        self._shutdown_event.set()

    def _serve(self):
        """Runs in the background thread waiting on proxy connections and executing their requests.
        """
        self.logger.info("Serving")
        timeout_s = 1

        while not self._shutdown_event.is_set():

            # We want to make sure that no new connections are added when we are waiting.
            #
            with self._proxy_connections_lock:
                ready_connections = connection.wait(self._proxy_connections, timeout=timeout_s)

            if not ready_connections:
                # We reached timeout. Let's check the shutdown_event and wait again. It's now that some other connections
                # could be added, though I'm not sure why we would ever do that.
                #
                continue

            # OK, we have some messages let's see if we can process them.
            #
            self.logger.info(f"{len(ready_connections)} are ready.")
            for conn in ready_connections:
                try:
                    request = conn.recv()
                    response = self._process_request(request=request)
                    conn.send(response)
                except EOFError:
                    # Connection was closed. TODO: remove that connection from the list
                    self.logger.info("Connection was closed.")
                    self._remove_closed_proxy_connection(connection_id=id(conn))

        # We are shutting down. Nothing to do here, other than maybe closing all the connections.
        #
        self.logger.info("Shutdown event set. Closing proxy connections.")
        with self._proxy_connections_lock:
            for conn in self._proxy_connections:
                conn.close()
            self._proxy_connections = []

    def _remove_closed_proxy_connection(self, connection_id):
        with self._proxy_connections_lock:
            connection_idx = None
            for i, conn in enumerate(self._proxy_connections):
                if id(conn) == connection_id:
                    connection_idx = i
                    break
            if connection_idx is not None:
                self.logger.info(f"Removing connection {connection_id}.")
                self._proxy_connections.pop(connection_idx)

    @request_handler()
    def _process_request(self, request: Request):
        if isinstance(request, TakeDataSetOwnershipRequest):
            return self._process_take_data_set_ownership_request(request=request)
        elif isinstance(request, UnlinkDataSetRequest):
            return self._process_unlink_data_set_request(request=request)
        elif isinstance(request, CreateDataSetRequest):
            return self._process_create_data_set_request(request)
        else:
            raise TypeError(f"Unknown request type: {str(type(request))}")

    def _process_take_data_set_ownership_request(self, request: TakeDataSetOwnershipRequest) -> Response:
        self.logger.info(f"Processing request {request.request_id}. Taking ownership of data set {request.data_set_info.data_set_id}")
        self.data_set_store.connect_to_data_set(data_set_info=request.data_set_info)
        return Response(success=True, request_id=request.request_id)

    def _process_unlink_data_set_request(self, request: UnlinkDataSetRequest) -> Response:
        self.logger.info(f"Processing {request.__class__.__name__} {request.request_id}")
        self.data_set_store.unlink_data_set(data_set_info=request.data_set_info)
        return Response(success=True, request_id=request.request_id)

    def _process_create_data_set_request(self, request: CreateDataSetRequest):
        self.logger.info(f"Processing {request.__class__.__name__}. Creating data set {request.data_set_info.data_set_id}")
        data_set = self.data_set_store.create_data_set(data_set_info=request.data_set_info)
        return CreateDataSetResponse(request_id=request.request_id, data_set_info=data_set.get_data_set_info())



