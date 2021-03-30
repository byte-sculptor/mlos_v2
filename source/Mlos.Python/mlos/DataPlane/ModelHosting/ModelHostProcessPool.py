#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from multiprocessing import Event, Process, Queue
from threading import Thread
import time

from mlos.DataPlane.SharedMemoryDataSets import SharedMemoryDataSetService
from mlos.DataPlane.ModelHosting.SharedMemoryModelHost import start_shared_memory_model_host
from mlos.Logger import create_logger


class ModelHostProcessPool:
    """The role of this class is to maintain a pool of model host processes.

    These processes crash occasionally as they run out of memory. I guess fragmentation is to blame as
    their memory consumption in the Task Manager hoovers between 100-300MB and total system memory
    utilization is low, except the page file keeps on growing... Anyway, the problem doesn't happen
    when trying to allocate a SharedMemory block, but rather when pandas/numpy try to manipulate the data
    and need space for intermittent calculation results.

    So role of this class is to maintain a pool of processes. If a host process crashes, it should
    be logged and a new should be spawned in its place. The thinking is that with the process crashing
    the OS reclaims all of its memory and the new process starts with low fragmentation.

    This is obviously not a fix, but rather a workaround. A fix would be to either manage memory better manually
    or to somehow reconfiugre Python's GC to curb fragmentation.

    """

    def __init__(
        self,
        shared_memory_data_set_service: SharedMemoryDataSetService,
        request_queue: Queue,
        response_queue: Queue,
        max_num_processes: int,
        logger=None
    ):
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger
        self.shared_memory_data_set_service = shared_memory_data_set_service
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.shutdown_event = Event()
        self.max_num_processes = max_num_processes
        self._daemon_thread: Thread = None

    def start(self):
        self._daemon_thread = Thread(target=self._launch_and_monitor_host_processes)
        self._daemon_thread.daemon = True
        self._daemon_thread.start()

    def stop(self):
        self.logger.info("Setting the shutdown event.")
        self.shutdown_event.set()
        self.logger.info("Waiting for daemon thread to exit.")
        self._daemon_thread.join()

    def _launch_and_monitor_host_processes(self):
        model_host_processes = []

        while not self.shutdown_event.is_set():
            time.sleep(1)

            while len(model_host_processes) < self.max_num_processes:
                service_connection = self.shared_memory_data_set_service.get_new_proxy_connection()
                model_host_process = Process(
                    target=start_shared_memory_model_host,
                    kwargs=dict(
                        request_queue=self.request_queue,
                        response_queue=self.response_queue,
                        shutdown_event=self.shutdown_event,
                        data_set_store_service_connection=service_connection
                    )
                )
                model_host_process.start()
                model_host_processes.append(model_host_process)

            # Now let's go over all processes and check if they are alive.
            #
            dead_process_ids = []
            for i, process in enumerate(model_host_processes):
                if not process.is_alive():
                    self.logger.info(f'Process {process.pid} exited with exit code: {process.exitcode}')
                    dead_process_ids.append(i)

            for process_index in reversed(dead_process_ids):
                model_host_processes.pop(process_index)

        for process in model_host_processes:
            self.logger.info(f"Joining process {process.pid}.")
            process.join()
            self.logger.info(f"Process {process.pid} exited with exit code: {process.exitcode}")






