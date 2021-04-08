#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from contextlib import contextmanager
import multiprocessing
from typing import Iterable, Iterator
from uuid import UUID

from mlos.Logger import create_logger
from mlos.MlosOptimizationServices.BayesianOptimizerStore.BayesianOptimizerStoreBase import BayesianOptimizerStoreBase
from mlos.Optimizers.BayesianOptimizer import BayesianOptimizer


class BayesianOptimizerInMemoryStore(BayesianOptimizerStoreBase):
    """The simplest implementation of the BayesianOptimizerStore simply keeps the optimizers in memory.

    Goal
        This is the first step towards separating the optimizer storage from the optimizer microservice. The main
        goal of this class is to maintain existing functionality, as well as to start building the OptimizerStore
        API.

    Functionality
        Stores optimizers in memory and retrieves them to help satisfy the requests handled by the Optimizer
        Microservice.

        All optimizers have an associated lock which is used to provide exlcusive access to the optimizer in
        a multi-threaded set up. A reader-writer lock would have been more efficient, and keeping the optimizer
        versioned might also help.

    """

    def __init__(self, logger=None):
        self._optimizers_by_id = dict()
        self._ordered_optimizer_ids = []
        self._lock_manager = multiprocessing.Manager()
        self._optimizer_locks_by_optimizer_id = dict()
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger
        self.logger.info(f"{self.__class__.__name__} initialized.")


    @contextmanager
    def exclusive_optimizer(self, optimizer_id: UUID, optimizer_version: int = None) -> Iterator[BayesianOptimizer]:
        """ Context manager to acquire the optimizer lock and yield the corresponding optimizer.

        This makes sure that:
            1. The lock is acquired before any operation on the optimizer commences.
            2. The lock is released even if exceptions are flying.


        :param optimizer_id:
        :param optimizer_version:
        :return:
        :raises: KeyError if the optimizer_id was not found.
        """
        with self._optimizer_locks_by_optimizer_id[optimizer_id]:
            yield self._optimizers_by_id[optimizer_id]

    def list_optimizers(self) -> Iterable[BayesianOptimizer]:
        # TODO: should we be locking them here?
        for optimizer_id in self._ordered_optimizer_ids:
            yield self._optimizers_by_id[optimizer_id]

