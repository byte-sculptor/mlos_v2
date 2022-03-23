from abc import ABC, abstractmethod
import logging
from typing import List

from mlos.Logger import create_logger
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel
from mlos.Optimizers.RegressionModels.RegressionModelFitState import RegressionModelFitState
from mlos.Spaces import Hypergrid, Point
from mlos.Spaces.HypergridAdapters import HierarchicalToFlatHypergridAdapter
from mlos.Tracer import trace


class EnsembleRegressionModelBase(RegressionModel):
    """ Base class for ensemble models.

    So far we've had the HomogeneousRandomForestRegressionModel as the only ensemble model.
    However, a lot of ideas there are general and can be used by other regression models,
    such as a bootstrapped ensemble lasso cv regression model.

    """

    @trace()
    def __init__(
        self,
        model_type: type,
        model_config: Point,
        input_space: Hypergrid,
        output_space: Hypergrid,
        fit_state: RegressionModelFitState = None,
        logger: logging.Logger = None
    ) -> None:
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger

        RegressionModel.__init__(
            self,
            model_type=model_type,
            model_config=model_config,
            input_space=input_space,
            output_space=output_space,
            fit_state=fit_state
        )
        assert len(self.target_dimension_names) == 1, "Single target predictions for now."

        self._input_space_adapter = HierarchicalToFlatHypergridAdapter(adaptee=self.input_space)

        self._regressors: List[RegressionModel] = []
        self._create_regressors()
        self._trained = False


    @abstractmethod
    def _create_regressors(self) -> None:
        raise NotImplementedError
