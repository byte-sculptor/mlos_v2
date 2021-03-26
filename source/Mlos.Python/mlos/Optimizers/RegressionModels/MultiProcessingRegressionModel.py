#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from mlos.Logger import create_logger
from mlos.Optimizers.RegressionModels.DecisionTreeRegressionModel import DecisionTreeRegressionModel, decision_tree_config_store
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel
from mlos.Spaces import Hypergrid, Point
from uuid import UUID, uuid4


class MultiProcessingRegressionModel(RegressionModel):
    """Wraps any regression model and makes it possible to host it in a different (worker) process.

    Ideally, we can rewrite the RegressionModel API to use datasets instead of data frames. But for now, we just wrap any
    regression model and we forward all calls to .fit() and to .predict() to worker processes.
    """

    def __init__(
        self,
        model_type: type,
        model_config: Point,
        input_space: Hypergrid,
        output_space: Hypergrid,
        logger=None,

    ):
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger

        assert model_type is DecisionTreeRegressionModel
        assert model_config in decision_tree_config_store.parameter_space

        RegressionModel.__init__(
            self,
            model_type=type(self),
            model_config=model_config,
            input_space=input_space,
            output_space=output_space
        )

        self._trained = False


    @property
    def trained(self):
        return self._trained

    def fit(self, feature_values_pandas_frame, target_values_pandas_frame, iteration_number):



    def predict(self, feature_values_pandas_frame, include_only_valid_rows=True):
