#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from typing import Dict
from mlos.OptimizerEvaluationTools.OptimumOverTime import OptimumOverTime
from mlos.Optimizers.RegressionModels.RegressionModelFitState import RegressionModelFitState
from mlos.Spaces import Point


class OptimizerEvaluationReport:
    """Contains all information gathered during an optimizer evaluation run.

    This includes:
        * optimizer configuration
        * objective function configuration
        * serialized optimizer (with random seeds, and all observations)
        * serialized objective function (with random seeds)
        * evaluation parameters:
            * num optimization iterations
            * evaluation frequency
        * optimizer's regression model goodness of fit metrics over time
        * optima over time for the following definitions:
            * best observation
            * best predicted value for observed config
            * best upper confidence bound on a 99% confidence interval for an observed config
            * best lower confidence bound on a 99% confidence interval for an observed config
    """

    def __init__(
        self,
        optimizer_configuration: Point,
        objective_function_configuration: Point,
        pickled_optimizer: str,
        pickled_objective_function: str,
        num_optimization_iterations: int,
        evaluation_frequency: int,
        regression_model_goodness_of_fit_state: RegressionModelFitState,
        optima_over_time: Dict[str, OptimumOverTime]
    ):
        ...
