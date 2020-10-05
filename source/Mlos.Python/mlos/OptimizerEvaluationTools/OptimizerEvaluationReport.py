#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from typing import Dict, List
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
        * execution trace as captured by the mlos.Tracer
    """

    def __init__(
        self,
        optimizer_configuration: Point = None,
        objective_function_configuration: Point = None,
        pickled_optimizer: str = None,
        pickled_objective_function: str = None,
        num_optimization_iterations: int = None,
        evaluation_frequency: int = None,
        regression_model_goodness_of_fit_state: RegressionModelFitState = None,
        optima_over_time: Dict[str, OptimumOverTime] = None,
        execution_trace: List[Dict[str, object]] = None
    ):
        self.optimizer_configuration = optimizer_configuration
        self.objective_function_configuration = objective_function_configuration
        self.pickled_optimizer = pickled_optimizer
        self.pickled_objective_function = pickled_objective_function
        self.num_optimization_iterations = num_optimization_iterations
        self.evaluation_frequency = evaluation_frequency
        self.regression_model_goodness_of_fit_state = regression_model_goodness_of_fit_state
        self.optima_over_time = optima_over_time
        self.execution_trace = execution_trace
