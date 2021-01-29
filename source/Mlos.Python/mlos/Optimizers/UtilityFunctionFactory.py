#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
from mlos.Optimizers.OptimizationProblem import OptimizationProblem
from mlos.Optimizers.ParetoFrontier import ParetoFrontier
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel
from mlos.Spaces import Point

class UtilityFunctionFactory:
    """Creates specialized instances of the abstract base class: UtilityFunction.

    """

    @classmethod
    def create_utility_function(
        cls,
        utility_function_config: Point,
        surrogate_model: RegressionModel,
        optimization_problem: OptimizationProblem,
        pareto_frontier: ParetoFrontier = None,
        logger=None
    ):

