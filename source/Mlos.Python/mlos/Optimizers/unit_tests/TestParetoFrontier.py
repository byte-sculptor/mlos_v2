#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#

from mlos.Optimizers.ParetoFrontier import ParetoFrontier
from mlos.Optimizers.OptimizationProblem import OptimizationProblem, Objective
from mlos.Spaces import SimpleHypergrid, ContinuousDimension, DiscreteDimension

class TestParetoFrontier:
    """Tests if the ParetoFrontier works."""

    def test_basic_functionality_on_2d_objective_space(self):
        """Basic sanity check. Mainly used to help us develop the API.
        """

        # Let's just create a bunch of random points, build a pareto frontier
        # and verify that the invariants hold.
        #
        parameter_space = SimpleHypergrid(
            name='params',
            dimensions=[
                ContinuousDimension(name='x1', min=0, max=10)
            ]
        )

        objective_space = SimpleHypergrid(
            name='objectives',
            dimensions=[
                ContinuousDimension(name='y1', min=0, max=10),
                ContinuousDimension(name='y2', min=0, max=10)
            ]
        )

        optimization_problem = OptimizationProblem(
            parameter_space=parameter_space,
            objective_space=objective_space,
            objectives=[
                Objective(name='y1', minimize=False),
                Objective(name='y2', minimize=False)
            ]
        )
