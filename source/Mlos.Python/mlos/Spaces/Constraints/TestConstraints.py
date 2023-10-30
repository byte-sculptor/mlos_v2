

import pytest

from mlos.OptimizerEvaluationTools.ObjectiveFunctionFactory import (
    ObjectiveFunctionFactory,
    objective_function_config_store
)
from mlos.Spaces.Constraints.Constraint import Constraint
from mlos.Spaces.SimpleHypergrid import SimpleHypergrid


class TestConstraints:
    """
    
    We'll need a lot of test cases:
        1. Not nested constraints.
        2. Nested constraints.
        3. Constraints on SimpleHypergrids
        4. Constraints on Adapters
    """

    @classmethod
    def setup_class(cls):
        cls.three_level_quadratic = ObjectiveFunctionFactory.create_objective_function(
            objective_function_config=objective_function_config_store.get_config_by_name(
                name="three_level_quadratic"
            )
        )

        cls.flower = ObjectiveFunctionFactory.create_objective_function(
            objective_function_config=objective_function_config_store.get_config_by_name(
                name="flower"
            )
        )

        cls.multi_objective_waves = ObjectiveFunctionFactory.create_objective_function(
            objective_function_config=objective_function_config_store.get_config_by_name(
                name="multi_objective_waves_3_params_2_objectives_half_pi_phase_difference"
            )
        )


    def test_valid_constraints(self):
        constraint = Constraint()

    def test_constraints_on_adapted_space(self):
        # TODO: add adapters and make sure they work.
        