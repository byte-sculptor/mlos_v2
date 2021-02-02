#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import math
import pytest

from mlos.OptimizerEvaluationTools.SyntheticFunctions.EnvelopedWaves import EnvelopedWaves, enveloped_waves_config_space
from mlos.Spaces import Point

class TestEnvelopedWaves:

    def test_enveloped_waves(self):
        function_config = Point(
            num_params=1,
            num_periods=1,
            amplitude=1,
            vertical_shift=1,
            phase_shift=0,
            period=2 * math.pi,
            envelope_type="none"
        )

        assert function_config in enveloped_waves_config_space
        objective_function = EnvelopedWaves(function_config)
        random_params_df = objective_function.parameter_space.random_dataframe(100)
        objectives_df = objective_function.evaluate_dataframe(random_params_df)
        assert ((objectives_df['y'] <= 2) & (objectives_df['y'] >= 0)).all()

    def test_random_configs(self):
        for _ in range(100):
            function_config = enveloped_waves_config_space.random()
            objective_function = EnvelopedWaves(function_config)
            random_params_df = objective_function.parameter_space.random_dataframe(100)
            objectives_df = objective_function.evaluate_dataframe(random_params_df)
