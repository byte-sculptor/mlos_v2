#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
import math
import pandas as pd

from mlos.OptimizerEvaluationTools.ObjectiveFunctionBase import ObjectiveFunctionBase
from mlos.Spaces import CategoricalDimension, ContinuousDimension, DiscreteDimension, Point, SimpleHypergrid

enveloped_waves_config_space = SimpleHypergrid(
    name="enveloped_waves_config",
    dimensions=[
        DiscreteDimension(name="num_params", min=1, max=100),
        ContinuousDimension(name="num_periods", min=1, max=100),
        ContinuousDimension(name="amplitude", min=0, max=10, include_min=False),
        ContinuousDimension(name="vertical_shift", min=0, max=10),
        ContinuousDimension(name="phase_shift", min=0, max=2 * math.pi),
        ContinuousDimension(name="period", min=0, max=10 * math.pi, include_min=False),
        CategoricalDimension(name="envelope_type", values=["linear", "quadratic", "sine"])
    ]
).join(
    subgrid=SimpleHypergrid(
        name="linear_envelope_config",
        dimensions=[
            ContinuousDimension(name="slope", min=-100, max=100)
        ]
    ),
    on_external_dimension=CategoricalDimension(name="envelope_type", values=["linear"])
).join(
    subgrid=SimpleHypergrid(
        name="quadratic_envelope_config",
        dimensions=[
            ContinuousDimension(name="a", min=-100, max=100),
            ContinuousDimension(name="p", min=-100, max=100),
            ContinuousDimension(name="q", min=-100, max=100),
        ]
    ),
    on_external_dimension=CategoricalDimension(name="envelope_type", values=["quadratic"])
).join(
    subgrid=SimpleHypergrid(
        name="sine_envelope_config",
        dimensions=[
            ContinuousDimension(name="amplitude", min=0, max=10, include_min=False),
            ContinuousDimension(name="phase_shift", min=0, max=2 * math.pi),
            ContinuousDimension(name="period", min=0, max=100 * math.pi, include_min=False),
        ]
    ),
    on_external_dimension=CategoricalDimension(name="envelope_type", values=["sine"])
)

class EnvelopedWaves(ObjectiveFunctionBase):
    """Sum of sine waves enveloped by another function, either linear, quadratic or another sine wave.

    An enveloped sine wave produces complexity for the optimizer that allows us evaluate its behavior on non-trivial problems.

    Simultaneously, sine waves have the advantage over normal polynomials that:
        1. They have well known optima - even when we envelop the function with another sine wave, as long as we keep their frequencies
            harmonic, we can know exactly where the optimum is.
        2. The cannot be well approximated by a polynomial (Taylor expansion is accurate only locally).
        3. For multi-objective problems, we can manipulate the phase shift of each objective to cotrol the shape of the pareto frontier.

    How the function works?
    -----------------------
    When creating the function we specify:
    1. Amplitute, vertical_shift, phase-shift and period of the sine wave.
    2. Envelope:
        1. Linear: slope (no need y_intercept as it's included in the vertical_shift)
        2. Quadratic: a, p, q
        3. Sine: again amplitude, phase shift, and period (no need to specify the vertical shift again.

    The function takes the form:

        y(x) = sum(
            amplitude * sin((x_i - phase_shift) / period) * envelope(x_i) + vertical_shift
            for x_i
            in x
        )

        WHERE:
            envelope(x_i) = envelope.slope * x_i + envelope.y_intercept
            OR
            envelope(x_i) = a * (x_i - p) * (x_i - q)
            OR
            envelope(x_i) = envelope.amplitude * sin((x_i - envelope.phase_shift) / envelope.period)

    """

    def __init__(self, objective_function_config: Point = None):
        ObjectiveFunctionBase.__init__(self, objective_function_config)
        self._parameter_space = SimpleHypergrid(
            name="domain",
            dimensions=[
                ContinuousDimension(name=f"x_{i}", min=0, max=objective_function_config.num_periods * objective_function_config.period)
                for i in range(self.objective_function_config.num_params)
            ]
        )

        self._output_space = SimpleHypergrid(
            name="range",
            dimensions=[
                ContinuousDimension(name="y", min=-math.inf, max=math.inf)
            ]
        )

    def evaluate_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        ...




