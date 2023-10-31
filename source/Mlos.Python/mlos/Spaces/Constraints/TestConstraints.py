import math

import pytest

from mlos.Exceptions import InvalidConstraintException, FailedToGenerateRandomConstrainedPointException
from mlos.Spaces.Constraints.Constraint import ConstraintSpec
from mlos.Spaces import CategoricalDimension, ContinuousDimension, DiscreteDimension, OrdinalDimension, Point, SimpleHypergrid


class TestConstraints:
    """

    We'll need a lot of test cases:
        1. Not nested constraints.
        2. Nested constraints.
        3. Constraints on SimpleHypergrids
        4. Constraints on Adapters
    """

    @classmethod
    def _make_small_circle(cls) -> SimpleHypergrid:
        return SimpleHypergrid(
            name="small_circle",
            dimensions=[
                ContinuousDimension(name='radius', min=0, max=5),
                ContinuousDimension(name='theta', min=2, max=2 * math.pi)
            ]
        )
    @classmethod
    def _make_large_circle(cls) -> SimpleHypergrid:
        return SimpleHypergrid(
            name="large_circle",
            dimensions=[
                ContinuousDimension(name='radius', min=0, max=10),
                ContinuousDimension(name='theta', min=2, max=2 * math.pi)
            ]
        )


    def test_donut(self):
        donut_param_space = SimpleHypergrid(
            name="donut",
            dimensions=[
                ContinuousDimension(name='outer_radius', min=0, max=10, include_min=False),
                ContinuousDimension(name='inner_radius', min=0, max=10, include_max=False),
            ],
            constraints=[
                ConstraintSpec(name="radius_ordering", expression="inner_radius < outer_radius")
            ]
        )

        valid_point = Point(inner_radius=1, outer_radius=9)
        invalid_point = Point(inner_radius=9, outer_radius=1)

        assert valid_point in donut_param_space
        assert invalid_point not in donut_param_space

    def test_nested_donuts(self):
        nested_donuts_param_space = SimpleHypergrid(
            name="nested_donuts",
            dimensions=[
                ContinuousDimension(name='large_outer_radius', min=0, max=10, include_min=False, include_max=True),
                ContinuousDimension(name='large_inner_radius', min=0, max=10, include_min=False, include_max=False),
                ContinuousDimension(name='small_outer_radius', min=0, max=10, include_min=False, include_max=False),
                ContinuousDimension(name='small_inner_radius', min=0, max=10, include_min=True, include_max=False),
            ],
        )

        nested_donuts_param_space.add_constraint(ConstraintSpec(name="large_radii", expression="large_inner_radius < large_outer_radius"))
        nested_donuts_param_space.add_constraint(ConstraintSpec(name="all_radii", expression="0 < small_inner_radius <= small_outer_radius <= large_inner_radius <= large_outer_radius < 10"))
        nested_donuts_param_space.add_constraint(ConstraintSpec(name="twice_smaller", expression="large_inner_radius < small_outer_radius * 2 < large_outer_radius"))
        nested_donuts_param_space.add_constraint(ConstraintSpec(name="max_circumference", expression=f"2*{math.pi}*sum([large_outer_radius, large_inner_radius, small_outer_radius, small_inner_radius]) < 30 * 2 * {math.pi}"))
        nested_donuts_param_space.add_constraint(ConstraintSpec(name="max_circumference_2", expression=f"2 * {math.pi} * (large_outer_radius + large_inner_radius + small_outer_radius + small_inner_radius) < 30 * 2 * {math.pi}"))

        with pytest.raises(InvalidConstraintException):
            nested_donuts_param_space.add_constraint(ConstraintSpec(
                name="",
                expression="sum([sqrt(large_outer_radius), abs(-1)]) > 5")
            )

        with pytest.raises(InvalidConstraintException):
            nested_donuts_param_space.add_constraint(
                ConstraintSpec(name="", expression="math.floor(large_outer_radius) > 0")
            )

        with pytest.raises(InvalidConstraintException):
            nested_donuts_param_space.add_constraint(ConstraintSpec(name="", expression="sqrt(large_outer_radius) > 1"))

    def test_donut_membership(self):
        """Tests if points are correctly filtered out."""

        large_circle = SimpleHypergrid(
            name="large_circle",
            dimensions=[
                ContinuousDimension(name="radius", min=5, max=10, include_min=True, include_max=True),
                ContinuousDimension(name="theta", min=-math.pi, max=math.pi, include_min=True, include_max=True)
            ]
        )

        small_circle = SimpleHypergrid(
            name="small_circle",
            dimensions=[
                ContinuousDimension(name="radius", min=0, max=5, include_min=True, include_max=False),
                ContinuousDimension(name="theta", min=-math.pi, max=math.pi, include_min=True, include_max=True)
            ]
        )

        donut = SimpleHypergrid(
            name="donut",
            dimensions=[
                ContinuousDimension(name="x", min=-10, max=10, include_min=True, include_max=True),
                ContinuousDimension(name="y", min=-10, max=10, include_min=True, include_max=True)
            ],
            constraints=[
                ConstraintSpec(name="in_outer", expression="x**2 + y**2 <= 10**2"),
                ConstraintSpec(name="outside_inner", expression="x**2 + y**2 >= 5**2")
            ]
        )

        NUM_POINTS = 1000
        for _ in range(NUM_POINTS):
            point = donut.random()
            polar_point = Point(
                radius=math.sqrt(point.x ** 2 + point.y ** 2),
                theta=math.atan2(point.y, point.x)
            )
            assert polar_point in large_circle, f"{point=} {polar_point=}"
            assert polar_point not in small_circle, f"{point=} {polar_point=}"

        for _ in range(NUM_POINTS):
            polar_point = large_circle.random()
            point = Point(
                x=polar_point.radius * math.cos(polar_point.theta),
                y=polar_point.radius * math.sin(polar_point.theta)
            )
            if polar_point in small_circle:
                assert point not in donut, f"{point=} {polar_point=}"
            else:
                assert point in donut, f"{point=} {polar_point=}"

    def test_failure_to_produce_random_suggestion(self):
        """Some constraints are so stringent, that we cannot produce random suggestions.

        Let's make sure we fail correctly then.
        """
        parameter_space = SimpleHypergrid(
            name="overconstrained",
            dimensions=[
                ContinuousDimension(name="x", min=0, max=100)
            ],
            constraints=[
                ConstraintSpec(name="too_restrictive", expression="x > 200")
            ]
        )

        with pytest.raises(FailedToGenerateRandomConstrainedPointException):
            parameter_space.random()

    def test_failure_to_produce_random_suggestion_nested(self):
        """Some constraints are so stringent, that we cannot produce random suggestions.

        Let's make sure we fail correctly then.
        """
        parameter_space = SimpleHypergrid(
            name="overconstrained",
            dimensions=[
                ContinuousDimension(name="x", min=0, max=100)
            ]
        ).join(
            subgrid=SimpleHypergrid(
                name="nested",
                dimensions=[
                    DiscreteDimension(name="y", min=0, max=100)
                ]
            ).join(
                subgrid=SimpleHypergrid(
                    name="very_nested",
                    dimensions=[CategoricalDimension(name="y", values=[i for i in range(10)])]
                ),
                on_external_dimension=DiscreteDimension(name="y", min=0, max=10)
            ),
            on_external_dimension=ContinuousDimension(name="x", min=0, max=100)
        )

        parameter_space.add_constraint(
                constraint_spec=ConstraintSpec(name="too_strict", expression="nested.very_nested.z + 100 < x")
        )
        parameter_space.add_constraint(
            constraint_spec=ConstraintSpec(name="too_strict", expression="nested.y + 100 < x")
        )

        with pytest.raises(FailedToGenerateRandomConstrainedPointException):
            parameter_space.random()

    def test_constraints_on_nested_space(self):
        parameter_space = SimpleHypergrid(
            name="params",
            dimensions=[
                ContinuousDimension(name="x", min=0, max=100),
                CategoricalDimension(name="use_left", values=[False, True]),
                CategoricalDimension(name="use_right", values=[False, True])
            ]
        ).join(
            subgrid=SimpleHypergrid(
                name="left",
                dimensions=[
                    DiscreteDimension(name="y", min=0, max=100)
                ]
            ),
            on_external_dimension=CategoricalDimension(name="use_left", values=[True])
        ).join(
            subgrid=SimpleHypergrid(
                name="right",
                dimensions=[
                    OrdinalDimension(name="z", ordered_values=[1, 2, 5])
                ]
            ),
            on_external_dimension=CategoricalDimension(name="use_right", values=[True])
        )

        parameter_space.add_constraint(
            constraint_spec=ConstraintSpec(
                name="max_sum",
                expression="x + left.y + right.z < 100",
                none_ok=True,
                none_val=0
            )
        )

        parameter_space.add_constraint(
            constraint_spec=ConstraintSpec(
                name="min_sum",
                expression="x + left.y + right.z >= 50",
                none_ok=True,
                none_val=0
            )
        )

        for _ in range(100):
            point = parameter_space.random()
            running_sum = point.x
            if point.use_left:
                running_sum += point.left.y
            if point.use_right:
                running_sum += point.right.z

            assert 50 <= running_sum < 100


