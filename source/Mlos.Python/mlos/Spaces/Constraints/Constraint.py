import ast

from dataclasses import dataclass
from numbers import Number
from typing import Any, Optional

from mlos.Spaces import Hypergrid, Point
from mlos.Spaces.Constraints.ConstraintValidator import ConstraintValidator

# Conceivably some auto-generated constraint strings could be huge,
# for now we limit it to 8kB arbitrarily, to avoid parsing huge piles
# of garbage.
MAX_CONSTRAINT_EXPRESSION_LENGTH = 8 * 1024


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    expression: str
    none_ok: bool = False
    none_val: Any = None

    def __post_init__(self):
        assert isinstance(self.expression, str)
        assert len(self.expression) < MAX_CONSTRAINT_EXPRESSION_LENGTH

        if self.none_ok:
            assert isinstance(self.none_val, Number)


class Constraint:
    """Models a single constraint.

    The constraint originates as a string representing a valid, boolean epression in Python.
    That string is parsed into an AST, validated and transformed in a number of ways.

    Constraints are used primarily to decide whether a given point belongs to a Hypergrid. In that sense
    they could be used to implement a rejection sampling scheme.

    But constraints contain so much internal structure, that conceivably a random Point
    generator could take them into account. For example: if all constraints happen to be linear, then we
    could construct a simplex object and use barycentric coordinates to sample randomly from that simplex.
    The user would decide if the inherent biases are acceptable.

    But we are getting ahead of ourselves, first we need a way to parse, validate, and store constraints.
    """

    def __init__(
        self,
        constraint_spec: ConstraintSpec,
        space: Hypergrid
    ) -> None:
        """Instantiates a Constraint object.

        Args:
            constraint_expression_str (str): a string representing a constraint expression
            space (Hypergrid): space to be constrained. This is needed to validate that all
                dimensions appearing in the constraint_expression_str are available.
        """
        self._constraint_spec: ConstraintSpec = constraint_spec
        self._space: Hypergrid = space
        self._variable_names: Optional[set[str]] = None
        self._constraint_ast: Optional[ast.Expression] = None

        self._validate()



    def _validate(self) -> None:
        """Validates the constraint.

        To be valid a constraint must:
            1. Be no longer than MAX_CONSTRAINT_EXPRESSION_LENGTH
            2. Not contain any function calls - we can relax that one day.
            3. Result in a boolean value (so it must be a boolean expression)
            4. Reference only variables whose names match dimension names match dimensions of self.space

        """
        self._constraint_ast: ast.Expression =  ast.parse(source=self._constraint_spec.expression, mode='eval')
        assert isinstance(self._constraint_ast, ast.Expression)
        validator = ConstraintValidator(
            allowed_variable_names=self._space.dimension_names
        )
        validator.visit(self._constraint_ast)
        self._variable_names = validator.variable_names

    def violated(self, point: Point) -> bool:
        if not self.applicable(point=point):
            return False

        # TODO: add eval
        return False

    def applicable(self, point: Point) -> bool:
        """Checks if the constraint is applicable to this poit.

        For a constraint to be applicable, the point must contain all dimensions
        mentioned by the constraint.

        TODO: allow more versatile treatment of missing dimensions. e.g. let the user
        TODO: set the default value if dimension is not present
        """
        return True

