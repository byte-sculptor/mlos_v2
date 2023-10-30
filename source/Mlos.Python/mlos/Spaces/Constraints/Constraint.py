import ast

from mlos.Spaces import Hypergrid

# Conceivably some auto-generated constraint strings could be huge,
# for now we limit it to 8kB arbitrarily, to avoid parsing huge piles
# of garbage.
MAX_CONSTRAINT_EXPRESSION_LENGTH = 8 * 1024

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
        constraint_expression_str: str,
        space: Hypergrid
    ) -> None:
        """Instantiates a Constraint object.

        Args:
            constraint_expression_str (str): a string representing a constraint expression
            space (Hypergrid): space to be constrained. This is needed to validate that all 
                dimensions appearing in the constraint_expression_str are available.
        """
        self._constraint_expression_str: str = constraint_expression_str
        self._space: Hypergrid = space
        

    def _validate(self) -> None:
        """Validates the constraint.

        To be valid a constraint must:
            1. Be no longer than MAX_CONSTRAINT_EXPRESSION_LENGTH
            2. Not contain any function calls - we can relax that one day.
            3. Result in a boolean value (so it must be a boolean expression)
            4. Reference only variables whose names match dimension names match dimensions of self.space

        """

        # TODO: make this a custom exception type
        assert len(self._constraint_expression_str) <= MAX_CONSTRAINT_EXPRESSION_LENGTH
        constraint_tree: ast.Expression =  ast.parse(source=self._constraint_expression_str, mode='eval')